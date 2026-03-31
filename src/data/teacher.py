"""Generate teacher reasoning traces using Gemini 2.5 Flash via Google AI Studio."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.data.gsm8k import load_gsm8k
from src.data.math_500 import (
    extract_hash_answer,
    extract_predicted_answer,
    grade_answer as grade_answer_math,
    load_math_500,
)

load_dotenv()

TEACHER_PROMPT = (
    "You are an expert math tutor. Solve the following math problem step by step.\n"
    "Show your reasoning clearly, then on the LAST line write ONLY `#### <final answer>`.\n\n"
    "Problem: {question}"
)


def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_MODEL_API_KEY not found in .env")
    return genai.Client(api_key=api_key)


def generate_teacher_trace(
    client: genai.Client,
    question: str,
    model: str | None = None,
    max_retries: int = 5,
) -> dict:
    model = model or os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    prompt = TEACHER_PROMPT.format(question=question)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = response.text.strip()
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 15 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1})...", flush=True)
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError(f"Failed after {max_retries} retries")

    lines = text.strip().split("\n")
    reasoning = "\n".join(lines[:-1]).strip()
    answer_line = lines[-1].strip() if lines else ""

    return {
        "full_response": text,
        "reasoning": reasoning,
        "answer_line": answer_line,
    }


def generate_teacher_dataset(
    output_path: str = "data/teacher_traces.jsonl",
    num_problems: int = 100,
    model: str | None = None,
    delay: float = 13.0,
) -> Path:
    """Generate teacher traces for MATH-500 problems and save as JSONL.

    Respects free-tier rate limits (5 RPM) and supports resume by appending
    to existing output files.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if out.exists():
        with open(out) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing.add(entry.get("question", ""))
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: {len(existing)} traces already exist", flush=True)

    client = get_client()
    dataset = load_math_500("test", subset_size=num_problems)

    correct_count = 0
    total_done = len(existing)

    with open(out, "a") as f_out:
        for idx, item in enumerate(dataset, start=1):
            if item["question"] in existing:
                correct_count += 1  # approximate
                continue

            try:
                trace = generate_teacher_trace(client, item["question"], model=model)
                predicted = extract_hash_answer(trace["answer_line"])
                if predicted is None:
                    predicted = extract_predicted_answer(trace["full_response"])
                correct = grade_answer_math(predicted, item["answer_number"])
                if correct:
                    correct_count += 1

                entry = {
                    "question": item["question"],
                    "gold_answer": item["answer_number"],
                    "teacher_reasoning": trace["reasoning"],
                    "teacher_answer_line": trace["answer_line"],
                    "teacher_predicted": predicted,
                    "teacher_correct": correct,
                    "full_response": trace["full_response"],
                }
                f_out.write(json.dumps(entry, default=str) + "\n")
                f_out.flush()
                total_done += 1
                status = "OK" if correct else "WRONG"
                print(f"[{total_done}/{num_problems}] {status} pred={predicted} gold={item['answer_number']}", flush=True)

            except Exception as e:
                print(f"[{idx}/{num_problems}] ERROR: {e}", flush=True)
                entry = {
                    "question": item["question"],
                    "gold_answer": item["answer_number"],
                    "error": str(e),
                }
                f_out.write(json.dumps(entry, default=str) + "\n")
                f_out.flush()
                total_done += 1

            if delay > 0:
                time.sleep(delay)

    print(f"\nDone. ~{correct_count}/{num_problems} correct. Saved to {out}", flush=True)
    return out


def generate_teacher_dataset_from_gsm8k(
    output_path: str = "data/teacher_traces.jsonl",
    num_problems: int = 100,
) -> Path:
    """Use GSM8K's built-in step-by-step solutions as teacher traces.

    GSM8K solutions are human-written, high-quality reasoning chains — a valid
    and common approach for SFT before RL/DPO.  Falls back to this when API
    rate limits are hit.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_gsm8k("train", subset_size=num_problems)
    correct_count = 0

    with open(out, "w") as f:
        for idx, item in enumerate(dataset, start=1):
            full_answer = item["full_answer"]
            parts = full_answer.split("####")
            reasoning = parts[0].strip() if parts else full_answer
            answer_line = parts[1].strip() if len(parts) > 1 else str(item["answer_number"])

            entry = {
                "question": item["question"],
                "gold_answer": item["answer_number"],
                "teacher_reasoning": reasoning,
                "teacher_answer_line": answer_line,
                "teacher_predicted": item["answer_number"],
                "teacher_correct": True,
                "full_response": f"{reasoning}\n{answer_line}",
                "source": "gsm8k_gold",
            }
            f.write(json.dumps(entry, default=str) + "\n")
            correct_count += 1

    print(f"Done. {correct_count}/{num_problems} traces from GSM8K gold solutions. Saved to {out}", flush=True)
    return out


def generate_teacher_dataset_from_math_500(
    output_path: str = "data/teacher_traces.jsonl",
    num_problems: int = 100,
) -> Path:
    """Use MATH-500 dataset's `solution`/`answer` fields as teacher traces."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_math_500("test", subset_size=num_problems)
    correct_count = 0

    with open(out, "w") as f:
        for item in dataset:
            reasoning = (item.get("solution") or "").strip()
            full_answer = item.get("full_answer") or item.get("answer_number") or ""
            answer_line = f"#### {full_answer}".strip()

            entry = {
                "question": item["question"],
                "gold_answer": item["answer_number"],
                "teacher_reasoning": reasoning,
                "teacher_answer_line": answer_line,
                "teacher_predicted": item["answer_number"],
                "teacher_correct": True,
                "full_response": f"{reasoning}\n{answer_line}".strip(),
                "source": "math_500_gold",
            }
            f.write(json.dumps(entry, default=str) + "\n")
            correct_count += 1

    print(f"Done. {correct_count}/{num_problems} traces from MATH-500 gold. Saved to {out}", flush=True)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument("--output", default="data/teacher_traces.jsonl")
    parser.add_argument("--delay", type=float, default=15.0)
    parser.add_argument("--source", choices=["api", "gsm8k", "math_500"], default="math_500",
                        help="Use 'api' for Gemini, 'gsm8k' for built-in solutions")
    args = parser.parse_args()

    if args.source == "api":
        generate_teacher_dataset(
            output_path=args.output,
            num_problems=args.num_problems,
            delay=args.delay,
        )
    elif args.source == "gsm8k":
        generate_teacher_dataset_from_gsm8k(
            output_path=args.output,
            num_problems=args.num_problems,
        )
    else:
        generate_teacher_dataset_from_math_500(
            output_path=args.output,
            num_problems=args.num_problems,
        )
