"""SFT: Fine-tune Qwen2.5-0.5B on teacher-generated reasoning traces via LoRA."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_teacher_traces(path: str, correct_only: bool = True) -> list[dict]:
    """Load JSONL teacher traces, optionally filtering to correct ones only."""
    entries = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if "error" in entry:
                continue
            if correct_only and not entry.get("teacher_correct", False):
                continue
            entries.append(entry)
    return entries


def traces_to_chat_dataset(traces: list[dict]) -> Dataset:
    """Convert teacher traces into chat-format records for SFT.

    Each record has a 'text' field with the full conversation including system
    prompt, question, and teacher reasoning + answer.
    """
    records = []
    for t in traces:
        reasoning = t.get("teacher_reasoning", "")
        answer_line = t.get("teacher_answer_line", "")
        assistant_text = f"{reasoning}\n{answer_line}".strip()

        text = (
            f"<|im_start|>system\n"
            f"You are a math problem solver. Solve problems step by step, "
            f"then give the final numerical answer on the last line.<|im_end|>\n"
            f"<|im_start|>user\n{t['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
        )
        records.append({"text": text})
    return Dataset.from_list(records)


def run_sft(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    teacher_data_path: str = "data/teacher_traces.jsonl",
    output_dir: str = "checkpoints/sft",
    num_epochs: int = 3,
    lr: float = 2e-5,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 1024,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
) -> Path:
    """Run SFT on teacher traces and save the LoRA adapter."""
    print(f"Loading teacher traces from {teacher_data_path}", flush=True)
    traces = load_teacher_traces(teacher_data_path, correct_only=True)
    print(f"  {len(traces)} correct traces loaded", flush=True)
    if not traces:
        raise RuntimeError("No correct teacher traces found. Check teacher data.")

    dataset = traces_to_chat_dataset(traces)
    print(f"  Dataset size: {len(dataset)}", flush=True)

    print(f"Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    out_path = Path(output_dir)
    training_args = SFTConfig(
        output_dir=str(out_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="epoch",
        max_length=max_seq_length,
        bf16=torch.backends.mps.is_available() or torch.cuda.is_available(),
        fp16=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    print("Starting SFT training...", flush=True)
    trainer.train()

    final_path = out_path / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"SFT model saved to {final_path}", flush=True)
    return final_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--teacher-data", default="data/teacher_traces.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    run_sft(
        model_name=args.model,
        teacher_data_path=args.teacher_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
    )
