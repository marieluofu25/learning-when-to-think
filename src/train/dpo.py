"""DPO: Train the SFT model to learn when to stop via preference pairs."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.data.math_500 import load_math_500, extract_predicted_answer, grade_answer
from src.policy.adaptive import cot_generate


def generate_preference_pairs(
    model,
    tokenizer,
    dataset: list[dict],
    samples_per_problem: int = 4,
    temperature: float = 0.8,
) -> list[dict]:
    """Generate (chosen, rejected) pairs from the SFT model.

    - chosen: response that gets the answer correct (or closest)
    - rejected: response that gets it wrong or terminates prematurely
    """
    pairs = []
    total = len(dataset)

    for idx, item in enumerate(dataset, start=1):
        correct_responses = []
        wrong_responses = []

        for _ in range(samples_per_problem):
            out = cot_generate(
                model, tokenizer, item["question"],
                max_tokens=512, temperature=temperature,
            )
            predicted = extract_predicted_answer(out["answer_text"])
            is_correct = grade_answer(predicted, item["answer_number"])

            if is_correct:
                correct_responses.append(out["answer_text"])
            else:
                wrong_responses.append(out["answer_text"])

        if correct_responses and wrong_responses:
            chosen = correct_responses[0]
            rejected = wrong_responses[0]

            prompt = (
                f"<|im_start|>system\n"
                f"You are a math problem solver. Solve problems step by step, "
                f"then give the final answer in `#### <final answer>` format on the last line.<|im_end|>\n"
                f"<|im_start|>user\n{item['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            pairs.append({
                "prompt": prompt,
                "chosen": chosen + "<|im_end|>",
                "rejected": rejected + "<|im_end|>",
            })

        if idx % 5 == 0 or idx == total:
            print(f"[pairs] {idx}/{total} — {len(pairs)} valid pairs so far", flush=True)

    return pairs


def run_dpo(
    sft_checkpoint: str = "checkpoints/sft/final",
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "checkpoints/dpo",
    num_problems: int = 80,
    samples_per_problem: int = 4,
    num_epochs: int = 2,
    lr: float = 5e-6,
    beta: float = 0.1,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    max_length: int = 1024,
    max_prompt_length: int = 256,
) -> Path:
    """Generate preference pairs from SFT model, then run DPO training."""
    sft_path = Path(sft_checkpoint)

    print(f"Loading base model: {base_model_name}", flush=True)
    tokenizer_source = base_model_name
    if sft_path.exists():
        tokenizer_source = str(sft_path)
        print(f"Loading tokenizer from SFT checkpoint: {sft_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32, device_map="auto"
    )
    base_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    if base_vocab_size != tokenizer_vocab_size:
        print(
            f"Resizing base embeddings from {base_vocab_size} to {tokenizer_vocab_size} "
            "to match tokenizer.",
            flush=True,
        )
        base_model.resize_token_embeddings(tokenizer_vocab_size)

    if sft_path.exists():
        print(f"Loading SFT LoRA from {sft_path}", flush=True)
        sft_model = PeftModel.from_pretrained(base_model, str(sft_path))
        sft_model = sft_model.merge_and_unload()
    else:
        print("WARNING: SFT checkpoint not found, using base model", flush=True)
        sft_model = base_model

    pairs_path = Path("data/dpo_pairs.jsonl")
    pairs_path.parent.mkdir(parents=True, exist_ok=True)

    regenerate = True
    pairs: list[dict] = []
    if pairs_path.exists() and pairs_path.stat().st_size > 0:
        print(f"Loading existing pairs from {pairs_path}", flush=True)
        with open(pairs_path) as f:
            for line in f:
                pairs.append(json.loads(line))
        print(f"  Loaded {len(pairs)} pairs", flush=True)

        prompt0 = pairs[0].get("prompt") if pairs else ""
        # If the prompt still uses the old GSM8K-style wording, regenerate.
        regenerate = "#### <final answer>" not in (prompt0 or "")
        if regenerate:
            print("  Existing pairs look GSM8K-style; regenerating for MATH-500...", flush=True)

    if regenerate:
        print("Generating preference pairs from SFT model...", flush=True)
        train_data = load_math_500("test", subset_size=num_problems)
        pairs = generate_preference_pairs(
            sft_model,
            tokenizer,
            train_data,
            samples_per_problem=samples_per_problem,
        )
        print(f"\nTotal valid pairs: {len(pairs)}", flush=True)

        with open(pairs_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"Saved pairs to {pairs_path}", flush=True)

    if len(pairs) < 3:
        print("WARNING: Very few pairs. DPO may not be effective.", flush=True)

    if not pairs:
        print("ERROR: No pairs generated. Saving SFT model as final output.", flush=True)
        out_path = Path(output_dir) / "final"
        out_path.mkdir(parents=True, exist_ok=True)
        sft_model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))
        return out_path

    dpo_dataset = Dataset.from_list(pairs)

    dpo_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    out_path = Path(output_dir)
    training_args = DPOConfig(
        output_dir=str(out_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        beta=beta,
        logging_steps=2,
        save_strategy="epoch",
        max_length=max_length,
        bf16=torch.backends.mps.is_available() or torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32, device_map="auto"
    )

    trainer = DPOTrainer(
        model=sft_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dpo_dataset,
        peft_config=dpo_lora_config,
        processing_class=tokenizer,
    )

    print("Starting DPO training...", flush=True)
    trainer.train()

    final_path = out_path / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"DPO model saved to {final_path}", flush=True)
    return final_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", default="checkpoints/sft/final")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="checkpoints/dpo")
    parser.add_argument("--num-problems", type=int, default=80)
    parser.add_argument("--samples-per-problem", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    run_dpo(
        sft_checkpoint=args.sft_checkpoint,
        base_model_name=args.model,
        output_dir=args.output_dir,
        num_problems=args.num_problems,
        samples_per_problem=args.samples_per_problem,
        num_epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
    )