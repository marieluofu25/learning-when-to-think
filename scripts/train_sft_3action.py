"""SFT on JSONL produced by generate_sft_3action (chat `messages` + action tokens)."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pivot.tokens import ACTION_TOKENS
from src.train.hf_hub_config import apply_hf_hub_config
from src.train.model_loading import load_base_causal_lm, load_tokenizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _init_new_embedding_rows(model, old_vocab_size: int, new_vocab_size: int) -> None:
    if new_vocab_size <= old_vocab_size:
        return
    with torch.no_grad():
        inp = model.get_input_embeddings().weight
        mean_emb = inp[:old_vocab_size].mean(dim=0)
        for i in range(old_vocab_size, new_vocab_size):
            noise = torch.randn_like(mean_emb) * 0.01
            inp[i] = mean_emb + noise
        out = model.get_output_embeddings()
        if out is not None and out.weight is not inp:
            ow = out.weight
            mean_out = ow[:old_vocab_size].mean(dim=0)
            for i in range(old_vocab_size, new_vocab_size):
                noise = torch.randn_like(mean_out) * 0.01
                ow[i] = mean_out + noise


def load_jsonl_dataset(
    path: Path,
    tokenizer,
    limit: int | None = None,
) -> Dataset:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            ex = json.loads(line)
            messages = ex.get("messages")
            if not messages:
                continue
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            rows.append({"text": text})
    if not rows:
        raise RuntimeError(f"No examples loaded from {path}")
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chpc_sft_3action.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_hf_hub_config(cfg)
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model_name"]
    data_path = Path(cfg["sft_data_path"])
    if not data_path.is_file():
        raise FileNotFoundError(
            f"SFT JSONL missing: {data_path.resolve()}\n"
            "It is not in git — build on a machine with GPU/API, then copy to CHPC, e.g.:\n"
            "  python -m scripts.generate_rollouts_pivot --config configs/generate_rollouts_qwen25.yaml\n"
            "  python -m scripts.generate_sft_3action --rollouts <rollouts.jsonl> --output data/sft_3action_math_train.jsonl\n"
            "Or set sft_data_path in your YAML to an existing file path."
        )
    use_qlora = bool(cfg.get("use_qlora", False))
    torch_dtype_name = str(cfg.get("torch_dtype", "bfloat16"))

    tokenizer = load_tokenizer(model_name)
    pre_vocab = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
    post_vocab = len(tokenizer)

    model = load_base_causal_lm(
        model_name,
        use_qlora=use_qlora,
        torch_dtype_name=torch_dtype_name,
    )
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    model.resize_token_embeddings(post_vocab)
    _init_new_embedding_rows(model, pre_vocab, post_vocab)

    lora_rank = int(cfg.get("lora_rank", 16))
    lora_alpha = int(cfg.get("lora_alpha", 32))
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_limit = cfg.get("train_limit")
    dataset = load_jsonl_dataset(data_path, tokenizer, limit=train_limit)

    use_bf16 = torch.cuda.is_available() and torch_dtype_name in (
        "bfloat16",
        "bf16",
    )
    sft_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=int(cfg.get("num_epochs", 3)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        save_strategy="epoch",
        max_length=int(cfg.get("max_seq_length", 4096)),
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        report_to="none",
    )
    if "dataset_text_field" in inspect.signature(SFTConfig).parameters:
        sft_kwargs["dataset_text_field"] = "text"
    training_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Saved adapter + tokenizer to {final_path}", flush=True)


if __name__ == "__main__":
    main()
