"""
Fine-tuning Script: Conversation Ending Detection
===================================================
Fine-tunes a small LLM (Qwen-2.5-3B or LLaMA-3.2-3B) using LoRA
to emit <|END|> when a conversation has naturally concluded.

Usage:
    pip install transformers trl peft accelerate bitsandbytes datasets torch
    python finetune.py --model_name Qwen/Qwen2.5-3B-Instruct --dataset_path ./processed/conversation_endings

Tested on: single A100/H100 40GB GPU. For smaller GPUs, reduce batch size or use 4-bit quant.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


# ── CLI Args ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--dataset_path", default="./processed/conversation_endings")
    parser.add_argument("--output_dir", default="./checkpoints/end-token-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (QLoRA)")
    return parser.parse_args()


# ── Tokenizer + Model ─────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_name: str, load_in_4bit: bool):
    END_TOKEN = "<|END|>"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add END token to vocabulary
    num_added = tokenizer.add_tokens([END_TOKEN], special_tokens=True)
    print(f"Added {num_added} new token(s): {END_TOKEN}")

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    return model, tokenizer


# ── LoRA Config ───────────────────────────────────────────────────────────────
def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                           # rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    print(f"\n{'='*60}")
    print(f"Model  : {args.model_name}")
    print(f"Data   : {args.dataset_path}")
    print(f"Output : {args.output_dir}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = load_from_disk(args.dataset_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.load_in_4bit)
    model = apply_lora(model)

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=False,
        bf16=True,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",           # set to "wandb" if you want tracking
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    # Save final adapter + tokenizer
    output_path = Path(args.output_dir) / "final"
    trainer.model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"\nModel saved to {output_path}")


# ── Self-conversation Evaluation ──────────────────────────────────────────────
def run_self_conversation(model_path: str, max_turns: int = 20):
    """
    Runs the fine-tuned model in self-conversation mode.
    Stops when <|END|> is emitted or max_turns is reached.
    Returns number of turns before ending.
    """
    from transformers import pipeline

    END_TOKEN = "<|END|>"
    pipe = pipeline("text-generation", model=model_path, device_map="auto",
                    trust_remote_code=True)

    history = "Human: Hello, how are you today?\n"
    print("\n--- Self-conversation ---")
    print(history.strip())

    turns = 0
    for _ in range(max_turns):
        output = pipe(history, max_new_tokens=80, do_sample=True, temperature=0.7,
                      pad_token_id=pipe.tokenizer.eos_token_id)[0]["generated_text"]
        new_text = output[len(history):].strip()
        print(new_text)

        if END_TOKEN in new_text:
            print(f"\n✓ Model emitted {END_TOKEN} after {turns+1} turns.")
            return turns + 1

        history = output + "\n"
        turns += 1

    print(f"\n✗ No {END_TOKEN} emitted after {max_turns} turns (attractor state risk).")
    return max_turns


if __name__ == "__main__":
    args = parse_args()
    train(args)

    # After training, run a quick self-conversation test
    print("\nRunning self-conversation test on fine-tuned model...")
    run_self_conversation(str(Path(args.output_dir) / "final"))
