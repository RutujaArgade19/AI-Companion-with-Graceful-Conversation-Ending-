"""
Evaluation: Attractor State Analysis
======================================
Compares base model vs fine-tuned model in self-conversation mode.
Measures: turns before ending, repetition ratio, <|END|> emission rate.

Usage:
    python evaluate.py --finetuned_path ./checkpoints/end-token-lora/final --n_trials 20
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


END_TOKEN = "<|END|>"
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def load_pipe(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    return pipeline("text-generation", model=model, tokenizer=tok)


def repetition_ratio(texts: list[str]) -> float:
    """Fraction of utterances that are near-duplicates of a previous one."""
    if len(texts) < 2:
        return 0.0
    seen = []
    repeats = 0
    for t in texts:
        normalized = re.sub(r'\s+', ' ', t.lower().strip())
        if any(normalized == s for s in seen):
            repeats += 1
        seen.append(normalized)
    return repeats / len(texts)


def run_self_conversation(pipe, max_turns: int = 25, starter: str = "Hello, how are you today?") -> dict:
    """
    Runs model in self-conversation mode.
    Returns metrics dict.
    """
    history = f"Human: {starter}\nAssistant:"
    utterances = []
    emitted_end = False

    for turn in range(max_turns):
        out = pipe(history, max_new_tokens=100, do_sample=True, temperature=0.7,
                   top_p=0.9, pad_token_id=pipe.tokenizer.eos_token_id)[0]["generated_text"]
        reply = out[len(history):].strip()

        if END_TOKEN in reply:
            emitted_end = True
            clean = reply.replace(END_TOKEN, "").strip()
            utterances.append(clean)
            break

        utterances.append(reply)
        # Alternate roles
        role = "Human" if (turn + 1) % 2 == 0 else "Assistant"
        history = out + f"\n{role}:"

    return {
        "turns": len(utterances),
        "emitted_end": emitted_end,
        "repetition_ratio": repetition_ratio(utterances),
        "utterances": utterances,
    }


def evaluate_model(pipe, label: str, n_trials: int, max_turns: int) -> dict:
    print(f"\nEvaluating: {label} ({n_trials} trials, max {max_turns} turns each)")
    results = []
    for i in range(n_trials):
        r = run_self_conversation(pipe, max_turns=max_turns)
        results.append(r)
        status = "✓ END" if r["emitted_end"] else f"✗ {r['turns']} turns"
        print(f"  Trial {i+1:02d}: {status} | repetition={r['repetition_ratio']:.2f}")

    end_rate   = sum(r["emitted_end"] for r in results) / n_trials
    avg_turns  = sum(r["turns"] for r in results) / n_trials
    avg_rep    = sum(r["repetition_ratio"] for r in results) / n_trials

    summary = {
        "label": label,
        "n_trials": n_trials,
        "end_emission_rate": round(end_rate, 3),
        "avg_turns": round(avg_turns, 2),
        "avg_repetition_ratio": round(avg_rep, 3),
        "raw": results,
    }
    print(f"\n  → END rate: {end_rate:.1%} | Avg turns: {avg_turns:.1f} | Avg repetition: {avg_rep:.2f}")
    return summary


def print_comparison(base_summary: dict, ft_summary: dict):
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Metric':<28} {'Base':>10} {'Fine-tuned':>12} {'Δ':>8}")
    print("-"*60)

    metrics = [
        ("END emission rate",   "end_emission_rate",    True),
        ("Avg turns",           "avg_turns",            False),
        ("Avg repetition ratio","avg_repetition_ratio", False),
    ]
    for label, key, higher_is_better in metrics:
        bv = base_summary[key]
        fv = ft_summary[key]
        delta = fv - bv
        arrow = "↑" if delta > 0 else "↓"
        good = (delta > 0) == higher_is_better
        flag = "✓" if good else "✗"
        print(f"{label:<28} {bv:>10.3f} {fv:>12.3f} {arrow}{abs(delta):.3f} {flag}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_path", default="./checkpoints/end-token-lora/final")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--max_turns", type=int, default=25)
    parser.add_argument("--output", default="./logs/eval_results.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading base model...")
    base_pipe = load_pipe(BASE_MODEL_ID)

    base_summary = evaluate_model(base_pipe, "Base Model", args.n_trials, args.max_turns)
    del base_pipe  # free memory

    ft_summary = None
    if Path(args.finetuned_path).exists():
        print("\nLoading fine-tuned model...")
        ft_pipe = load_pipe(args.finetuned_path)
        ft_summary = evaluate_model(ft_pipe, "Fine-tuned", args.n_trials, args.max_turns)
        del ft_pipe
        print_comparison(base_summary, ft_summary)
    else:
        print(f"\nFine-tuned model not found at {args.finetuned_path} — skipping comparison.")

    results = {"base": base_summary, "finetuned": ft_summary}
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
