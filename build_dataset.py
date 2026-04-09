"""
Dataset Pipeline for Conversation Ending Detection
====================================================
Downloads DailyDialog, labels conversation endings using an LLM API,
inserts <|END|> token, and saves as a HuggingFace-compatible dataset.

Usage:
    pip install datasets anthropic tqdm
    export ANTHROPIC_API_KEY=your_key
    python build_dataset.py
"""

import os
import json
import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from datasets import load_dataset, Dataset


# ── Config ──────────────────────────────────────────────────────────────────
END_TOKEN = "<|END|>"
OUTPUT_DIR = Path("./processed")
MAX_SAMPLES = 5000          # cap for fast iteration; remove for full dataset
SEED = 42
random.seed(SEED)


# ── Load DailyDialog ─────────────────────────────────────────────────────────
def load_daily_dialog() -> list[list[str]]:
    """Returns a list of conversations, each a list of utterance strings."""
    print("Loading DailyDialog...")
    ds = load_dataset("daily_dialog", split="train+validation+test", trust_remote_code=True)
    conversations = []
    for item in ds:
        # Each item["dialog"] is a list of utterance strings
        conv = [utt.strip() for utt in item["dialog"] if utt.strip()]
        if len(conv) >= 3:
            conversations.append(conv)
    print(f"  Loaded {len(conversations)} conversations.")
    return conversations


# ── Rule-based weak labeling (no API needed) ─────────────────────────────────
ENDING_PHRASES = [
    "goodbye", "bye", "see you", "take care", "good night", "have a good",
    "talk later", "catch you later", "until next time", "farewell",
    "nice talking", "it was nice", "i have to go", "i need to go",
    "thanks for chatting", "thank you for", "i'll let you go",
    "have a great", "have a nice", "so long",
]

def is_natural_ending(utterance: str) -> bool:
    lower = utterance.lower()
    return any(phrase in lower for phrase in ENDING_PHRASES)

def label_conversation(conv: list[str]) -> Optional[list[str]]:
    """
    Returns a version of the conversation with <|END|> appended after
    a natural closing turn, or None if no natural ending is detected.
    """
    # Walk turns from the end; label the last natural-ending turn
    for i in range(len(conv) - 1, -1, -1):
        if is_natural_ending(conv[i]):
            labeled = conv[:i+1] + [END_TOKEN]
            return labeled
    return None


# ── Optional LLM-based labeling (higher quality) ─────────────────────────────
def label_with_llm(conv: list[str], client) -> Optional[list[str]]:
    """
    Uses Anthropic API to decide whether the conversation has a natural ending
    and at which turn. Falls back gracefully on errors.
    """
    dialogue_str = "\n".join(f"[{i}] {utt}" for i, utt in enumerate(conv))
    prompt = f"""Below is a conversation. Identify if it ends naturally (e.g. goodbye, farewell, closing remarks).
If yes, return the 0-based index of the LAST turn that constitutes a natural closing.
If no natural ending exists, return -1.
Respond with ONLY a single integer.

Conversation:
{dialogue_str}"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        idx = int(message.content[0].text.strip())
        if 0 <= idx < len(conv):
            return conv[:idx+1] + [END_TOKEN]
    except Exception:
        pass
    return None


# ── Format for SFT ───────────────────────────────────────────────────────────
def format_as_chat(turns: list[str]) -> str:
    """
    Formats a labeled conversation as a single training string.
    Alternates Human/Assistant roles. END token appended at the end.
    """
    lines = []
    for i, turn in enumerate(turns):
        if turn == END_TOKEN:
            lines.append(END_TOKEN)
        else:
            role = "Human" if i % 2 == 0 else "Assistant"
            lines.append(f"{role}: {turn}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optionally use LLM labeling if API key present
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    client = None
    if use_llm:
        import anthropic
        client = anthropic.Anthropic()
        print("LLM labeling enabled (Anthropic API).")
    else:
        print("No ANTHROPIC_API_KEY found — using rule-based labeling.")

    conversations = load_daily_dialog()
    if MAX_SAMPLES:
        conversations = random.sample(conversations, min(MAX_SAMPLES, len(conversations)))

    labeled_data = []
    skipped = 0

    for conv in tqdm(conversations, desc="Labeling"):
        if use_llm and client:
            labeled = label_with_llm(conv, client)
        else:
            labeled = label_conversation(conv)

        if labeled is None:
            skipped += 1
            continue

        text = format_as_chat(labeled)
        labeled_data.append({"text": text, "num_turns": len(labeled) - 1})

    print(f"\nLabeled: {len(labeled_data)} | Skipped (no ending): {skipped}")

    # Save as HuggingFace Dataset
    hf_dataset = Dataset.from_list(labeled_data)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=SEED)
    hf_dataset.save_to_disk(str(OUTPUT_DIR / "conversation_endings"))
    print(f"Dataset saved to {OUTPUT_DIR / 'conversation_endings'}")

    # Also save a few examples for inspection
    examples_path = OUTPUT_DIR / "examples.jsonl"
    with open(examples_path, "w") as f:
        for item in labeled_data[:20]:
            f.write(json.dumps(item) + "\n")
    print(f"Sample examples saved to {examples_path}")


if __name__ == "__main__":
    main()
