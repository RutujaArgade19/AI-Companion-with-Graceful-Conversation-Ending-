# Project Problem Statement

Endless loops in AI companions are not just a technical failure — they are a user experience and wellbeing problem. A companion that cannot end a conversation naturally may feel unsettling to vulnerable users. We fine-tune a small LLM to detect closing moments and gracefully conclude, then deploy it in a companion interface and measure both technical (attractor state reduction) and experiential (naturalness ratings) outcomes.


## Project Structure

```
companion/
├── data/
│   └── build_dataset.py      # Download, label, and format training data
├── training/
│   └── finetune.py           # LoRA fine-tuning with HuggingFace trl/peft
├── backend/
│   └── backend.py            # FastAPI server — serves model, logs sessions
├── frontend/
│   └── index.html            # Chat UI — works as a static file
├── evaluation/
│   └── evaluate.py           # Self-conversation attractor state analysis
└── requirements.txt
```

---

