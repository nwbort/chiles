"""
Fine-tune Gemma-2-2B-IT to generate Adrian Chiles-style Guardian article titles.

Requirements:
    pip install torch transformers peft trl datasets accelerate

Needs:
    - A GPU (8 GB+ VRAM recommended for bfloat16 + LoRA)
    - HuggingFace token with access to google/gemma-2-2b-it
      set HF_TOKEN env var or pass --hf-token
"""

import argparse
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "google/gemma-2-2b-it"
GENERATION_PROMPT = "Write a Guardian article headline in the style of Adrian Chiles."


def load_titles(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def format_example(title: str, tokenizer) -> str:
    messages = [
        {"role": "user", "content": GENERATION_PROMPT},
        {"role": "assistant", "content": title},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--titles", default="titles.txt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--n-generate", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. Fine-tuning Gemma requires a CUDA GPU (8 GB+ VRAM). "
            "See the workflow README for GPU runner setup."
        )

    random.seed(args.seed)

    titles = load_titles(args.titles)
    random.shuffle(titles)
    print(f"Loaded {len(titles)} titles")

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)

    texts = [format_example(t, tokenizer) for t in titles]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Training on {len(dataset)} examples")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=args.hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    training_args = SFTConfig(
        output_dir="./lora-adapter",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=128,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
    )

    print("Training...")
    trainer.train()
    print("Training complete.")

    # Generate
    model.eval()
    device = next(model.parameters()).device

    prompt_tokens = tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": GENERATION_PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        return_tensors="pt",
    ).to(device)

    generated: list[str] = []
    attempts = 0
    while len(generated) < args.n_generate and attempts < args.n_generate * 4:
        attempts += 1
        with torch.no_grad():
            output = model.generate(
                **prompt_tokens,
                max_new_tokens=64,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][prompt_tokens["input_ids"].shape[1]:]
        title = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if len(title) >= 8 and title not in generated:
            generated.append(title)

    with open("generated.txt", "w", encoding="utf-8") as f:
        f.write("# Generated Adrian Chiles-style Guardian titles\n")
        f.write(f"# {args.model} / LoRA r={args.lora_r} / {args.epochs} epochs\n\n")
        for title in generated:
            f.write(title + "\n")

    print(f"Saved {len(generated)} titles to generated.txt")
    print("\nSample:")
    for t in generated[:5]:
        print(f"  {t}")


if __name__ == "__main__":
    main()
