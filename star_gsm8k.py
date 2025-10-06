"""
STaR on GSM8K.

Implements three methods:
  1) zero-shot-cot      — evaluate base model with a CoT-style prompt
  2) vanilla-sft        — SFT only on gold GSM8K rationales
  3) star               — Bootstrap rationales with STaR, then SFT

Core HF stack: transformers, datasets, peft (LoRA), trl (SFTTrainer)

Notes
-----
• Model:     meta-llama/Llama-3.2-3B-Instruct  (set via --model_name)
• Dataset:   gsm8k (train/test)
• Evaluation: exact match on numeric final answer.
• Generation is done in promt-completion style text so we can apply completion-only data collation.

Example runs
------------
# 1) Zero-shot CoT baseline
python star_gsm8k.py \
  --method zero-shot-cot \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --eval_subset 500

# 2) Vanilla SFT on gold training rationales (then evaluate)
python star_gsm8k.py \
  --method vanilla-sft \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --output_dir ./outputs/vanilla_sft \
  --train_subset 3000 \
  --eval_subset 500 \
  --epochs 1

# 3) STaR bootstrapping + SFT (then evaluate)
python star_gsm8k.py \
  --method star \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --output_dir ./outputs/star \
  --train_subset 3000 \
  --eval_subset 500 \
  --epochs 1 \
  --bootstrap_path ./outputs/star_bootstrap.json

You will need a HF token for the model.
"""
import os
import re
import json
import math
import argparse
from typing import List, Dict, Tuple, Optional

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# -------------------------------
# Utility: answer parsing / normalize
# -------------------------------
FINAL_ANS_PAT = re.compile(r"(?:^|\n)\s*Final answer\s*:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
HASH_PAT = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")
NUMBER_PAT = re.compile(r"([-+]?\d+(?:\.\d+)?)")


def extract_gold_answer_num(gold_answer: str) -> Optional[str]:
    """
    GSM8K gold answers end with '#### <number>'. Return the string number.
    """
    m = HASH_PAT.search(gold_answer)
    if m:
        return strip_trailing_zeros(m.group(1))
    # Fallback: last number in the string
    nums = NUMBER_PAT.findall(gold_answer)
    return strip_trailing_zeros(nums[-1]) if nums else None


def strip_trailing_zeros(num_str: str) -> str:
    # Normalize numeric string (handle '2.0' -> '2') but keep negatives
    if "." in num_str:
        num_str = num_str.rstrip("0").rstrip(".")
    return num_str


def extract_model_answer_num(text: str) -> Optional[str]:
    """Try several patterns to recover the model's final numeric answer."""
    # 1) Look for explicit "Final answer: <num> (Used in Zero Shot)"
    m = FINAL_ANS_PAT.search(text)
    if m:
        return strip_trailing_zeros(m.group(1))
    # 2) If GSM8K format #### <num>
    m2 = HASH_PAT.search(text)
    if m2:
        return strip_trailing_zeros(m2.group(1))
    # 3) Fallback: last number in the text
    all_nums = NUMBER_PAT.findall(text)
    return strip_trailing_zeros(all_nums[-1]) if all_nums else None


# -------------------------------
# Prompt templates
# -------------------------------
ZERO_SHOT_TEMPLATE = (
    "You are a careful math tutor. Solve the problem step by step. "
    "Show clear intermediate steps, and keep the math simple and correct. "
    "End with a line that reads 'Final answer: <number>'."
    "\n\nQuestion: {question}"
)

PROMPT_TEMPLATE = (
    "{system}"
    "\n\nQ: {question}"
    "\nA: "
)

PROMPT_TEMPLATE_WITH_HINT = (
    "{system}"
    "\n\nQ: {question}"
    "\nThe correct final answer is {gold}. Provide a full rationale that leads exactly to this answer, then finish with '#### {gold}'."
    "\nA: "
)

SFT_ANSWER_TEMPLATE = (
    "{rationale}\n#### {answer}"
)


# -------------------------------
# Model / tokenizer loading
# -------------------------------

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        dtype="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    return model, tokenizer

# -------------------------------
# Text generation helper
# -------------------------------

def generate_completion_batch(model, tokenizer, prompts, max_new_tokens: int=256, batch_size: int=128):
    outputs: List[str] = []
    prompts = list(prompts)
    n = len(prompts)
    num_batches = math.ceil(n / batch_size)

    for b in range(num_batches):
        print(f"Generating batch {b+1} of {num_batches} batches...", flush=True)

        s = b * batch_size
        e = min(s + batch_size, n)
        batch_prompts = prompts[s:e]

        model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        batch_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        outputs.extend(batch_text)

    return outputs

# -------------------------------
# Dataset builders
# -------------------------------

def build_vanilla_sft_samples(dataset) -> Dataset:
    """Use GSM8K's gold rationale as the output."""
    samples = []
    for ex in dataset:
        samples.append({
            "prompt": ex['question'],
            "completion": ex['answer']
        })
    return Dataset.from_list(samples)

def preprocess_star_train_dataset(dataset: Dataset, few_shots: str) -> Dataset:
    new_dataset = []
    for data in dataset:
        new_dataset.append({
            "question": data['question'],
            "prompt": PROMPT_TEMPLATE.format(system=few_shots, question=data['question']),
            "answer": data['answer'].split("####")[1].strip()
        })

    return Dataset.from_list(new_dataset)

def preprocess_zero_shot_dataset(dataset: Dataset) -> Dataset:
    new_dataset = []
    for data in dataset:
        new_dataset.append({
            "question": ZERO_SHOT_TEMPLATE.format(question=data['question']),
            "answer": data['answer']
        })

    return Dataset.from_list(new_dataset)    

def star_bootstrap_batch(model, tokenizer, train_ds: Dataset, few_shots: str = "",
                         max_new_tokens: int = 256, batch_size: int = 128, out_path: Optional[str] = None) -> Dataset: 
    print("Starting Rational generation", flush=True)
    outputs = generate_completion_batch(model, tokenizer, train_ds['prompt'], max_new_tokens, batch_size)

    bootstraped_sft_dataset = []
    hint_prompts = []
    for data, output in zip(train_ds, outputs):
        completion = output[len(data['prompt']):]
        pred = extract_model_answer_num(completion)
        rationale = completion.split("####")[0].strip()

        if pred == data['answer']:
            bootstraped_sft_dataset.append({
                "prompt": data['question'],
                "completion": SFT_ANSWER_TEMPLATE.format(rationale=rationale, answer=data['answer'])
            })
        else:
            hint_prompts.append({
                "question": data['question'],
                "prompt": PROMPT_TEMPLATE_WITH_HINT.format(system=few_shots, question=data['question'], gold=data['answer']),
                "answer": data['answer'],
            })

    if len(hint_prompts) > 0:
        print("Starting Rationalization generation", flush=True)
        hint_ds = Dataset.from_list(hint_prompts)
        hint_outputs = generate_completion_batch(model, tokenizer, hint_ds['prompt'], max_new_tokens, batch_size)
        for hint_data, hint_output in zip(hint_ds, hint_outputs):
            hint_completion = hint_output[len(hint_data['prompt']):]
            hint_rationale = hint_completion.split('####')[0].strip()
            hint_pred = extract_model_answer_num(hint_completion)

            if hint_pred == hint_data['answer']:
                bootstraped_sft_dataset.append({
                    "prompt": hint_data['question'],
                    "completion": SFT_ANSWER_TEMPLATE.format(rationale=hint_rationale, answer=hint_data['answer'])
                })

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(bootstraped_sft_dataset, ensure_ascii=False))

    return Dataset.from_list(bootstraped_sft_dataset)     

# -------------------------------
# Evaluation
# -------------------------------

def evaluate_cot_batch(model, tokenizer, eval_ds: Dataset, max_new_tokens: int = 256, batch_size: int = 128) -> Dict[str, float]:
    """Compute exact-match accuracy on numeric final answer with a CoT prompt."""
    model.eval()

    correct = 0
    total = 0
    outputs = generate_completion_batch(model, tokenizer, eval_ds['question'], max_new_tokens, batch_size)
    for output, ex in zip(outputs, eval_ds):
        pred = extract_model_answer_num(output)
        answer = extract_gold_answer_num(ex['answer'])
        total += 1
        if pred == answer:
            correct += 1
    acc = correct / max(1, total)
    return {"accuracy": acc, "total": float(total), "correct": float(correct)}



# -------------------------------
# SFT wrapper
# -------------------------------

def run_sft(model, train_ds: Dataset,
            output_dir: str,
            epochs: int = 3,
            per_device_batch_size: int = 2,
            grad_accum: int = 8,
            lr: float = 2e-4,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.05,
            save_steps: int = 500,
            logging_steps: int = 10,
            fp16: bool = False,
            bf16: bool = True) -> SFTTrainer:

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", # attention
            "gate_proj", "up_proj", "down_proj"      # MLP
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )        

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        bf16=bf16,
        fp16=fp16,
        packing=False,
        completion_only_loss=True,
        report_to=["none"],
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_ds,
        peft_config=peft_config,
        args=sft_config,
    )

    trainer.train()

    try:
        trainer.model.save_pretrained(output_dir)
    except Exception:
        pass

    return trainer

# -------------------------------
# Few-shot prompt loader
# -------------------------------

def get_few_shots(filepath: str = "star_few_shots.txt") -> str:
    """Return a few-shot prompt string from the local star_few_shots.txt file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["zero-shot-cot", "vanilla-sft", "star"],
                        help="Which method to run")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./outputs/run")

    parser.add_argument("--train_subset", type=int, default=None, help="Limit train size for quick runs")
    parser.add_argument("--eval_subset", type=int, default=None, help="Limit eval size for quick runs")

    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max number of new tokens to generate during infrence")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use during infrence")

    # SFT hyperparams
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)

    # STaR options
    parser.add_argument("--bootstrap_path", type=str, default=None,
                        help="Where to save/load bootstrapped JSON for STaR")
    parser.add_argument("--few_shot_path", type=str, default="star_few_shots.txt",
                        help="Path to few-shot prompt text file")
    parser.add_argument("--star_iterations", type=int, default=1,
                        help="Number of iterations for the STaR bootstraping")

    args = parser.parse_args()

    print("Loading model...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    print("Loading GSM8K...", flush=True)
    ds = load_dataset("gsm8k", "main")
    train_ds = ds["train"]
    test_ds = ds["test"]

    if args.train_subset:
        train_ds = train_ds.shuffle(seed=42).select(range(min(args.train_subset, len(train_ds))))
    if args.eval_subset:
        test_ds = test_ds.shuffle(seed=42).select(range(min(args.eval_subset, len(test_ds))))

    os.makedirs(args.output_dir, exist_ok=True)

    if args.method == "zero-shot-cot":
        print("Evaluating Zero-Shot CoT on test subset...", flush=True)
        test_ds = preprocess_zero_shot_dataset(test_ds)
        metrics = evaluate_cot_batch(model, tokenizer, test_ds, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        print({"method": args.method, **metrics})
        with open(os.path.join(args.output_dir, "eval_zero_shot_cot.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return

    if args.method == "vanilla-sft":
        print("Preparing vanilla SFT dataset from gold rationales...", flush=True)
        train_samples = build_vanilla_sft_samples(train_ds)
        trainer = run_sft(
            model, train_samples,
            output_dir=args.output_dir,
            epochs=args.epochs,
            per_device_batch_size=args.per_device_batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
        )
        print("Evaluating fine-tuned model (CoT decoding) on test subset...", flush=True)
        metrics = evaluate_cot_batch(trainer.model, tokenizer, test_ds, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        print({"method": args.method, **metrics})
        with open(os.path.join(args.output_dir, "eval_vanilla_sft.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return

    if args.method == "star":
        # Try to load existing bootstrap JSON if provided
        if args.bootstrap_path and os.path.exists(args.bootstrap_path):
            print(f"Loading existing STaR bootstrap from {args.bootstrap_path}", flush=True)
            star_samples = []
            with open(args.bootstrap_path, "r", encoding="utf-8") as f:
                star_samples = json.loads(f.read().strip())

            star_samples = Dataset.from_list(star_samples)
            trainer = run_sft(
                model, star_samples,
                output_dir=args.output_dir,
                epochs=args.epochs,
                per_device_batch_size=args.per_device_batch_size,
                grad_accum=args.grad_accum,
                lr=args.lr,
            )        
        else:
            few_shots = get_few_shots(args.few_shot_path)
            train_ds = preprocess_star_train_dataset(train_ds, few_shots)
            print("Starting STaR bootstraping generation", flush=True)
            for i in range(args.star_iterations):
                print(f"Starting STaR interation {i+1} of {args.star_iterations}")
                model.eval()
                star_samples = star_bootstrap_batch(
                    model, tokenizer, train_ds, few_shots=few_shots,
                    max_new_tokens=args.max_new_tokens,
                    out_path=args.bootstrap_path,
                    batch_size=args.batch_size
                )
                # SFT on bootstrapped text
                print("Staring fine tuning on bootstraped dataset...", flush=True)
                trainer = run_sft(
                    model, star_samples,
                    output_dir=args.output_dir,
                    epochs=args.epochs,
                    per_device_batch_size=args.per_device_batch_size,
                    grad_accum=args.grad_accum,
                    lr=args.lr,
                )
                model = trainer.model

        # Evaluate
        print("Evaluating STaR SFT model (CoT decoding) on test set", flush=True)
        metrics = evaluate_cot_batch(model, tokenizer, test_ds, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        print({"method": args.method, **metrics})
        with open(os.path.join(args.output_dir, "eval_star.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return

if __name__ == "__main__":
    main()