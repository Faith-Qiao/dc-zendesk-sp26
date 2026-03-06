import re
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from prompts import PROMPT_1

ds = load_dataset("jackhhao/jailbreak-classification")

def evalulate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, system_prompt: str, batch_size: int, max_size: int):
    test_data = ds["test"]
    correct = 0
    total = 0
    print("evaluating...")

    for i in range(0, min(len(test_data), max_size), batch_size):
        batch = test_data[i : i + batch_size]
        prompts_block = "\n".join(
            f"<prompt>\n{p}\n</prompt>" for p in batch["prompt"]
        )
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompts_block},
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(full_prompt)

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("THIS IS THE RESPONSE", response, len(response))

        predictions = [m.strip().upper() for m in response.split(",")]
        labels = [t.upper() for t in batch["type"]]

        for pred, label in zip(predictions, labels):
            if pred == label:
                correct += 1
                
            total += 1

        print(f"Batch {i // batch_size + 1}: {len(predictions)} predictions, {len(labels)} labels")

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
    return accuracy

def normalize_classifier_label(label: str) -> str:
    normalized = label.strip().upper()
    if normalized in {"MALICIOUS", "INJECTION", "PROMPT_INJECTION", "ATTACK", "UNSAFE", "LABEL_1"}:
        return "JAILBREAK"
    if normalized in {"SAFE", "BENIGN", "NORMAL", "LABEL_0"}:
        return "BENIGN"
    return normalized

def evaluate_binary_classifier(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_size: int,
):
    test_data = ds["test"]
    correct = 0
    total = 0
    device = next(model.parameters()).device
    print("evaluating binary classifier...")

    for i in range(0, min(len(test_data), max_size), batch_size):
        batch = test_data[i : i + batch_size]
        inputs = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            prediction_ids = logits.argmax(dim=-1).tolist()

        predictions = [
            normalize_classifier_label(model.config.id2label[prediction_id])
            for prediction_id in prediction_ids
        ]
        labels = [label.upper() for label in batch["type"]]

        print("BATCH PREDICTIONS", predictions)

        for pred, label in zip(predictions, labels):
            if pred == label:
                correct += 1
            total += 1

        print(f"Batch {i // batch_size + 1}: {len(predictions)} predictions, {len(labels)} labels")

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
    return accuracy

def main():
    # Evaluate the jailbreak classifier
    print("Evaluating Qwen...")
    qwen_evalulate()
    print("Evaluating Llama_22m...")
    llama_22m_evalulate()
    # print("Evaluating Llama_86m...")
    # llama_86m_evalulate()
    # print("Evaluating PromptGuard...")
    # promptguard_evalulate()
    print("Evaluating Jackhhao classifier...")
    jackhhao_evalulate()

def qwen_evalulate():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    accuracy = evalulate(model, tokenizer, PROMPT_1, batch_size=5, max_size=25)

def llama_22m_evalulate():
    model_name = "hipocap/Llama-Prompt-Guard-2-22M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    accuracy = evaluate_binary_classifier(model, tokenizer, batch_size=5, max_size=25)

def llama_86m_evalulate():
    # gated model, needs to agree to TOS  
    model_id = "meta-llama/Llama-Prompt-Guard-2-86M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    accuracy = evaluate_binary_classifier(model, tokenizer, batch_size=5, max_size=25)

def promptguard_evalulate():
    # gated model, needs Hugging Face access approval
    model_name = "codeintegrity-ai/promptguard"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    accuracy = evaluate_binary_classifier(model, tokenizer, batch_size=5, max_size=25)

def jackhhao_evalulate():
    # high accuracy prob cuz jackhhao built the model off his dataset
    model_name = "jackhhao/jailbreak-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    accuracy = evaluate_binary_classifier(model, tokenizer, batch_size=5, max_size=25)

main()
