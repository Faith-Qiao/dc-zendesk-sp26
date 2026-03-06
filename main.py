import re
from transformers import AutoModelForCausalLM, AutoTokenizer
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

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

accuracy = evalulate(model, tokenizer, PROMPT_1, batch_size=5, max_size=25)
