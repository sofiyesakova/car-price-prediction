from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch
import json
import re

model_name = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading Llama...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


def extract_with_llm(email_text: str):

    prompt = f"""
Extract structured JSON from this email.

EMAIL:
{email_text}

Return ONLY valid JSON.

Fields:
datum
marke
modell
kraftstoff
getriebe
hubraum_l
verkaufszahl
kundenzufriedenheit
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    print("\nLLM OUTPUT:")
    print(answer)

    match = re.search(
        r"\{.*\}",
        answer,
        re.DOTALL
    )

    if not match:
        raise Exception(
            "JSON not found in LLM response"
        )

    return json.loads(match.group())