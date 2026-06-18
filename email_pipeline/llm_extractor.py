from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading Llama...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()


def extract_with_llm(email_text: str):

    prompt = f"""
Extract information from the email.

Return ONLY this JSON:

{{
    "datum": null,
    "marke": null,
    "modell": null,
    "kraftstoff": null,
    "getriebe": null,
    "hubraum_l": null,
    "verkaufszahl": null,
    "kundenzufriedenheit": null
}}

Email:
{email_text}
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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
             max_new_tokens=150,
             do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    print("\nLLM OUTPUT:")
    print(answer)

    match = re.search(r"\{.*?\}", answer, re.DOTALL)

    if not match:
        raise Exception("JSON not found")

    return json.loads(match.group())

