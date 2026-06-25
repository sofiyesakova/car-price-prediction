from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Loading Qwen2.5...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

model.eval()


def extract_with_llm(email_text: str):

    prompt = f"""
You are a strict data extraction system.

Return ONLY valid JSON.

Schema:
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

Rules:
- no text
- no explanation
- only JSON
- use null if unknown

Email:
{email_text}

JSON:
"""

    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return safe_extract(text)


def safe_extract(text: str):

    DEFAULT = {
        "datum": None,
        "marke": None,
        "modell": None,
        "kraftstoff": None,
        "getriebe": None,
        "hubraum_l": None,
        "verkaufszahl": None,
        "kundenzufriedenheit": None
    }

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return DEFAULT

    raw = match.group()

    try:
        data = json.loads(raw)
    except:
        return DEFAULT

    result = DEFAULT.copy()

    for k in result:
        result[k] = data.get(k, None)

    return result