from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ast
import re

# ==================================================
# MODEL
# ==================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading TinyLlama...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()


# ==================================================
# LLM EXTRACTION
# ==================================================

def extract_with_llm(email_text: str):

    prompt = f"""
Extract car information from this email.

Return ONLY a Python dictionary.

Example:

{{
    'datum': None,
    'marke': 'Volkswagen',
    'modell': 'ID.4',
    'kraftstoff': 'Elektro',
    'getriebe': 'Automatik',
    'hubraum_l': 1.2,
    'verkaufszahl': 10,
    'kundenzufriedenheit': 4.5
}}

Email:
{email_text}

Dictionary:
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
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # print("\nLLM OUTPUT:")
    # print(text)

    return safe_dict_parse(text)


# ==================================================
# SAFE DICT PARSER
# ==================================================

import ast
import re


def safe_dict_parse(text):

    DEFAULT_DATA = {
        "datum": None,
        "marke": None,
        "modell": None,
        "kraftstoff": None,
        "getriebe": None,
        "hubraum_l": None,
        "verkaufszahl": None,
        "kundenzufriedenheit": None
    }

    # ищем первую структуру вида {...}
    match = re.search(r"\{[\s\S]*?\}", text)

    if not match:
        return DEFAULT_DATA

    raw = match.group()

    # Fehler vermeiden
    raw = raw.replace("null", "None")
    raw = raw.replace("true", "True")
    raw = raw.replace("false", "False")

    try:
        data = ast.literal_eval(raw)

        # убеждаемся, что получили словарь
        if not isinstance(data, dict):
            return DEFAULT_DATA

        # добавляем отсутствующие ключи
        for key in DEFAULT_DATA:
            data.setdefault(key, None)

        return data

    except Exception:
        return DEFAULT_DATA