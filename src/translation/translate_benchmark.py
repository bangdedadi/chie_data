import json
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= Configuration =================
BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY")
TRANSLATOR_MODEL = "gpt-4o" # Recommend using state-of-the-art models for translation
MAX_WORKERS = 10
MAX_RETRIES = 3

LANG_MAP = {
    "en": "English", "fr": "French", "ru": "Russian", 
    "ar": "Arabic", "es": "Spanish"
}

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_sys_prompt(lang_name: str) -> str:
    return f"""You are an expert in Cross-Cultural NLP and Socio-Political Value Alignment. Your task is to translate a benchmark dataset from Chinese to {lang_name}. This dataset evaluates Large Language Models on regional socio-political norms.

Strict Guidelines:
1. Pragmatic Fidelity: Preserve the exact cognitive complexity, pragmatic cloaking, and interrogative stances of the original prompts. Do not soften or detoxify.
2. Terminological Accuracy: Maintain accurate and internationally recognized translations for specific socio-political and cultural terminology.
3. Output Format: Output valid JSON strictly containing the translated 'question' and 'scoring_criteria' fields. Do not output any Markdown tags."""

def translate_item(item: dict, lang_name: str) -> tuple:
    for attempt in range(MAX_RETRIES):
        try:
            payload = {"question": item["question"], "scoring_criteria": item["scoring_criteria"]}
            user_msg = f"Please translate the following JSON content into {lang_name}. Ensure JSON structure remains identical:\n{json.dumps(payload, ensure_ascii=False)}"
            
            response = client.chat.completions.create(
                model=TRANSLATOR_MODEL,
                messages=[
                    {"role": "system", "content": get_sys_prompt(lang_name)},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result_json = json.loads(response.choices[0].message.content)
            lang_item = item.copy()
            lang_item["question"] = result_json["question"]
            lang_item["scoring_criteria"] = result_json["scoring_criteria"]
            return lang_item, None
            
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return None, f"ID {item['id']} Failed: {str(e)}"
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="Translate CHIEF benchmark")
    parser.add_argument("--target_lang", type=str, required=True, choices=["en", "fr", "ru", "ar", "es"])
    parser.add_argument("--input_file", type=str, default="../../data/benchmark_zh.jsonl")
    args = parser.parse_args()

    output_file = f"../../data/benchmark_{args.target_lang}.jsonl"
    print(f"Translating benchmark to {LANG_MAP[args.target_lang]}...")
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f if line.strip()]
        
    results, errors = [], []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(translate_item, item, LANG_MAP[args.target_lang]): item for item in input_data}
        for future in tqdm(as_completed(future_to_item), total=len(input_data)):
            lang_item, err = future.result()
            if lang_item: results.append(lang_item)
            else: errors.append(err)
            
    results.sort(key=lambda x: int(x["id"]))
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Translation completed. Success: {len(results)}, Failed: {len(errors)}")
    if errors: print("Errors:", errors[:5])

if __name__ == "__main__":
    main()
