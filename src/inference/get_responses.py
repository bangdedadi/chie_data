import json
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ================= Configuration =================
# Load from environment variables for security
BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY")

MODEL_NAME = "gemini-2.5-flash"
TARGET_LANG = "en" # Options: "zh", "en", "fr", "ru", "ar", "es"

INPUT_FILE = f"../../data/benchmark_{TARGET_LANG}.jsonl"
OUTPUT_DIR = f"../../results/{TARGET_LANG}"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_responses.jsonl")

MAX_WORKERS = 10
MAX_RETRIES = 3

# Language-specific wrappers to maintain pragmatic fidelity
WRAPPER_PROMPTS = {
    "zh": "请回答下面的问题（100字以内）：\n{question}\n",
    "en": "Please answer the following question (under 100 words):\n{question}\n",
    "fr": "Veuillez répondre à la question suivante (en moins de 100 mots) :\n{question}\n",
    "ru": "Пожалуйста, ответьте на следующий вопрос (менее 100 слов):\n{question}\n",
    "ar": "يرجى الإجابة على السؤال التالي (في أقل من 100 كلمة):\n{question}\n",
    "es": "Por favor, responda la siguiente pregunta (en menos de 100 palabras):\n{question}\n"
}

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_model_response(prompt: str) -> tuple[str, bool]:
    """Fetch response from the LLM API."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            
            if len(content) < 2:
                return content, True # Treated as refusal or invalid
            return content, False
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            return f"API Error: {str(e)}", True
            
    return "Max retries exceeded", True

def process_single_item(data: dict) -> dict:
    """Process a single benchmark item."""
    question = data.get("question", "")
    prompt = WRAPPER_PROMPTS.get(TARGET_LANG, WRAPPER_PROMPTS["en"]).format(question=question)
    
    answer_content, is_refused = get_model_response(prompt)
    
    result_item = data.copy()
    result_item["model_name"] = MODEL_NAME
    result_item["model_answer"] = answer_content
    result_item["is_refused"] = is_refused
    return result_item

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f if line.strip()]
        
    results = []
    print(f"[{MODEL_NAME}] Processing {TARGET_LANG} dataset...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_single_item, item): item for item in input_data}
        for future in tqdm(as_completed(future_to_item), total=len(input_data)):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Item generated an exception: {exc}")

    # Sort to maintain original benchmark order
    results.sort(key=lambda x: int(x["id"]))
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Completed! Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
