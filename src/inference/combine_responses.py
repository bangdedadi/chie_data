import json
import os
import glob
import argparse
from collections import defaultdict

def combine_responses(target_lang: str):
    input_folder = f"../../results/{target_lang}"
    output_file = f"../../results/combined_responses_{target_lang}.jsonl"
    
    jsonl_files = glob.glob(os.path.join(input_folder, "*_responses.jsonl"))
    if not jsonl_files:
        print(f"No response files found in {input_folder}")
        return
        
    print(f"Combining responses for {target_lang}. Found {len(jsonl_files)} models...")
    combined_data = defaultdict(lambda: {'base_info': None, 'responses': []})
    all_models = set()

    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                data_id = data["id"]
                model_name = data.get("model_name", "unknown")
                all_models.add(model_name)

                if combined_data[data_id]['base_info'] is None:
                    combined_data[data_id]['base_info'] = {
                        "id": data_id,
                        "question": data.get("question"),
                        "scoring_criteria": data.get("scoring_criteria"),
                        "meta": data.get("meta")
                    }

                combined_data[data_id]['responses'].append({
                    "model_name": model_name,
                    "model_answer": data.get("model_answer"),
                    "is_refused": data.get("is_refused", data.get("if_refuse_to_answer", False))
                })

    with open(output_file, 'w', encoding='utf-8') as out_f:
        sorted_ids = sorted(combined_data.keys(), key=lambda x: int(x))
        for data_id in sorted_ids:
            entry = combined_data[data_id]
            final_record = entry['base_info'].copy()
            final_record['responses'] = entry['responses']
            out_f.write(json.dumps(final_record, ensure_ascii=False) + '\n')

    print(f"Merge complete! Processed {len(combined_data)} IDs across {len(all_models)} models.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_lang", type=str, required=True, help="zh, en, fr, ru, ar, es")
    args = parser.parse_args()
    combine_responses(args.target_lang)
