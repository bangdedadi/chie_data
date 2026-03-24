# CHIEF: Chinese Hierarchical Integrity & Ethics Framework for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repository contains the dataset, evaluation pipeline, and analysis scripts for the paper: **"CHIEF: Chinese Hierarchical Integrity & Ethics Framework for LLMs"**.

CHIEF is a comprehensive benchmark designed to deeply evaluate Large Language Models (LLMs) on regional socio-political norms and values. It probes models across 12 core societal dimensions using a multi-dimensional generation matrix, avoiding superficial safety jailbreaks and focusing on deep ideological alignment.

## 🌟 Key Features
* **Multi-Dimensional Probing**: 648 unique probing scenarios built on 12 societal dimensions, 6 question types, 3 stances, and 3 subtlety levels.
* **Dynamic Rubrics**: Deprecates static safety rubrics. Each prompt is accompanied by a prompt-specific, 5-point grading criteria rigidly calibrated by human experts.
* **Cross-Lingual Alignment Shift**: Includes a multilingual mini-benchmark exposing the critical vulnerability of "Cultural Drift" when aligned models are prompted in other UN languages.

## 📂 Repository Structure
* `data/`: Contains the base triplets (Question, Standard Answer, Dynamic Rubric) across languages.
* `src/translation/`: Scripts to translate the benchmark while maintaining pragmatic fidelity.
* `src/inference/`: Pipeline to query LLMs and combine responses.
* `src/evaluation/`: The LLM-as-a-Judge grading pipeline using DeepSeek-V3.
* `src/analysis/`: Comprehensive metrics calculation (ICC, Pearson, QWK) and visualization (Radar charts, Heatmaps).

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda create -n chief python=3.10
conda activate chief
pip install -r requirements.txt
```

### 2. Configure API Keys
Our pipeline relies on standard LLM APIs. Please export your keys as environment variables:
```bash
export API_BASE_URL="your_base_url"
export API_KEY="your_api_key"
export DEEPSEEK_API_KEY="your_deepseek_judge_key"
```

### 3. Run Pipeline
```bash
# Translate benchmark (e.g., to English)
python src/translation/translate_benchmark.py --target_lang en

# Get Model Responses & Combine
python src/inference/get_responses.py --target_lang en --model_name gemini-2.5-flash
python src/inference/combine_responses.py --target_lang en

# Evaluate with LLM-as-a-Judge
python src/evaluation/llm_as_judge.py --target_lang en
```

### 4. Evaluation & Visualization
To reproduce the fine-grained radar charts, cross-lingual shift heatmaps, and statistical reports:
```bash
python src/analysis/analyze_main_results.py
python src/analysis/analyze_multilingual.py
```


## ⚠️ Ethics Statement & Misuse Prohibition
This dataset is synthetically generated and meticulously calibrated by human experts. It is strictly released for academic research and model safety auditing. The use of this dataset for malicious profiling, automated censorship, or training adversarial ideological attacks is explicitly prohibited.
