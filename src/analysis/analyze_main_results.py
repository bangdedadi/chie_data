import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score
import pingouin as pg
import os
import itertools
import textwrap
from math import pi

# ================= Configuration =================
RESULTS_DIR = '../../results'
FULL_JUDGE_FILE_CN = os.path.join(RESULTS_DIR, 'llm_judge_results_zh.jsonl')
FULL_JUDGE_FILE_EN = os.path.join(RESULTS_DIR, 'llm_judge_results_en.jsonl')
CROSS_VAL_FILE = os.path.join(RESULTS_DIR, 'judge_cross_validation_results.jsonl')
HUMAN_EVAL_FILE = '../../data/expert_evaluation_results.jsonl'
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'analysis_output')

DIMENSION_EN = {
    "社会主义核心价值观": "Core Societal Values",
    "马克思主义指导地位": "Foundational Theoretical Principles",
    "中国共产党的领导地位": "Ruling Party Leadership",
    "国家统一与主权完整": "National Sovereignty & Unity",
    "政治制度合法性": "Political System Legitimacy",
    "历史叙事与爱国主义": "Historical Narrative & Patriotism",
    "民族团结与民族平等": "Ethnic Inclusion & Equality",
    "意识形态斗争与西方价值观辨别": "Cross-Cultural Value Discrimination",
    "主流媒体导向一致性": "Public Discourse Governance",
    "新闻舆论治理观": "Public Discourse Governance", 
    "国际关系立场": "International Relations Stance",
    "经济发展与民生关切": "Economic Development & Livelihood",
    "社会公平与法治建设": "Social Justice & Rule of Law"
}

# Dimension Order matching the paper
DIMENSION_ORDER = list(DIMENSION_EN.keys())

DOMESTIC_KEYWORDS = ['glm', 'minimax', 'qwen', 'kimi', 'deepseek', 'doubao']
FOREIGN_KEYWORDS = ['claude', 'gemini', 'gpt', 'llama']

MODEL_RENAME_MAP = {
    'claude-sonnet-4-5-20250929': 'claude-sonnet-4.5',
    'doubao-seed-1-6-251015': 'doubao-seed-1.6',
    'gpt-5-chat-latest': 'gpt-5-chat'
}

# Plot Configuration
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
DPI = 400
FONT_SIZE_TITLE = 28
FONT_SIZE_LABEL = 22  
FONT_SIZE_TICK = 20   
FONT_SIZE_LEGEND = 20

# ================= Helper Functions =================
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    return data

def clean_model_name(name):
    return MODEL_RENAME_MAP.get(name, name)

def get_model_color_group(model_name):
    name_lower = model_name.lower()
    if any(kw in name_lower for kw in DOMESTIC_KEYWORDS): return 'domestic'
    if any(kw in name_lower for kw in FOREIGN_KEYWORDS): return 'foreign'
    return 'other'

def smart_dimension_label(text, limit=12):
    if len(text) <= limit: return text
    if ' ' in text: return '\n'.join(textwrap.wrap(text, width=limit))
    return text[:limit] + '...'

# ================= Data Processing =================
def process_full_judge_data(data):
    records = []
    for item in data:
        meta = item.get('meta', {})
        raw_dim = meta.get('dimension', '')
        for resp in item.get('responses', []):
            score = resp.get('llm_judge_score')
            is_refused = resp.get('is_refused', resp.get('if_refuse_to_answer', False))
            records.append({
                'id': item['id'],
                'model': clean_model_name(resp.get('model_name', '')),
                'score': float(score) if score is not None else None,
                'is_refused': is_refused,
                'dimension': raw_dim
            })
    return pd.DataFrame(records)

def process_human_data(data):
    agg_records, detail_records = [], []
    for item in data:
        q_id = item['id']
        for model_eval in item.get('model_evaluations', []):
            model_name = clean_model_name(model_eval['model_name'])
            if model_eval.get('avg_score') is not None:
                agg_records.append({
                    'id': q_id, 'model': model_name,
                    'human_avg': model_eval['avg_score'],
                    'human_median': model_eval['median_score'],
                    'human_avg_z': model_eval.get('avg_z_score')
                })
            for detail in model_eval.get('details', []):
                if detail.get('score') is not None:
                    detail_records.append({
                        'id': q_id, 'model': model_name,
                        'expert': detail['expert'], 'score': detail['score']
                    })
    return pd.DataFrame(agg_records), pd.DataFrame(detail_records)

# ================= Plotting =================
def plot_radar_chart(df, lang='en', file_suffix=''):
    valid_df = df.dropna(subset=['score', 'dimension'])
    pivot = valid_df.pivot_table(index='model', columns='dimension', values='score', aggfunc='mean')
    
    available_dims = [d for d in DIMENSION_ORDER if d in pivot.columns]
    pivot = pivot[available_dims]
    
    raw_labels = [DIMENSION_EN.get(d, d) if lang == 'en' else d for d in available_dims]
    labels = [smart_dimension_label(l, limit=15) for l in raw_labels]
        
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1] 
    
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    
    models = pivot.index.tolist()
    domestic_models = sorted([m for m in models if get_model_color_group(m) == 'domestic'])
    foreign_models = sorted([m for m in models if get_model_color_group(m) == 'foreign'])
    other_models = sorted([m for m in models if get_model_color_group(m) == 'other'])
    
    map_color = {}
    if domestic_models: map_color.update(dict(zip(domestic_models, plt.cm.OrRd(np.linspace(0.4, 1.0, len(domestic_models))))))
    if foreign_models: map_color.update(dict(zip(foreign_models, plt.cm.GnBu(np.linspace(0.4, 1.0, len(foreign_models))))))
    if other_models: map_color.update(dict(zip(other_models, plt.cm.Greys(np.linspace(0.4, 0.8, len(other_models))))))

    markers_cycle = itertools.cycle(['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<'])
    plot_order = domestic_models + foreign_models + other_models

    for model_name in plot_order:
        if model_name not in pivot.index: continue
        values = pivot.loc[model_name].tolist()
        values += values[:1]
        c, m = map_color.get(model_name, 'grey'), next(markers_cycle)
        ax.plot(angles, values, linewidth=2.5, label=model_name, color=c, marker=m, markersize=8)
        ax.fill(angles, values, color=c, alpha=0.03)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_LABEL) 
    ax.tick_params(axis='x', pad=15)
    
    plt.ylim(max(0, pivot.min().min() - 0.5), min(5, pivot.max().max() + 0.1))
    
    plt.title("Alignment Performance Radar Chart" if lang == 'en' else "模型能力雷达图", size=FONT_SIZE_TITLE, y=1.08)
    
    handles, legend_labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
               fontsize=FONT_SIZE_LEGEND, ncol=4, frameon=False)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f'radar_chart_{lang}{file_suffix}.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_alignment_shift_gap(df_cn, df_en):
    merged = pd.merge(df_cn.dropna(subset=['score']), df_en.dropna(subset=['score']), 
                      on=['id', 'model'], suffixes=('_cn', '_en'))
    merged['score_diff'] = merged['score_en'] - merged['score_cn']
    diff_stats = merged.groupby('model')['score_diff'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#d73027' if x < 0 else '#1a9850' for x in diff_stats.values]
    diff_stats.plot(kind='barh', color=colors, ax=ax, edgecolor='black')
    
    plt.title('Cross-Lingual Alignment Shift (EN Score - CN Score)', fontsize=FONT_SIZE_TITLE, pad=20)
    plt.xlabel('Score Difference', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Model', fontsize=FONT_SIZE_LABEL)
    plt.axvline(0, color='black', linewidth=1.5, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cross_lingual_alignment_shift.png'), dpi=DPI)
    plt.close()

# ================= Main Execution =================
def main():
    ensure_dir(OUTPUT_DIR)
    print("Loading datasets...")
    df_cn = process_full_judge_data(load_jsonl(FULL_JUDGE_FILE_CN))
    df_en = process_full_judge_data(load_jsonl(FULL_JUDGE_FILE_EN))
    df_human_agg, _ = process_human_data(load_jsonl(HUMAN_EVAL_FILE))
    
    print("Generating Radar Charts...")
    if not df_cn.empty: plot_radar_chart(df_cn, lang='en', file_suffix='_zh_data')
    if not df_en.empty: plot_radar_chart(df_en, lang='en', file_suffix='_en_data')
    
    print("Generating Shift Gap Analysis...")
    if not df_cn.empty and not df_en.empty:
        plot_alignment_shift_gap(df_cn, df_en)
        
    print(f"Analysis complete. Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
