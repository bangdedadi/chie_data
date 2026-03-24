import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Configuration =================
RESULTS_DIR = '../../results'
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'analysis_output')
LANGS = ["zh", "en", "fr", "ru", "ar", "es"]
LANG_COLORS = {'zh': '#d73027', 'en': '#fc8d59', 'fr': '#fee08b', 'ru': '#d9ef8b', 'ar': '#91bfdb', 'es': '#4575b4'}

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
DPI = 400

# ================= Data Loading =================
def load_multilingual_data():
    df_dict = {}
    for lang in LANGS:
        file_path = os.path.join(RESULTS_DIR, f'llm_judge_results_{lang}.jsonl')
        if not os.path.exists(file_path): continue
        
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                for resp in item.get('responses', []):
                    records.append({
                        'id': item['id'],
                        'model': resp['model_name'],
                        'score': float(resp['llm_judge_score']) if resp.get('llm_judge_score') is not None else None
                    })
        df_dict[lang] = pd.DataFrame(records).dropna(subset=['score'])
    return df_dict

def get_mean_scores_df(df_dict):
    means = {lang: df.groupby('model')['score'].mean() for lang, df in df_dict.items()}
    df_mean = pd.DataFrame(means)
    if 'zh' in df_mean.columns: df_mean = df_mean.sort_values('zh', ascending=True)
    return df_mean

# ================= Visualization & Export =================
def plot_multilingual_mean_scores(df_mean):
    fig, ax = plt.subplots(figsize=(16, 14))
    x_min, x_max = max(1.0, df_mean.min().min() - 0.2), min(5.0, df_mean.max().max() + 0.2)
    
    df_mean.plot(kind='barh', ax=ax, width=0.85, color=[LANG_COLORS.get(l, '#333333') for l in df_mean.columns], edgecolor='white')
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Average Judge Score (1-5)', fontsize=26)
    ax.set_ylabel('Model', fontsize=26)
    ax.tick_params(axis='both', labelsize=20)
    
    legend = ax.legend(title='Language', fontsize=22, loc='lower right')
    plt.setp(legend.get_title(), fontsize=24)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multilingual_mean_scores.png'), dpi=DPI)
    plt.close()

def export_latex_table(df_mean):
    df_display = df_mean.sort_values('zh', ascending=False) if 'zh' in df_mean.columns else df_mean.sort_index()
    latex_path = os.path.join(OUTPUT_DIR, 'multilingual_latex_table.txt')
    
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table*}[htbp]\n  \\centering\n  \\caption{Mean alignment scores across six official UN languages.}\n  \\label{tab:multilingual_scores}\n")
        col_format = "l" + "c" * len(df_display.columns)
        f.write(f"  \\begin{{tabular}}{{{col_format}}}\n    \\toprule\n")
        
        headers = ["Model"] + [lang.upper() for lang in df_display.columns]
        f.write("    " + " & ".join(headers) + " \\\\\n    \\midrule\n")
        
        for model in df_display.index:
            scores = df_display.loc[model].values
            row_strs = [model] + [f"{score:.2f}" for score in scores]
            f.write("    " + " & ".join(row_strs) + " \\\\\n")
            
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table*}\n")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print("Loading multilingual data...")
    df_dict = load_multilingual_data()
    if not df_dict:
        print("No multilingual data found.")
        return
        
    df_mean = get_mean_scores_df(df_dict)
    print("Plotting Multilingual Scores...")
    plot_multilingual_mean_scores(df_mean)
    
    print("Exporting LaTeX Table...")
    export_latex_table(df_mean)
    print(f"Multilingual analysis complete. Check {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
