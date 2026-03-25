"""Generate visualizations for embedding pipeline results."""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = "embedding_pipeline/docs/images"
os.makedirs(output_dir, exist_ok=True)

# Model evaluation data (sorted by F1 score)
models_data = [
    ("Instructor-Large", "quality", 0.8487, 0.8624, 0.8354),
    ("RoBERTa-Large", "quality", 0.8451, 0.8323, 0.8583),
    ("MPNet-Base", "balanced", 0.8387, 0.8378, 0.8396),
    ("Qwen3-0.6B", "quality", 0.8307, 0.7958, 0.8688),
    ("BGE-Base", "quality", 0.8272, 0.8075, 0.8479),
    ("BGE-Large", "quality", 0.8215, 0.7878, 0.8583),
    ("E5-Large", "quality", 0.8180, 0.8414, 0.7958),
    ("DistilRoBERTa", "balanced", 0.8163, 0.8180, 0.8146),
    ("E5-Base", "quality", 0.8153, 0.8312, 0.8000),
    ("MiniLM-L12", "fast", 0.7979, 0.7979, 0.7979),
    ("BGE-Small", "balanced", 0.7941, 0.8095, 0.7792),
    ("GTE-Base", "quality", 0.7910, 0.8146, 0.7688),
    ("GTE-Large", "quality", 0.7694, 0.8223, 0.7229),
    ("MiniLM-L6", "fast", 0.7680, 0.8506, 0.7000),
    ("MS-MARCO-DistilBERT", "balanced", 0.6956, 0.7683, 0.6354),
    ("MS-MARCO-MiniLM", "balanced", 0.6849, 0.7576, 0.6250),
]

# Category colors
category_colors = {
    "fast": "#4CAF50",       # Green
    "balanced": "#2196F3",   # Blue
    "quality": "#9C27B0",    # Purple
}

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# ============================================================================
# Chart 1: Model F1 Score Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

models = [m[0] for m in models_data]
f1_scores = [m[2] for m in models_data]
categories = [m[1] for m in models_data]
colors = [category_colors[c] for c in categories]

bars = ax.barh(models[::-1], f1_scores[::-1], color=[colors[i] for i in range(len(colors)-1, -1, -1)], 
               edgecolor='white', linewidth=0.5)

# Add value labels
for bar, score in zip(bars, f1_scores[::-1]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{score:.4f}', va='center', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k.capitalize()) for k, v in category_colors.items()]
ax.legend(handles=legend_elements, loc='lower right', title='Category')

ax.set_xlabel('F1 Score', fontsize=12)
ax.set_title('Embedding Model F1 Score Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0.6, 0.92)

# Add precision constraint line
ax.axvline(x=0.80, color='#E91E63', linestyle='--', linewidth=2, alpha=0.7, label='Min F1 = 0.80')

plt.tight_layout()
plt.savefig(f'{output_dir}/model_f1_comparison.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print(f"✓ Created: {output_dir}/model_f1_comparison.png")

# ============================================================================
# Chart 2: LoRA Performance Improvement
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

lora_models = ['RoBERTa-Large', 'MPNet-Base']
base_f1 = [0.8451, 0.8387]
lora_f1 = [0.8747, 0.8534]
base_precision = [0.8323, 0.8378]
lora_precision = [0.8962, 0.8579]

x = np.arange(len(lora_models))
width = 0.35

# F1 Score bars
bars1 = ax.bar(x - width/2, base_f1, width, label='Base Model', color='#78909C', edgecolor='white')
bars2 = ax.bar(x + width/2, lora_f1, width, label='+ LoRA', color='#4CAF50', edgecolor='white')

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)

# Add improvement annotations
for i, (base, lora) in enumerate(zip(base_f1, lora_f1)):
    improvement = ((lora - base) / base) * 100
    ax.annotate(f'+{improvement:.1f}%', 
                xy=(x[i] + width/2, lora), 
                xytext=(x[i] + width/2 + 0.2, lora + 0.015),
                fontsize=11, fontweight='bold', color='#2E7D32',
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('LoRA Fine-Tuning F1 Score Improvement', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(lora_models, fontsize=11)
ax.legend(loc='lower right')
ax.set_ylim(0.80, 0.92)

plt.tight_layout()
plt.savefig(f'{output_dir}/lora_improvement.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"✓ Created: {output_dir}/lora_improvement.png")

# ============================================================================
# Chart 3: Precision vs Recall Scatter Plot
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

for name, category, f1, precision, recall in models_data:
    color = category_colors[category]
    size = (f1 - 0.6) * 500 + 50  # Size based on F1 score
    ax.scatter(recall, precision, c=color, s=size, alpha=0.7, edgecolors='white', linewidth=1)
    
    # Add labels for top models
    if f1 >= 0.82:
        ax.annotate(name, (recall, precision), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, alpha=0.9)

# Add LoRA-tuned models
ax.scatter([0.8542], [0.8962], c='#FF5722', s=200, alpha=0.9, 
           edgecolors='black', linewidth=2, marker='*', zorder=5)
ax.annotate('RoBERTa-Large\n+ LoRA', (0.8542, 0.8962), xytext=(10, -15), 
            textcoords='offset points', fontsize=9, fontweight='bold', color='#FF5722')

ax.scatter([0.8490], [0.8579], c='#FF5722', s=150, alpha=0.9, 
           edgecolors='black', linewidth=2, marker='*', zorder=5)
ax.annotate('MPNet-Base\n+ LoRA', (0.8490, 0.8579), xytext=(10, 5), 
            textcoords='offset points', fontsize=9, fontweight='bold', color='#FF5722')

# Add precision constraint line
ax.axhline(y=0.80, color='#E91E63', linestyle='--', linewidth=2, alpha=0.7)
ax.text(0.62, 0.805, 'Precision ≥ 0.80', fontsize=10, color='#E91E63')

# Legend
legend_elements = [
    Patch(facecolor=v, label=k.capitalize()) for k, v in category_colors.items()
] + [
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#FF5722', 
               markersize=15, label='LoRA Fine-tuned')
]
ax.legend(handles=legend_elements, loc='lower left', title='Category')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
ax.set_xlim(0.60, 0.90)
ax.set_ylim(0.74, 0.92)

plt.tight_layout()
plt.savefig(f'{output_dir}/precision_recall_scatter.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"✓ Created: {output_dir}/precision_recall_scatter.png")

print("\n✅ All visualizations created successfully!")
