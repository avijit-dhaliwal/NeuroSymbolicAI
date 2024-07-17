import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import List, Dict, Any

def visualize_neuro_symbolic_ai(model_outputs: List[np.ndarray], image: np.ndarray, 
                                predicted_class: int, reasoning_results: Dict[str, Any]):
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)
    
    # Original Image
    ax_original = fig.add_subplot(gs[0, 0])
    ax_original.imshow(image.squeeze(), cmap='gray')
    ax_original.set_title('Original Image')
    ax_original.axis('off')

    # Layer Activations
    for i, output in enumerate(model_outputs[:4]):  # Display first 4 layer outputs
        ax = fig.add_subplot(gs[i//2, i%2+1])
        if len(output.shape) == 4:
            ax.imshow(output[0, :, :, 0], cmap='viridis')
        else:
            ax.imshow(output[0].reshape((8, 8)), cmap='viridis')
        ax.set_title(f'Layer {i+1} Output')
        ax.axis('off')

    # Classification Probabilities
    ax_probs = fig.add_subplot(gs[2, 0])
    probs = model_outputs[-1][0]
    sns.barplot(x=list(range(10)), y=probs, ax=ax_probs)
    ax_probs.set_title('Classification Probabilities')
    ax_probs.set_xlabel('Digit')
    ax_probs.set_ylabel('Probability')

    # Symbolic Reasoning
    ax_symbolic = fig.add_subplot(gs[2, 1])
    G = nx.DiGraph()
    G.add_edge('Input', 'Properties')
    G.add_edge('Input', 'Meta')
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax_symbolic, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold')
    
    properties_text = '\n'.join([f"{k}: {v}" for k, v in reasoning_results.items() 
                                 if k not in ['certainty', 'rule_complexity', 'confidence']])
    meta_text = f"Certainty: {reasoning_results['certainty']:.2f}\n" \
                f"Confidence: {reasoning_results['confidence']:.2f}"
    
    ax_symbolic.text(pos['Properties'][0], pos['Properties'][1]-0.1, properties_text, 
                     ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
    ax_symbolic.text(pos['Meta'][0], pos['Meta'][1]-0.1, meta_text, 
                     ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
    ax_symbolic.set_title('Symbolic Reasoning')

    # Neuro-Symbolic Integration (text-based summary)
    ax_summary = fig.add_subplot(gs[2, 2])
    ax_summary.axis('off')
    summary_text = f"Predicted: {predicted_class}\n\n" \
                   f"Certainty: {reasoning_results['certainty']:.2f}\n" \
                   f"Confidence: {reasoning_results['confidence']:.2f}\n\n" \
                   f"Properties:\n" + '\n'.join([f"{k}: {v}" for k, v in reasoning_results.items() 
                                                 if k not in ['certainty', 'rule_complexity', 'confidence']])
    ax_summary.text(0.5, 0.5, summary_text,
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='black', pad=10))
    ax_summary.set_title('Neuro-Symbolic Integration')

    plt.tight_layout()
    plt.savefig('neuro_symbolic_visualization.png', dpi=300, bbox_inches='tight')
    plt.close(fig)