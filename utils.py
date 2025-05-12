import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def visualize(data, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    _ = ax1.hist(data, bins="auto", cumulative=True)
    _ = ax2.hist(data, bins="auto", cumulative=False)
    fig.savefig(path)


def normalize(data, path, mode="train"):
    if mode == "train":
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(np.array(data).reshape(-1, 1)).reshape(1, -1)[
            0
        ]
        with open(os.path.join(path, "scaler.pickle"), "wb") as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return normalized
    elif mode == "eval":
        scaler = None
        with open(os.path.join(path, "scaler.pickle"), "rb") as handle:
            scaler = pickle.load(handle)
        normalized = scaler.fit_transform(np.array(data).reshape(-1, 1)).reshape(1, -1)[
            0
        ]
        return normalized
    else:
        raise Exception("mode parameter can only take eval or train as its values")


def create_analysis_visualizations(test_losses, test_probs, ood_losses, ood_probs, alpha, path):
    """
    Create comprehensive visualizations to analyze the ensemble approach
    """
    # Calculate ensemble scores
    test_scores = [alpha * (1 - np.max(p)) + (1 - alpha) * l for l, p in zip(test_losses, test_probs)]
    ood_scores = [alpha * (1 - np.max(p)) + (1 - alpha) * l for l, p in zip(ood_losses, ood_probs)]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Reconstruction Loss Distribution
    axes[0, 0].hist([test_losses, ood_losses], bins=30, alpha=0.7, label=['In-domain', 'OOD'])
    axes[0, 0].set_xlabel('Normalized Reconstruction Loss')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Reconstruction Losses')
    axes[0, 0].legend()
    
    # Plot 2: Max Probability Distribution
    test_max_probs = [np.max(p) for p in test_probs]
    ood_max_probs = [np.max(p) for p in ood_probs]
    axes[0, 1].hist([test_max_probs, ood_max_probs], bins=30, alpha=0.7, label=['In-domain', 'OOD'])
    axes[0, 1].set_xlabel('Max Softmax Probability')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Max Softmax Probabilities')
    axes[0, 1].legend()
    
    # Plot 3: Ensemble Score Distribution
    axes[1, 0].hist([test_scores, ood_scores], bins=30, alpha=0.7, label=['In-domain', 'OOD'])
    axes[1, 0].set_xlabel('Ensemble Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Distribution of Ensemble Scores (alpha={alpha:.2f})')
    axes[1, 0].legend()
    
    # Plot 4: ROC Curve
    y_true = [0] * len(test_scores) + [1] * len(ood_scores)
    y_score = test_scores + ood_scores
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    
    axes[1, 1].plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    axes[1, 1].plot([0, 1], [0, 1], 'k--')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'ensemble_analysis.png'))
    
    # Create scatter plot to see relationship between reconstruction loss and max probability
    plt.figure(figsize=(10, 8))
    plt.scatter([np.max(p) for p in test_probs], test_losses, alpha=0.5, label='In-domain')
    plt.scatter([np.max(p) for p in ood_probs], ood_losses, alpha=0.5, label='OOD')
    plt.xlabel('Max Softmax Probability')
    plt.ylabel('Normalized Reconstruction Loss')
    plt.title('Relationship Between Classifier Confidence and Reconstruction Loss')
    plt.legend()
    plt.savefig(os.path.join(path, 'confidence_vs_loss.png'))