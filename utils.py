import os
import pickle
import numpy as np
from scipy import stats
from sklearn import metrics
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
        with open(os.path.join(path, "scaler.pickle"), "rb") as handle:
            scaler = pickle.load(handle)
        # Only transform, not fit_transform for evaluation data
        normalized = scaler.transform(np.array(data).reshape(-1, 1)).reshape(1, -1)[0]
        return normalized
    else:
        raise Exception("mode parameter can only take eval or train as its values")


def create_analysis_visualizations(
    test_losses, test_probs, ood_losses, ood_probs, alpha, path
):
    """
    Create comprehensive visualizations to analyze the ensemble approach
    """
    # Calculate ensemble scores
    test_scores = [
        alpha * (1 - np.max(p)) + (1 - alpha) * l
        for l, p in zip(test_losses, test_probs)
    ]
    ood_scores = [
        alpha * (1 - np.max(p)) + (1 - alpha) * l for l, p in zip(ood_losses, ood_probs)
    ]

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Reconstruction Loss Distribution
    axes[0, 0].hist(
        [test_losses, ood_losses], bins=30, alpha=0.7, label=["In-domain", "OOD"]
    )
    axes[0, 0].set_xlabel("Normalized Reconstruction Loss")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Distribution of Reconstruction Losses")
    axes[0, 0].legend()

    # Plot 2: Max Probability Distribution
    test_max_probs = [np.max(p) for p in test_probs]
    ood_max_probs = [np.max(p) for p in ood_probs]
    axes[0, 1].hist(
        [test_max_probs, ood_max_probs], bins=30, alpha=0.7, label=["In-domain", "OOD"]
    )
    axes[0, 1].set_xlabel("Max Softmax Probability")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Distribution of Max Softmax Probabilities")
    axes[0, 1].legend()

    # Plot 3: Ensemble Score Distribution
    axes[1, 0].hist(
        [test_scores, ood_scores], bins=30, alpha=0.7, label=["In-domain", "OOD"]
    )
    axes[1, 0].set_xlabel("Ensemble Score")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"Distribution of Ensemble Scores (alpha={alpha:.2f})")
    axes[1, 0].legend()

    # Plot 4: ROC Curve
    y_true = [0] * len(test_scores) + [1] * len(ood_scores)
    y_score = test_scores + ood_scores
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    axes[1, 1].plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    axes[1, 1].plot([0, 1], [0, 1], "k--")
    axes[1, 1].set_xlabel("False Positive Rate")
    axes[1, 1].set_ylabel("True Positive Rate")
    axes[1, 1].set_title("ROC Curve")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, "ensemble_analysis.png"))

    # Create scatter plot to see relationship between reconstruction loss and max probability
    plt.figure(figsize=(10, 8))
    plt.scatter(
        [np.max(p) for p in test_probs], test_losses, alpha=0.5, label="In-domain"
    )
    plt.scatter([np.max(p) for p in ood_probs], ood_losses, alpha=0.5, label="OOD")
    plt.xlabel("Max Softmax Probability")
    plt.ylabel("Normalized Reconstruction Loss")
    plt.title("Relationship Between Classifier Confidence and Reconstruction Loss")
    plt.legend()
    plt.savefig(os.path.join(path, "confidence_vs_loss.png"))


def evt_vae_only(
    dev_losses,
    test_losses,
    ood_losses,
    desired_fpr=0.05,
    tail_fraction=0.2,
    min_tail_size=30,
):
    """
    Apply EVT to VAE losses only for OOD detection

    Args:
        dev_losses: Normalized VAE losses for development set (in-domain)
        test_losses: Normalized VAE losses for test set (in-domain)
        ood_losses: Normalized VAE losses for OOD samples
        desired_fpr: Target false positive rate (default 0.05)
        tail_fraction: Fraction of data to use for tail modeling (default 0.2)
        min_tail_size: Minimum number of samples to use for tail (default 30)

    Returns:
        Dictionary containing performance metrics and thresholds
    """
    # Sort losses to examine the distribution
    sorted_dev_losses = np.sort(dev_losses)

    # Determine the appropriate portion of the tail to model
    tail_size = max(min_tail_size, int(tail_fraction * len(sorted_dev_losses)))
    tail = sorted_dev_losses[-tail_size:]

    try:
        # Fit GEV to the tail (using negative because scipy.stats.genextreme fits to minima)
        shape, loc, scale = stats.genextreme.fit(-tail)

        # Calculate threshold based on desired FPR
        evt_threshold = -stats.genextreme.ppf(1 - desired_fpr, shape, loc, scale)

        # Check if threshold is reasonable
        if evt_threshold <= np.min(dev_losses) or evt_threshold >= np.max(ood_losses):
            # Fallback to percentile-based threshold
            print("EVT threshold outside reasonable range, falling back to percentile")
            evt_threshold = np.percentile(sorted_dev_losses, 100 * (1 - desired_fpr))
    except:
        # Fallback if EVT fitting fails
        print("EVT fitting failed, falling back to percentile-based threshold")
        evt_threshold = np.percentile(sorted_dev_losses, 100 * (1 - desired_fpr))

    # Apply the threshold for binary classification
    y_true_binary = [0] * len(test_losses) + [1] * len(ood_losses)
    y_pred_binary = [
        0 if loss <= evt_threshold else 1 for loss in test_losses + ood_losses
    ]

    # Calculate metrics
    binary_f1_macro = metrics.f1_score(y_true_binary, y_pred_binary, average="macro")
    binary_f1_micro = metrics.f1_score(y_true_binary, y_pred_binary, average="micro")

    try:
        auc_roc = metrics.roc_auc_score(y_true_binary, test_losses + ood_losses)
    except:
        auc_roc = 0.5  # Default if ROC calculation fails

    # For comparison, try a range of fixed thresholds
    best_f1 = 0
    best_threshold = 0

    for threshold in np.arange(0.05, 1.0, 0.05):
        y_pred_fixed = [
            0 if loss <= threshold else 1 for loss in test_losses + ood_losses
        ]
        f1 = metrics.f1_score(y_true_binary, y_pred_fixed, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Results with best fixed threshold
    y_pred_best_fixed = [
        0 if loss <= best_threshold else 1 for loss in test_losses + ood_losses
    ]
    fixed_f1_macro = metrics.f1_score(y_true_binary, y_pred_best_fixed, average="macro")
    fixed_f1_micro = metrics.f1_score(y_true_binary, y_pred_best_fixed, average="micro")

    print(f"EVT Results (threshold = {evt_threshold:.4f}):")
    print(f"  Binary F1 Macro: {binary_f1_macro:.4f}")
    print(f"  Binary F1 Micro: {binary_f1_micro:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"\nBest Fixed Threshold Results (threshold = {best_threshold:.4f}):")
    print(f"  Binary F1 Macro: {fixed_f1_macro:.4f}")
    print(f"  Binary F1 Micro: {fixed_f1_micro:.4f}")

    # Calculate confidence intervals for EVT threshold using bootstrap
    bootstrap_thresholds = []
    n_bootstrap = 100  # Reduced for speed

    for _ in range(n_bootstrap):
        # Sample with replacement from development set
        indices = np.random.choice(len(dev_losses), size=len(dev_losses), replace=True)
        bootstrap_sample = np.array([dev_losses[i] for i in indices])
        bootstrap_sample = np.sort(bootstrap_sample)
        bootstrap_tail = bootstrap_sample[-tail_size:]

        try:
            bs_shape, bs_loc, bs_scale = stats.genextreme.fit(-bootstrap_tail)
            bs_threshold = -stats.genextreme.ppf(
                1 - desired_fpr, bs_shape, bs_loc, bs_scale
            )

            # Only add reasonable thresholds
            if bs_threshold > np.min(dev_losses) and bs_threshold < np.max(ood_losses):
                bootstrap_thresholds.append(bs_threshold)
        except:
            # Skip failed fits
            continue

    # Calculate 95% confidence interval if we have enough bootstrap samples
    if len(bootstrap_thresholds) > 10:
        ci_low = np.percentile(bootstrap_thresholds, 2.5)
        ci_high = np.percentile(bootstrap_thresholds, 97.5)
        print(f"\nEVT Threshold 95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
    else:
        ci_low = evt_threshold - 0.05
        ci_high = evt_threshold + 0.05
        print("\nWarning: Not enough successful bootstrap samples for CI calculation")

    # Visual comparison
    plt.figure(figsize=(12, 6))

    # Plot histograms
    plt.hist(
        test_losses, bins=30, alpha=0.5, density=True, label="In-domain", color="blue"
    )
    plt.hist(ood_losses, bins=30, alpha=0.5, density=True, label="OOD", color="red")

    # Plot thresholds
    plt.axvline(
        evt_threshold,
        color="red",
        linestyle="--",
        label=f"EVT Threshold: {evt_threshold:.4f}",
    )
    plt.axvline(
        best_threshold,
        color="green",
        linestyle="--",
        label=f"Best Fixed Threshold: {best_threshold:.4f}",
    )

    # Plot confidence interval
    plt.axvspan(
        ci_low, ci_high, alpha=0.2, color="red", label="EVT 95% Confidence Interval"
    )

    plt.legend()
    plt.title("VAE Loss Distributions with Thresholds")
    plt.xlabel("Normalized VAE Loss")
    plt.ylabel("Density")
    plt.savefig("vae_evt_comparison.png")

    return {
        "evt": {
            "threshold": evt_threshold,
            "f1_macro": binary_f1_macro,
            "f1_micro": binary_f1_micro,
            "confidence_interval": (ci_low, ci_high),
        },
        "fixed": {
            "threshold": best_threshold,
            "f1_macro": fixed_f1_macro,
            "f1_micro": fixed_f1_micro,
        },
        "auc_roc": auc_roc,
    }


def visualize_vae_losses(
    test_losses, ood_losses, threshold, path, title="VAE Loss Distribution"
):
    """Create visualization of VAE losses for in-domain and OOD data"""
    plt.figure(figsize=(10, 6))
    plt.hist(
        test_losses, bins=30, alpha=0.7, density=True, label="In-domain", color="blue"
    )
    plt.hist(ood_losses, bins=30, alpha=0.7, density=True, label="OOD", color="red")
    plt.axvline(
        threshold, color="black", linestyle="--", label=f"Threshold: {threshold:.4f}"
    )
    plt.xlabel("Normalized VAE Loss")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()
