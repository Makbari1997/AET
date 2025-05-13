import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Import your existing run function
from predict import run


def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # Force deterministic operations in TensorFlow
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        print("TensorFlow version doesn't support enable_op_determinism")
        # For older TensorFlow versions
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

    return seed


def run_with_seeds(config_path="./config.json"):
    """Run experiments with multiple seeds from config file"""

    # Load config
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Get the seeds from config
    seeds = config.get("random_seeds", [42])  # Default to seed 42 if not specified

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"./results_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reference
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Create dataframe to store results
    results_df = pd.DataFrame(
        columns=[
            "seed",
            "approach",
            "threshold",
            "multi_class_macro_f1",
            "multi_class_micro_f1",
            "binary_class_macro_f1",
            "binary_class_micro_f1",
            "auc_roc",
        ]
    )

    # Run for each seed
    for i, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"Running with seed {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*80}\n")

        # Set all seeds for reproducibility
        set_seeds(seed)

        # Create seed-specific directory
        seed_dir = results_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Modified config for this seed
        seed_config = config.copy()
        seed_config["seed"] = seed
        seed_config["output_dir"] = str(seed_dir)

        # Run your algorithm
        metrics = run(seed_config)

        # Store results
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    [
                        {
                            "seed": seed,
                            "approach": (
                                "evt_vae"
                                if config.get("use_evt_vae", False)
                                else (
                                    "ensemble"
                                    if config.get("use_ensemble", False)
                                    else "fixed_threshold"
                                )
                            ),
                            "threshold": metrics.get("threshold", "N/A"),
                            "multi_class_macro_f1": metrics.get(
                                "multi_class_macro_f1", 0
                            ),
                            "multi_class_micro_f1": metrics.get(
                                "multi_class_micro_f1", 0
                            ),
                            "binary_class_macro_f1": metrics.get(
                                "binary_class_macro_f1", 0
                            ),
                            "binary_class_micro_f1": metrics.get(
                                "binary_class_micro_f1", 0
                            ),
                            "auc_roc": metrics.get("auc_roc", 0),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        # Save intermediate results after each seed
        results_df.to_csv(results_dir / "results.csv", index=False)

    # Generate summary
    print("\n\nResults Summary:")
    print("=" * 80)

    # Calculate average and standard deviation
    summary = results_df.groupby("approach").agg(
        {
            "multi_class_macro_f1": ["mean", "std"],
            "multi_class_micro_f1": ["mean", "std"],
            "binary_class_macro_f1": ["mean", "std"],
            "binary_class_micro_f1": ["mean", "std"],
            "auc_roc": ["mean", "std"],
        }
    )

    # Find best seed for each approach
    best_seeds = results_df.loc[
        results_df.groupby("approach")["binary_class_macro_f1"].idxmax()
    ]
    best_seeds = best_seeds[["approach", "seed", "binary_class_macro_f1"]]
    best_seeds.columns = ["approach", "best_seed", "best_f1_score"]

    # Save summary
    summary.to_csv(results_dir / "summary.csv")
    best_seeds.to_csv(results_dir / "best_seeds.csv", index=False)

    print("\nAverage Metrics:")
    print(summary)
    print("\nBest Seeds:")
    print(best_seeds)
    print(f"\nDetailed results saved to {results_dir}")

    return results_df, summary, best_seeds


if __name__ == "__main__":
    run_with_seeds()
