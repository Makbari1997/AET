import os
import json
import pickle

import numpy as np
import tensorflow as tf

from scipy import stats
from pathlib import Path
from sklearn import metrics
from sklearn.metrics import roc_auc_score


from data_modules.data_utils import *
from data_modules.dataloader import DataLoader

from model.vae import *
from model.encoder import *
from model.model_utils import *
from model.train import compute_loss

from utils import *


def collect_metrics(
    y_true_multiclass,
    y_pred_multiclass,
    y_true_binary,
    y_pred_binary,
    scores=None,
    threshold=None,
):
    """Collect metrics into a dictionary for easier handling"""
    metrics_dict = {
        "multi_class_macro_f1": metrics.f1_score(
            y_true_multiclass, y_pred_multiclass, average="macro"
        ),
        "multi_class_micro_f1": metrics.f1_score(
            y_true_multiclass, y_pred_multiclass, average="micro"
        ),
        "binary_class_macro_f1": metrics.f1_score(
            y_true_binary, y_pred_binary, average="macro"
        ),
        "binary_class_micro_f1": metrics.f1_score(
            y_true_binary, y_pred_binary, average="micro"
        ),
        "threshold": threshold,
    }

    # Add AUC-ROC if scores are provided
    if scores is not None:
        try:
            metrics_dict["auc_roc"] = metrics.roc_auc_score(y_true_binary, scores)
        except:
            metrics_dict["auc_roc"] = 0.5

    return metrics_dict


def adaptive_alpha(probs):
    """
    Calculate an adaptive alpha based on the classifier's probability distribution

    When classifier is very confident (high max prob), we rely more on the classifier
    When classifier is uncertain (low max prob), we rely more on the VAE
    """
    # Calculate entropy of the probability distribution
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = -np.log(1.0 / len(probs))  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy

    # Calculate adaptive alpha - higher entropy means lower alpha (rely more on VAE)
    return 0.8 * (1 - normalized_entropy) + 0.2


def ensemble_predict_adaptive(
    classifier, tokenizer, losses, sentences, threshold_dict, max_length
):
    """
    Enhanced prediction using adaptive alpha
    """
    labels = []
    ood_scores = []

    for loss, sen in zip(losses, sentences):
        # Get classifier output
        inputs = __predict_preprocess__(sen, tokenizer, max_length)
        logits = classifier.predict(inputs)[0]
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]

        # Get maximum probability and predicted class
        max_prob = np.max(probs)
        predicted_class = np.argmax(probs)

        # Calculate adaptive alpha
        alpha = adaptive_alpha(probs)

        # Calculate ensemble score
        ood_score = alpha * (1 - max_prob) + (1 - alpha) * loss
        ood_scores.append(ood_score)

        # Get threshold for predicted class
        threshold = threshold_dict.get(
            predicted_class, np.mean(list(threshold_dict.values()))
        )

        # Make prediction
        if ood_score > threshold:
            labels.append(len(threshold_dict))  # OOD class
        else:
            labels.append(predicted_class)

    return labels, ood_scores


def ensemble_predict(
    classifier,
    tokenizer,
    losses,
    sentences,
    threshold_dict,
    alpha,
    max_length,
    ensemble_method="fixed",
):
    """
    Enhanced prediction using both classifier confidence and reconstruction loss

    Args:
        classifier: The fine-tuned BERT classifier
        tokenizer: BERT tokenizer
        losses: List of reconstruction losses
        sentences: List of input sentences
        threshold_dict: Dictionary mapping class IDs to thresholds
        alpha: Fixed alpha value (used when ensemble_method is "fixed")
        max_length: Maximum sequence length
        ensemble_method: One of "fixed", "adaptive"

    Returns:
        List of predicted labels
    """
    labels = []
    ood_scores = []
    alphas_used = []  # Store alphas used for analysis

    for loss, sen in zip(losses, sentences):
        # Get classifier output
        inputs = __predict_preprocess__(sen, tokenizer, max_length)
        logits = classifier.predict(inputs)[0]
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]

        # Get maximum probability and predicted class
        max_prob = np.max(probs)
        predicted_class = np.argmax(probs)

        # Determine alpha based on method
        if ensemble_method == "adaptive":
            current_alpha = adaptive_alpha(probs)
        else:  # fixed
            current_alpha = alpha

        alphas_used.append(current_alpha)

        # Calculate ensemble score
        ood_score = current_alpha * (1 - max_prob) + (1 - current_alpha) * loss
        ood_scores.append(ood_score)

        # Get threshold for predicted class
        threshold = threshold_dict.get(
            predicted_class, np.mean(list(threshold_dict.values()))
        )

        # Make prediction
        if ood_score > threshold:
            labels.append(len(threshold_dict))  # OOD class
        else:
            labels.append(predicted_class)

    return labels, ood_scores, alphas_used


def fit_evt_models(
    classifier, tokenizer, losses, sentences, true_classes, max_length, fpr=0.05
):
    """
    Fit EVT models for each class based on reconstruction losses and classifier probabilities

    Args:
        classifier: The fine-tuned BERT classifier
        tokenizer: BERT tokenizer
        losses: List of reconstruction losses for in-domain validation data
        sentences: List of in-domain validation sentences
        true_classes: List of true class indices for in-domain validation data
        max_length: Maximum sequence length
        fpr: Desired false positive rate

    Returns:
        Dictionary mapping class IDs to thresholds
    """
    # Group validation samples by class
    class_to_samples = {}

    for i, (loss, sen, cls) in enumerate(zip(losses, sentences, true_classes)):
        # Get classifier output
        inputs = __predict_preprocess__(sen, tokenizer, max_length)
        logits = classifier.predict(inputs)[0]
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]

        # Get maximum probability
        max_prob = np.max(probs)

        # Calculate ensemble score with alpha=0.5 (will optimize later)
        ood_score = 0.5 * (1 - max_prob) + 0.5 * loss

        if cls not in class_to_samples:
            class_to_samples[cls] = []

        class_to_samples[cls].append(ood_score)

    # Fit EVT models for each class
    evt_models = {}
    thresholds = {}

    for cls, scores in class_to_samples.items():
        # Fit GEV distribution
        scores_array = np.array(scores)
        shape, loc, scale = stats.genextreme.fit(-scores_array)
        evt_models[cls] = (shape, loc, scale)

        # Calculate threshold based on desired FPR
        threshold = -stats.genextreme.ppf(1 - fpr, shape, loc, scale)
        thresholds[cls] = threshold

    return thresholds, evt_models


def find_optimal_alpha(
    classifier,
    tokenizer,
    dev_losses,
    dev_sentences,
    dev_classes,
    ood_losses,
    ood_sentences,
    max_length,
):
    best_alpha = 0.5
    best_auc = 0

    for alpha in np.arange(0.1, 1.0, 0.1):
        dev_scores = []
        ood_scores = []

        # Calculate scores for in-domain validation data
        for loss, sen in zip(dev_losses, dev_sentences):
            inputs = __predict_preprocess__(sen, tokenizer, max_length)
            logits = classifier.predict(inputs)[0]
            probs = tf.nn.softmax(logits, axis=1).numpy()[0]
            max_prob = np.max(probs)
            ood_score = alpha * (1 - max_prob) + (1 - alpha) * loss
            dev_scores.append(ood_score)

        # Calculate scores for OOD validation data
        for loss, sen in zip(ood_losses, ood_sentences):
            inputs = __predict_preprocess__(sen, tokenizer, max_length)
            logits = classifier.predict(inputs)[0]
            probs = tf.nn.softmax(logits, axis=1).numpy()[0]
            max_prob = np.max(probs)
            ood_score = alpha * (1 - max_prob) + (1 - alpha) * loss
            ood_scores.append(ood_score)

        # Create binary labels (0 for in-domain, 1 for OOD)
        y_true = [0] * len(dev_scores) + [1] * len(ood_scores)
        y_score = dev_scores + ood_scores

        # Calculate AUC-ROC
        auc = roc_auc_score(y_true, y_score)
        print(f"Alpha: {alpha:.1f}, AUC-ROC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_alpha = alpha

    print(f"Best alpha: {best_alpha}, Best AUC-ROC: {best_auc:.4f}")
    return best_alpha


def __predict_preprocess__(x, tokenizer, max_length):
    x_tokenized = tokenizer(
        x,
        return_tensors="tf",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    return {i: x_tokenized[i] for i in tokenizer.model_input_names}


def predict(
    classifier: object,
    tokenizer: object,
    losses: list,
    sentences: list,
    threshold: float,
    ood_label: int,
    max_length: int,
) -> list:
    labels = []
    for loss, sen in zip(losses, sentences):
        if loss <= threshold:
            labels.append(
                np.argmax(
                    classifier.predict(
                        __predict_preprocess__(sen, tokenizer, max_length)
                    )[0],
                    axis=1,
                )[0]
            )
        else:
            labels.append(ood_label)
    return labels


def run(config):
    output_dir = config.get("output_dir", f'./artifacts/{config["dataset"]}')
    path_bert = Path(f"{output_dir}/bert/")
    path_vae = Path(f"{output_dir}/vae/")
    path_bert.mkdir(parents=True, exist_ok=True)
    path_vae.mkdir(parents=True, exist_ok=True)
    print("Loading data from {}...".format(os.path.join("data", config["dataset"])))
    dataloader = DataLoader(path=os.path.join("dataset", config["dataset"]))
    train_sentences, train_intents = dataloader.train_loader()
    dev_sentences, dev_intents = dataloader.dev_loader()
    test_sentences, test_intents = dataloader.test_loader()
    ood_sentences, ood_intents = dataloader.ood_loader()
    print("Data is loaded successfully!")

    print("------------------------------------------------------------------")

    print("Encoding intent labels...")
    in_lbl_2_indx = get_lbl_2_indx(
        path=os.path.join("dataset", config["dataset"], "in_lbl_2_indx.txt")
    )

    train_intents_encoded = one_hot_encoder(train_intents, in_lbl_2_indx)
    test_intents_encoded = one_hot_encoder(test_intents, in_lbl_2_indx)
    dev_intents_encoded = one_hot_encoder(dev_intents, in_lbl_2_indx)

    ood_lbl_2_indx = get_lbl_2_indx(
        path=os.path.join("dataset", config["dataset"], "ood_lbl_2_indx.txt"),
        intents=ood_intents,
    )
    ood_intents_encoded = one_hot_encoder(ood_intents, ood_lbl_2_indx)
    print("Encoding done successfully!")

    print("------------------------------------------------------------------")

    max_length = max_sentence_length(train_sentences, policy=config["seq_length"])

    print("Downloading {}".format(config["bert"]))
    bert, tokenizer = get_bert(config["bert"])
    print("Download finished successfully!")

    print("------------------------------------------------------------------")

    print("Preparing data for bert, it may take a few minutes...")
    train_input_ids, train_attention_mask, train_token_type_ids = preprocessing(
        tokenizer, train_sentences, max_length
    )
    test_input_ids, test_attention_mask, test_token_type_ids = preprocessing(
        tokenizer, test_sentences, max_length
    )
    dev_input_ids, dev_attention_mask, dev_token_type_ids = preprocessing(
        tokenizer, dev_sentences, max_length
    )
    ood_input_ids, ood_attention_mask, ood_token_type_ids = preprocessing(
        tokenizer, ood_sentences, max_length
    )

    # train_tf = to_tf_format((train_input_ids, train_attention_mask, train_token_type_ids), None, len(train_sentences), batch_size=1)
    test_tf = to_tf_format(
        (test_input_ids, test_attention_mask, test_token_type_ids),
        None,
        len(test_sentences),
        batch_size=1,
    )
    # dev_tf = to_tf_format((dev_input_ids, dev_attention_mask, dev_token_type_ids), None, len(dev_sentences), batch_size=1)
    ood_tf = to_tf_format(
        (ood_input_ids, ood_attention_mask, ood_token_type_ids),
        None,
        len(ood_sentences),
        batch_size=1,
    )
    print("Data preparation finished successfully!")

    print("------------------------------------------------------------------")

    print("Loading bert weights from {}".format(os.path.join(output_dir, "bert/")))
    classifier = finetune(
        x_train=train_sentences + dev_sentences,
        y_train=np.concatenate((train_intents_encoded, dev_intents_encoded), axis=0),
        x_validation=test_sentences,
        y_validation=test_intents_encoded,
        max_length=max_length,
        num_labels=len(np.unique(np.array(train_intents))),
        path=os.path.join(output_dir, "bert/"),
        train=False,
        first_layers_to_freeze=11,
        num_epochs=config["finetune_epochs"],
        model_name=config["bert"],
    )
    classifier.load_weights(os.path.join(output_dir, "bert/"))
    bert.layers[0].set_weights(classifier.layers[0].get_weights())
    print("------------------------------------------------------------------")

    print("VAE model creation is in progress...")
    model = vae(
        bert=bert,
        encoder=encoder_model(
            (config["vector_dim"],),
            config["latent_dim"],
            dims=config["encoder"],
            activation=config["activation"],
        ),
        decoder=decoder_model(
            (config["latent_dim"],),
            dims=config["decoder"],
            activation=config["activation"],
        ),
        input_shape=((max_length,)),
    )

    model.layers[3].trainable = False
    # optimizer = tf.keras.optimizers.Adam(learning_rate=config['vae_learning_rate'])
    # train_loss_metric = tf.keras.metrics.Mean()
    # val_loss_metric = tf.keras.metrics.Mean()

    model.load_weights(os.path.join(output_dir, "vae", "vae.h5"))
    print(
        "Model was created successfully and weights were loaded from {}.".format(
            os.path.join(output_dir, "vae", "vae.h5")
        )
    )

    print("------------------------------------------------------------------")

    # Calculate losses for dev, test, and ood sets
    dev_tf = to_tf_format(
        (dev_input_ids, dev_attention_mask, dev_token_type_ids),
        None,
        len(dev_sentences),
        batch_size=1,
    )
    test_tf = to_tf_format(
        (test_input_ids, test_attention_mask, test_token_type_ids),
        None,
        len(test_sentences),
        batch_size=1,
    )
    ood_tf = to_tf_format(
        (ood_input_ids, ood_attention_mask, ood_token_type_ids),
        None,
        len(ood_sentences),
        batch_size=1,
    )

    dev_loss = compute_loss(model, dev_tf)
    test_loss = compute_loss(model, test_tf)
    ood_loss = compute_loss(model, ood_tf)

    # Fix normalization - use proper function
    normalized_dev_loss = normalize(
        dev_loss, path=os.path.join(output_dir), mode="eval"
    )
    normalized_test_loss = normalize(
        test_loss, path=os.path.join(output_dir), mode="eval"
    )
    normalized_ood_loss = normalize(
        ood_loss, path=os.path.join(output_dir), mode="eval"
    )

    # Visualize test and OOD losses
    visualize(
        normalized_test_loss,
        os.path.join(
            output_dir,
            "vae_loss_for_{}_test.png".format(config["dataset"]),
        ),
    )
    visualize(
        normalized_ood_loss,
        os.path.join(
            output_dir,
            "vae_loss_for_{}_ood.png".format(config["dataset"]),
        ),
    )

    # Choose detection approach based on configuration
    if config.get("use_evt_vae", False):
        # EVT-VAE Only approach
        print("------------------------------------------------------------------")
        print("Using EVT for VAE losses only")

        # Apply EVT to VAE losses
        evt_results = evt_vae_only(
            normalized_dev_loss,
            normalized_test_loss,
            normalized_ood_loss,
            desired_fpr=config.get("evt_fpr", 0.05),
            tail_fraction=config.get("tail_fraction", 0.2),
            min_tail_size=config.get("min_tail_size", 30),
        )

        # Save EVT results
        evt_path = os.path.join(output_dir, "evt_vae")
        os.makedirs(evt_path, exist_ok=True)

        with open(os.path.join(evt_path, "evt_results.pkl"), "wb") as f:
            pickle.dump(evt_results, f)

        # Visualize VAE losses with EVT threshold
        visualize_vae_losses(
            normalized_test_loss,
            normalized_ood_loss,
            evt_results["evt"]["threshold"],
            os.path.join(evt_path, "vae_losses_evt_threshold.png"),
            title="VAE Loss Distribution with EVT Threshold",
        )

        # Use the EVT threshold for final prediction
        evt_threshold = evt_results["evt"]["threshold"]
        print(f"Using EVT threshold: {evt_threshold:.4f}")

        # Evaluate on test and OOD data
        eval_losses = np.concatenate([normalized_test_loss, normalized_ood_loss])
        eval_sentences = test_sentences + ood_sentences

        # Make predictions using VAE loss threshold
        y_pred_multiclass = predict(
            classifier,
            tokenizer,
            eval_losses,
            eval_sentences,
            evt_threshold,
            len(in_lbl_2_indx),
            max_length,
        )
        y_true_multiclass = [in_lbl_2_indx[i] for i in test_intents] + [
            len(in_lbl_2_indx)
        ] * len(ood_sentences)

        # For binary classification (in-domain vs OOD)
        y_true_binary = [0] * len(test_sentences) + [1] * len(ood_sentences)
        y_pred_binary = [0 if loss <= evt_threshold else 1 for loss in eval_losses]

        metrics_dict = collect_metrics(
            y_true_multiclass,
            y_pred_multiclass,
            y_true_binary,
            y_pred_binary,
            eval_losses,
            evt_threshold,
        )

        # Calculate metrics
        print("----------------------------------")
        print(
            f'Multi class macro f1: {metrics.f1_score(y_true_multiclass, y_pred_multiclass, average="macro"):.4f}'
        )
        print(
            f'Multi class micro f1: {metrics.f1_score(y_true_multiclass, y_pred_multiclass, average="micro"):.4f}'
        )
        print("\n")
        print(
            f'Binary class macro f1: {metrics.f1_score(y_true_binary, y_pred_binary, average="macro"):.4f}'
        )
        print(
            f'Binary class micro f1: {metrics.f1_score(y_true_binary, y_pred_binary, average="micro"):.4f}'
        )

        try:
            auc_roc = metrics.roc_auc_score(y_true_binary, eval_losses)
            print(f"AUC-ROC: {auc_roc:.4f}")
        except:
            print("Could not calculate AUC-ROC")

    elif config.get("use_ensemble", False):
        # Your existing ensemble code - keep this block exactly as it is
        print("------------------------------------------------------------------")
        print(
            f"Using ensemble approach with method: {config.get('ensemble_method', 'fixed')}"
        )

        # Get dev class indices for EVT modeling
        dev_class_indices = [in_lbl_2_indx[i] for i in dev_intents]

        # Collect classifier probabilities for dev set
        dev_probs = []
        for sen in dev_sentences:
            inputs = __predict_preprocess__(sen, tokenizer, max_length)
            logits = classifier.predict(inputs)[0]
            dev_probs.append(tf.nn.softmax(logits, axis=1).numpy()[0])

        # Determine alpha value
        alpha = config.get("initial_alpha", 0.5)

        if (
            config.get("optimize_alpha", True)
            and config.get("ensemble_method", "fixed") == "fixed"
        ):
            print("Finding optimal alpha parameter...")
            # Use development set and a portion of OOD data to find optimal alpha
            ood_val_size = min(len(dev_sentences), len(ood_sentences) // 2)
            ood_val_losses = normalized_ood_loss[:ood_val_size]
            ood_val_sentences = ood_sentences[:ood_val_size]

            # Collect probabilities for OOD validation set
            ood_val_probs = []
            for sen in ood_val_sentences:
                inputs = __predict_preprocess__(sen, tokenizer, max_length)
                logits = classifier.predict(inputs)[0]
                ood_val_probs.append(tf.nn.softmax(logits, axis=1).numpy()[0])

            # Find optimal alpha
            best_alpha = 0.5
            best_auc = 0

            for alpha_test in np.arange(0.1, 1.0, 0.1):
                dev_scores = []
                ood_scores = []

                # Calculate scores for in-domain validation data
                for loss, prob in zip(normalized_dev_loss, dev_probs):
                    max_prob = np.max(prob)
                    ood_score = alpha_test * (1 - max_prob) + (1 - alpha_test) * loss
                    dev_scores.append(ood_score)

                # Calculate scores for OOD validation data
                for loss, prob in zip(ood_val_losses, ood_val_probs):
                    max_prob = np.max(prob)
                    ood_score = alpha_test * (1 - max_prob) + (1 - alpha_test) * loss
                    ood_scores.append(ood_score)

                # Create binary labels (0 for in-domain, 1 for OOD)
                y_true = [0] * len(dev_scores) + [1] * len(ood_scores)
                y_score = dev_scores + ood_scores

                # Calculate AUC-ROC
                auc = roc_auc_score(y_true, y_score)
                print(f"Alpha: {alpha_test:.1f}, AUC-ROC: {auc:.4f}")

                if auc > best_auc:
                    best_auc = auc
                    best_alpha = alpha_test

            print(f"Best alpha: {best_alpha}, Best AUC-ROC: {best_auc:.4f}")
            alpha = best_alpha

        print("------------------------------------------------------------------")
        print("Fitting EVT models for class-specific thresholds...")

        # Prepare data for EVT modeling
        class_scores = {}

        for i, (loss, prob, cls) in enumerate(
            zip(normalized_dev_loss, dev_probs, dev_class_indices)
        ):
            # Get maximum probability
            max_prob = np.max(prob)

            # Calculate ensemble score with current alpha (fixed or optimal)
            if config.get("ensemble_method", "fixed") == "adaptive":
                curr_alpha = adaptive_alpha(prob)
            else:
                curr_alpha = alpha

            ood_score = curr_alpha * (1 - max_prob) + (1 - curr_alpha) * loss

            if cls not in class_scores:
                class_scores[cls] = []

            class_scores[cls].append(ood_score)

        # Fit EVT models for each class
        thresholds = {}
        evt_models = {}

        for cls, scores in class_scores.items():
            # Fit GEV distribution
            scores_array = np.array(scores)
            shape, loc, scale = stats.genextreme.fit(-scores_array)
            evt_models[cls] = (shape, loc, scale)

            # Calculate threshold based on desired FPR
            fpr = config.get("evt_fpr", 0.05)
            threshold = -stats.genextreme.ppf(1 - fpr, shape, loc, scale)
            thresholds[cls] = threshold

        # Save EVT models and thresholds
        evt_path = os.path.join(output_dir, "evt")
        os.makedirs(evt_path, exist_ok=True)

        with open(os.path.join(evt_path, "thresholds.pkl"), "wb") as f:
            pickle.dump(thresholds, f)

        with open(os.path.join(evt_path, "evt_models.pkl"), "wb") as f:
            pickle.dump(evt_models, f)

        print(f"Class-specific thresholds: {thresholds}")

        print("------------------------------------------------------------------")
        print("Evaluating on test and OOD data...")

        # Collect test probabilities
        test_probs = []
        for sen in test_sentences:
            inputs = __predict_preprocess__(sen, tokenizer, max_length)
            logits = classifier.predict(inputs)[0]
            test_probs.append(tf.nn.softmax(logits, axis=1).numpy()[0])

        # Collect OOD probabilities (excluding validation portion if used)
        ood_test_size = len(ood_sentences)
        if config.get("optimize_alpha", True):
            ood_test_size = len(ood_sentences) - min(
                len(dev_sentences), len(ood_sentences) // 2
            )

        ood_test_losses = normalized_ood_loss[-ood_test_size:]
        ood_test_sentences = ood_sentences[-ood_test_size:]

        ood_test_probs = []
        for sen in ood_test_sentences:
            inputs = __predict_preprocess__(sen, tokenizer, max_length)
            logits = classifier.predict(inputs)[0]
            ood_test_probs.append(tf.nn.softmax(logits, axis=1).numpy()[0])

        # Combine test and OOD data
        eval_losses = np.concatenate([normalized_test_loss, ood_test_losses])
        eval_sentences = test_sentences + ood_test_sentences
        eval_probs = test_probs + ood_test_probs

        # Make predictions using ensemble approach
        y_pred_multiclass, ood_scores, alphas_used = ensemble_predict(
            classifier,
            tokenizer,
            eval_losses,
            eval_sentences,
            thresholds,
            alpha,
            max_length,
            ensemble_method=config.get("ensemble_method", "fixed"),
        )

        y_true_multiclass = [in_lbl_2_indx[i] for i in test_intents] + [
            len(in_lbl_2_indx)
        ] * len(ood_test_sentences)

        # For binary classification (in-domain vs OOD)
        y_true_binary = [0] * len(test_sentences) + [1] * len(ood_test_sentences)
        y_pred_binary = [0 if y < len(in_lbl_2_indx) else 1 for y in y_pred_multiclass]

        # Calculate metrics
        if config.get("ensemble_method", "fixed") == "fixed":
            print(f"Fixed alpha : {alpha:.4f}")
        else:
            print(
                f"Adaptive alpha - Average: {np.mean(alphas_used):.4f}, Min: {np.min(alphas_used):.4f}, Max: {np.max(alphas_used):.4f}"
            )
        
        metrics_dict = collect_metrics(
            y_true_multiclass,
            y_pred_multiclass,
            y_true_binary,
            y_pred_binary,
            eval_losses,
            evt_threshold,
        )

        print("----------------------------------")
        print(
            f'Multi class macro f1 : {metrics.f1_score(y_true_multiclass, y_pred_multiclass, average="macro"):.4f}'
        )
        print(
            f'Multi class micro f1 : {metrics.f1_score(y_true_multiclass, y_pred_multiclass, average="micro"):.4f}'
        )
        print("\n")
        print(
            f'Binary class macro f1 : {metrics.f1_score(y_true_binary, y_pred_binary, average="macro"):.4f}'
        )
        print(
            f'Binary class micro f1 : {metrics.f1_score(y_true_binary, y_pred_binary, average="micro"):.4f}'
        )
        print(f"AUC-ROC : {metrics.roc_auc_score(y_true_binary, ood_scores):.4f}")

        # Save visualization of ensemble scores
        plt.figure(figsize=(10, 6))
        plt.hist(
            [ood_scores[: len(test_sentences)], ood_scores[len(test_sentences) :]],
            bins=30,
            alpha=0.7,
            label=["In-domain", "OOD"],
        )
        plt.xlabel("Ensemble Score")
        plt.ylabel("Count")
        if config.get("ensemble_method", "fixed") == "fixed":
            plt.title(f"Distribution of Ensemble Scores (fixed alpha={alpha:.2f})")
        else:
            plt.title(f"Distribution of Ensemble Scores (adaptive alpha)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "ensemble_scores.png"))

        # Create comprehensive visualization
        create_analysis_visualizations(
            normalized_test_loss,
            test_probs,
            ood_test_losses,
            ood_test_probs,
            (
                alpha
                if config.get("ensemble_method", "fixed") == "fixed"
                else np.mean(alphas_used)
            ),
            os.path.join(output_dir),
        )

    else:
        # Simple threshold approach
        print("------------------------------------------------------------------")
        print("Using fixed threshold approach")

        # Use fixed threshold from config
        fixed_threshold = config.get("fixed_threshold", 0.2)
        print(f"Using fixed threshold: {fixed_threshold:.4f}")

        # Visualize VAE losses with fixed threshold
        plt.figure(figsize=(10, 6))
        plt.hist(
            normalized_test_loss, bins=30, alpha=0.7, label="In-domain", color="blue"
        )
        plt.hist(normalized_ood_loss, bins=30, alpha=0.7, label="OOD", color="red")
        plt.axvline(
            fixed_threshold,
            color="black",
            linestyle="--",
            label=f"Threshold: {fixed_threshold:.4f}",
        )
        plt.xlabel("Normalized VAE Loss")
        plt.ylabel("Count")
        plt.title("VAE Loss Distribution with Fixed Threshold")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "vae_losses_fixed_threshold.png"))

        # Combine test and OOD data
        eval_loss = np.concatenate([normalized_test_loss, normalized_ood_loss])
        eval_sentences = test_sentences + ood_sentences

        # Make predictions using fixed threshold
        y_pred_multiclass = predict(
            classifier,
            tokenizer,
            eval_loss,
            eval_sentences,
            fixed_threshold,
            len(in_lbl_2_indx),
            max_length,
        )
        y_true_multiclass = [in_lbl_2_indx[i] for i in test_intents] + [
            len(in_lbl_2_indx)
        ] * len(ood_sentences)

        # For binary classification (in-domain vs OOD)
        y_true_binary = [0] * len(test_sentences) + [1] * len(ood_sentences)
        y_pred_binary = [0 if loss <= fixed_threshold else 1 for loss in eval_loss]
        
        metrics_dict = collect_metrics(
            y_true_multiclass,
            y_pred_multiclass,
            y_true_binary,
            y_pred_binary,
            eval_losses,
            evt_threshold,
        )
        # Calculate metrics
        print("----------------------------------")
        print(
            f'Multi class macro f1: {metrics.f1_score(y_true_multiclass, y_pred_multiclass, average="macro"):.4f}'
        )
        print(
            f'Multi class micro f1: {metrics.f1_score(y_true_multiclass, y_pred_multiclass, average="micro"):.4f}'
        )
        print("\n")
        print(
            f'Binary class macro f1: {metrics.f1_score(y_true_binary, y_pred_binary, average="macro"):.4f}'
        )
        print(
            f'Binary class micro f1: {metrics.f1_score(y_true_binary, y_pred_binary, average="micro"):.4f}'
        )

        try:
            auc_roc = metrics.roc_auc_score(y_true_binary, eval_loss)
            print(f"AUC-ROC: {auc_roc:.4f}")
        except:
            print("Could not calculate AUC-ROC")

    print("------------------------------------------------------------------")
    return metrics_dict


if __name__ == "__main__":
    config_file = open("./config.json")
    config = json.load(config_file)

    run(config)
