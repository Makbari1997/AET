import os
import json
import pickle

import numpy as np
import tensorflow as tf

from scipy import stats
from sklearn import metrics
from sklearn.metrics import roc_auc_score


from data_modules.dataloader import DataLoader
from data_modules.data_utils import *

from model.train import compute_loss
from model.vae import *
from model.model_utils import *
from model.encoder import *

from utils import *


def ensemble_predict(
    classifier, tokenizer, losses, sentences, threshold_dict, alpha, max_length
):
    """
    Enhanced prediction using both classifier confidence and reconstruction loss

    Args:
        classifier: The fine-tuned BERT classifier
        tokenizer: BERT tokenizer
        losses: List of reconstruction losses
        sentences: List of input sentences
        threshold_dict: Dictionary mapping class IDs to thresholds
        alpha: Weight for combining classifier confidence and reconstruction loss
        max_length: Maximum sequence length

    Returns:
        List of predicted labels
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

    print(
        "Loading bert weights from {}".format(
            os.path.join("artifacts", config["dataset"], "bert/")
        )
    )
    classifier = finetune(
        x_train=train_sentences + dev_sentences,
        y_train=np.concatenate((train_intents_encoded, dev_intents_encoded), axis=0),
        x_validation=test_sentences,
        y_validation=test_intents_encoded,
        max_length=max_length,
        num_labels=len(np.unique(np.array(train_intents))),
        path=os.path.join("artifacts", config["dataset"], "bert/"),
        train=False,
        first_layers_to_freeze=11,
        num_epochs=config["finetune_epochs"],
        model_name=config["bert"],
    )
    classifier.load_weights(os.path.join("artifacts", config["dataset"], "bert/"))
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

    model.load_weights(os.path.join("artifacts", config["dataset"], "vae", "vae.h5"))
    print(
        "Model was created successfully and weights were loaded from {}.".format(
            os.path.join("artifacts", config["dataset"], "vae", "vae.h5")
        )
    )

    print('------------------------------------------------------------------')
    
    # Calculate losses for dev, test, and ood sets
    dev_tf = to_tf_format((dev_input_ids, dev_attention_mask, dev_token_type_ids), None, len(dev_sentences), batch_size=1)
    test_tf = to_tf_format((test_input_ids, test_attention_mask, test_token_type_ids), None, len(test_sentences), batch_size=1)
    ood_tf = to_tf_format((ood_input_ids, ood_attention_mask, ood_token_type_ids), None, len(ood_sentences), batch_size=1)
    
    dev_loss = compute_loss(model, dev_tf)
    test_loss = compute_loss(model, test_tf)
    ood_loss = compute_loss(model, ood_tf)
    
    # Normalize losses
    normalized_dev_loss = normalize(dev_loss, path=os.path.join('artifacts', config['dataset']), mode='eval')
    normalized_test_loss = normalize(test_loss, path=os.path.join('artifacts', config['dataset']), mode='eval')
    normalized_ood_loss = normalize(ood_loss, path=os.path.join('artifacts', config['dataset']), mode='eval')
    
    print('------------------------------------------------------------------')
    print('Finding optimal alpha parameter...')
    # Use development set and a portion of OOD data to find optimal alpha
    ood_val_size = min(len(dev_sentences), len(ood_sentences) // 2)
    ood_val_losses = normalized_ood_loss[:ood_val_size]
    ood_val_sentences = ood_sentences[:ood_val_size]
    
    optimal_alpha = find_optimal_alpha(
        classifier, tokenizer, 
        normalized_dev_loss, dev_sentences, [in_lbl_2_indx[i] for i in dev_intents],
        ood_val_losses, ood_val_sentences, max_length
    )
    
    print('------------------------------------------------------------------')
    print('Fitting EVT models for class-specific thresholds...')
    
    # Get true class indices for development set
    dev_class_indices = [in_lbl_2_indx[i] for i in dev_intents]
    
    # Fit EVT models and get thresholds
    thresholds, evt_models = fit_evt_models(
        classifier, tokenizer, normalized_dev_loss, dev_sentences, 
        dev_class_indices, max_length, fpr=0.05
    )
    
    # Save EVT models and thresholds
    evt_path = os.path.join('artifacts', config['dataset'], 'evt')
    os.makedirs(evt_path, exist_ok=True)
    
    with open(os.path.join(evt_path, 'thresholds.pkl'), 'wb') as f:
        pickle.dump(thresholds, f)
    
    with open(os.path.join(evt_path, 'evt_models.pkl'), 'wb') as f:
        pickle.dump(evt_models, f)
    
    print(f'Class-specific thresholds: {thresholds}')
    
    print('------------------------------------------------------------------')
    print('Evaluating on test and OOD data...')
    
    # Combine test and remaining OOD data for evaluation
    ood_test_size = len(ood_sentences) - ood_val_size
    ood_test_losses = normalized_ood_loss[ood_val_size:]
    ood_test_sentences = ood_sentences[ood_val_size:]
    
    eval_losses = normalized_test_loss + ood_test_losses
    eval_sentences = test_sentences + ood_test_sentences
    
    # Make predictions using ensemble approach
    y_pred_multiclass, ood_scores = ensemble_predict(
        classifier, tokenizer, eval_losses, eval_sentences, 
        thresholds, optimal_alpha, max_length
    )
    
    y_true_multiclass = [in_lbl_2_indx[i] for i in test_intents] + [len(in_lbl_2_indx)] * len(ood_test_sentences)
    
    # For binary classification (in-domain vs OOD)
    y_true_binary = [0] * len(test_sentences) + [1] * len(ood_test_sentences)
    y_pred_binary = [0 if y < len(in_lbl_2_indx) else 1 for y in y_pred_multiclass]
    
    # Calculate metrics
    print('alpha : {}'.format(optimal_alpha))
    print('----------------------------------')
    print('multi class macro f1 : {}'.format(np.round(metrics.f1_score(y_true_multiclass, y_pred_multiclass, average='macro'), 4)))
    print('multi class micro f1 : {}'.format(np.round(metrics.f1_score(y_true_multiclass, y_pred_multiclass, average='micro'), 4)))
    print('\n')
    print('binary class macro f1 : {}'.format(np.round(metrics.f1_score(y_true_binary, y_pred_binary, average='macro'), 4)))
    print('binary class micro f1 : {}'.format(np.round(metrics.f1_score(y_true_binary, y_pred_binary, average='micro'), 4)))
    print('AUC-ROC : {}'.format(np.round(metrics.roc_auc_score(y_true_binary, ood_scores), 4)))
    
    # Save visualization of ensemble scores
    plt.figure(figsize=(10, 6))
    plt.hist([ood_scores[:len(test_sentences)], ood_scores[len(test_sentences):]], 
             bins=30, alpha=0.7, label=['In-domain', 'OOD'])
    plt.xlabel('Ensemble Score')
    plt.ylabel('Count')
    plt.title('Distribution of Ensemble Scores (alpha={:.2f})'.format(optimal_alpha))
    plt.legend()
    plt.savefig(os.path.join('artifacts', config['dataset'], 'ensemble_scores.png'))
    
    print('------------------------------------------------------------------')
    
    # Add analysis for each class
    class_metrics = {}
    for cls in set(y_true_multiclass):
        if cls == len(in_lbl_2_indx):  # Skip OOD class
            continue
            
        # Calculate precision, recall, and F1 for this class
        y_true_class = [1 if y == cls else 0 for y in y_true_multiclass]
        y_pred_class = [1 if y == cls else 0 for y in y_pred_multiclass]
        
        precision = metrics.precision_score(y_true_class, y_pred_class)
        recall = metrics.recall_score(y_true_class, y_pred_class)
        f1 = metrics.f1_score(y_true_class, y_pred_class)
        
        class_metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    print('Per-class metrics:')
    for cls, scores in class_metrics.items():
        print(f'Class {cls}: Precision={scores["precision"]:.4f}, Recall={scores["recall"]:.4f}, F1={scores["f1"]:.4f}')

if __name__ == "__main__":
    config_file = open("./config.json")
    config = json.load(config_file)

    run(config)
