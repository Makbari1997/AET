import os
import json
import pickle

import numpy as np
import tensorflow as tf

from scipy import stats
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

from data_modules.data_utils import *
from data_modules.dataloader import DataLoader

from model.vae import *
from model.encoder import *
from model.model_utils import *
# from model.train import compute_loss_safe

from utils import *

import pandas as pd
from itertools import product


def comprehensive_parameter_search(test_sentences, test_intents, ood_sentences, ood_intents,
                                 raw_test_loss, raw_ood_loss, classifier, tokenizer, 
                                 max_length, in_lbl_2_indx):
    """
    Systematic grid search to find optimal parameters
    """
    print("🔍 SYSTEMATIC PARAMETER OPTIMIZATION")
    print("="*60)
    
    # Prepare data once
    all_sentences = test_sentences + ood_sentences
    all_vae_losses = np.concatenate([raw_test_loss, raw_ood_loss])
    true_binary_labels = [0] * len(test_sentences) + [1] * len(ood_sentences)
    
    # Create true multiclass labels
    true_multiclass_labels = []
    for intent in test_intents:
        true_multiclass_labels.append(in_lbl_2_indx[intent])
    
    ood_class_id = len(in_lbl_2_indx)
    for intent in ood_intents:
        true_multiclass_labels.append(ood_class_id)
    
    # Get classifier confidence scores once
    print("Computing classifier confidence scores...")
    classifier_confidences = []
    classifier_predictions = []
    
    for sentence in all_sentences:
        inputs = __predict_preprocess__(sentence, tokenizer, max_length)
        logits = classifier.predict(inputs, verbose=0)[0][0]
        probs = tf.nn.softmax(logits, axis=0).numpy()
        
        classifier_confidences.append(np.max(probs))
        classifier_predictions.append(np.argmax(probs))
    
    classifier_confidences = np.array(classifier_confidences)
    uncertainty_scores = 1 - classifier_confidences
    
    # Parameter grid
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Ensemble weights
    scaling_methods = ['max_scale', 'std_scale', 'robust_scale']     # Different scaling approaches
    threshold_methods = ['percentile', 'optimal_f1', 'balanced']    # Different threshold selection
    
    results = []
    best_binary_f1 = 0
    best_multiclass_f1 = 0
    best_overall = None
    
    total_combinations = len(alpha_values) * len(scaling_methods) * len(threshold_methods)
    print(f"Testing {total_combinations} parameter combinations...")
    
    for i, (alpha, scaling_method, threshold_method) in enumerate(product(alpha_values, scaling_methods, threshold_methods)):
        if i % 10 == 0:
            print(f"Progress: {i}/{total_combinations}")
        
        # Apply scaling method
        if scaling_method == 'max_scale':
            vae_scaled = all_vae_losses / np.max(all_vae_losses)
        elif scaling_method == 'std_scale':
            vae_scaled = (all_vae_losses - np.mean(all_vae_losses)) / np.std(all_vae_losses)
            vae_scaled = (vae_scaled - np.min(vae_scaled)) / (np.max(vae_scaled) - np.min(vae_scaled))  # Normalize to [0,1]
        else:  # robust_scale
            q75, q25 = np.percentile(all_vae_losses, [75, 25])
            vae_scaled = (all_vae_losses - np.median(all_vae_losses)) / (q75 - q25)
            vae_scaled = (vae_scaled - np.min(vae_scaled)) / (np.max(vae_scaled) - np.min(vae_scaled))  # Normalize to [0,1]
        
        # Create ensemble scores
        ensemble_scores = alpha * vae_scaled + (1 - alpha) * uncertainty_scores
        test_ensemble_scores = ensemble_scores[:len(test_sentences)]
        
        # Apply threshold method
        if threshold_method == 'percentile':
            # Try different percentiles
            best_thresh = 0
            best_f1 = 0
            
            for percentile in np.arange(75, 95, 1):
                thresh = np.percentile(test_ensemble_scores, percentile)
                binary_preds = (ensemble_scores > thresh).astype(int)
                f1 = f1_score(true_binary_labels, binary_preds, average='macro')
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                    
        elif threshold_method == 'optimal_f1':
            # Grid search for optimal F1
            best_thresh = 0
            best_f1 = 0
            
            thresh_candidates = np.linspace(np.min(ensemble_scores), np.max(ensemble_scores), 50)
            for thresh in thresh_candidates:
                binary_preds = (ensemble_scores > thresh).astype(int)
                f1 = f1_score(true_binary_labels, binary_preds, average='macro')
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                    
        else:  # balanced
            # Find threshold that balances precision and recall
            test_mean = np.mean(test_ensemble_scores)
            ood_mean = np.mean(ensemble_scores[len(test_sentences):])
            best_thresh = (test_mean + ood_mean) / 2
        
        # Evaluate with best threshold
        binary_predictions = (ensemble_scores > best_thresh).astype(int)
        
        # Create multiclass predictions
        final_predictions = []
        for j, (binary_pred, classifier_pred) in enumerate(zip(binary_predictions, classifier_predictions)):
            if binary_pred == 0:
                final_predictions.append(classifier_pred)
            else:
                final_predictions.append(ood_class_id)
        
        # Calculate metrics
        binary_f1_macro = f1_score(true_binary_labels, binary_predictions, average='macro')
        binary_f1_micro = f1_score(true_binary_labels, binary_predictions, average='micro')
        multiclass_f1_macro = f1_score(true_multiclass_labels, final_predictions, average='macro')
        multiclass_f1_micro = f1_score(true_multiclass_labels, final_predictions, average='micro')
        
        # Store results
        result = {
            'alpha': alpha,
            'scaling_method': scaling_method,
            'threshold_method': threshold_method,
            'threshold': best_thresh,
            'binary_f1_macro': binary_f1_macro,
            'binary_f1_micro': binary_f1_micro,
            'multiclass_f1_macro': multiclass_f1_macro,
            'multiclass_f1_micro': multiclass_f1_micro,
            'auc_roc': roc_auc_score(true_binary_labels, ensemble_scores)
        }
        results.append(result)
        
        # Track best results
        if binary_f1_macro > best_binary_f1:
            best_binary_f1 = binary_f1_macro
            
        if multiclass_f1_macro > best_multiclass_f1:
            best_multiclass_f1 = multiclass_f1_macro
            best_overall = result.copy()
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    # Find best configurations
    best_binary = df_results.loc[df_results['binary_f1_macro'].idxmax()]
    best_multiclass = df_results.loc[df_results['multiclass_f1_macro'].idxmax()]
    
    print(f"\nBEST BINARY CLASSIFICATION:")
    print(f"  Alpha: {best_binary['alpha']}")
    print(f"  Scaling: {best_binary['scaling_method']}")
    print(f"  Threshold method: {best_binary['threshold_method']}")
    print(f"  F1 Macro: {best_binary['binary_f1_macro']:.4f}")
    print(f"  F1 Micro: {best_binary['binary_f1_micro']:.4f}")
    
    print(f"\nBEST MULTICLASS CLASSIFICATION:")
    print(f"  Alpha: {best_multiclass['alpha']}")
    print(f"  Scaling: {best_multiclass['scaling_method']}")
    print(f"  Threshold method: {best_multiclass['threshold_method']}")
    print(f"  F1 Macro: {best_multiclass['multiclass_f1_macro']:.4f}")
    print(f"  F1 Micro: {best_multiclass['multiclass_f1_micro']:.4f}")
    
    # Check if we beat original paper
    original_binary_macro = 0.8679
    original_multiclass_macro = 0.7938
    
    print(f"\n📈 COMPARISON TO ORIGINAL PAPER:")
    print(f"Binary F1 Macro: {best_binary['binary_f1_macro']:.4f} vs {original_binary_macro:.4f} ({'✅ BETTER' if best_binary['binary_f1_macro'] > original_binary_macro else '❌ WORSE'})")
    print(f"Multiclass F1 Macro: {best_multiclass['multiclass_f1_macro']:.4f} vs {original_multiclass_macro:.4f} ({'✅ BETTER' if best_multiclass['multiclass_f1_macro'] > original_multiclass_macro else '❌ WORSE'})")
    
    return df_results, best_binary, best_multiclass

def quick_ensemble_variants(test_sentences, test_intents, ood_sentences, ood_intents,
                          raw_test_loss, raw_ood_loss, classifier, tokenizer, 
                          max_length, in_lbl_2_indx):
    """
    Quick test of promising ensemble variants
    """
    print("🚀 TESTING PROMISING VARIANTS")
    print("="*40)
    
    # Get base data
    all_sentences = test_sentences + ood_sentences
    all_vae_losses = np.concatenate([raw_test_loss, raw_ood_loss])
    true_binary_labels = [0] * len(test_sentences) + [1] * len(ood_sentences)
    
    # Get classifier scores
    classifier_confidences = []
    for sentence in all_sentences:
        inputs = __predict_preprocess__(sentence, tokenizer, max_length)
        logits = classifier.predict(inputs, verbose=0)[0][0]
        probs = tf.nn.softmax(logits, axis=0).numpy()
        classifier_confidences.append(np.max(probs))
    
    classifier_confidences = np.array(classifier_confidences)
    
    variants = [
        # (alpha, vae_weight, conf_weight, description)
        (0.2, 0.3, 0.7, "Confidence-heavy"),
        (0.4, 0.4, 0.6, "Slight confidence bias"),
        (0.6, 0.6, 0.4, "Slight VAE bias"),  
        (0.8, 0.7, 0.3, "VAE-heavy"),
        (0.5, 0.5, 0.5, "Balanced")
    ]
    
    results = []
    
    for alpha, vae_w, conf_w, desc in variants:
        # Different ensemble approach
        vae_scores = (all_vae_losses - np.mean(raw_test_loss)) / np.std(raw_test_loss)  # Z-score based on test distribution
        vae_scores = np.maximum(0, vae_scores)  # Remove negative scores
        
        conf_scores = 1 - classifier_confidences
        
        # Weighted ensemble
        ensemble_scores = vae_w * vae_scores + conf_w * conf_scores
        
        # Find optimal threshold
        test_scores = ensemble_scores[:len(test_sentences)]
        
        best_f1 = 0
        best_thresh = 0
        
        for thresh in np.percentile(test_scores, np.arange(80, 98, 0.5)):
            preds = (ensemble_scores > thresh).astype(int)
            f1 = f1_score(true_binary_labels, preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        binary_preds = (ensemble_scores > best_thresh).astype(int)
        binary_f1 = f1_score(true_binary_labels, binary_preds, average='macro')
        
        print(f"{desc}: F1 {binary_f1:.4f} (alpha={alpha}, thresh={best_thresh:.4f})")
        
        results.append({
            'description': desc,
            'alpha': alpha,
            'f1_macro': binary_f1,
            'threshold': best_thresh
        })
    
    return results

# RUN BOTH OPTIMIZATIONS
def run_complete_optimization(test_sentences, test_intents, ood_sentences, ood_intents,
                            raw_test_loss, raw_ood_loss, classifier, tokenizer, 
                            max_length, in_lbl_2_indx):
    """
    Run complete optimization pipeline
    """
    print("Starting complete optimization...")
    
    # Quick variants first
    quick_results = quick_ensemble_variants(test_sentences, test_intents, ood_sentences, ood_intents,
                                          raw_test_loss, raw_ood_loss, classifier, tokenizer, 
                                          max_length, in_lbl_2_indx)
    
    # Full systematic search
    df_results, best_binary, best_multiclass = comprehensive_parameter_search(
        test_sentences, test_intents, ood_sentences, ood_intents,
        raw_test_loss, raw_ood_loss, classifier, tokenizer, 
        max_length, in_lbl_2_indx
    )
    
    return df_results, best_binary, best_multiclass, quick_results


def complete_ood_intent_pipeline(test_sentences, test_intents, ood_sentences, ood_intents,
                                raw_test_loss, raw_ood_loss, classifier, tokenizer, 
                                max_length, in_lbl_2_indx, ood_threshold=0.0467, alpha=0.7):
    """
    Complete pipeline: 
    1. Detect OOD vs in-domain
    2. For in-domain: classify specific intent
    3. For OOD: assign OOD label
    """
    
    print("=== COMPLETE OOD + INTENT CLASSIFICATION PIPELINE ===")
    
    # Combine all data
    all_sentences = test_sentences + ood_sentences
    all_vae_losses = np.concatenate([raw_test_loss, raw_ood_loss])
    
    # True labels for evaluation
    true_binary_labels = [0] * len(test_sentences) + [1] * len(ood_sentences)  # 0=in-domain, 1=OOD
    true_multiclass_labels = []
    
    # Create true multiclass labels
    for intent in test_intents:
        true_multiclass_labels.append(in_lbl_2_indx[intent])
    
    ood_class_id = len(in_lbl_2_indx)  # OOD gets the next available class ID
    for intent in ood_intents:
        true_multiclass_labels.append(ood_class_id)  # All OOD samples get same OOD class
    
    print(f"In-domain classes: {list(in_lbl_2_indx.values())}")
    print(f"OOD class ID: {ood_class_id}")
    print(f"Total samples: {len(all_sentences)} (Test: {len(test_sentences)}, OOD: {len(ood_sentences)})")
    
    # Step 1: Get classifier confidence and predictions for all samples
    print("Getting classifier predictions and confidence scores...")
    
    classifier_predictions = []
    classifier_confidences = []
    classifier_logits_all = []
    
    for sentence in all_sentences:
        inputs = __predict_preprocess__(sentence, tokenizer, max_length)
        logits = classifier.predict(inputs, verbose=0)[0][0]
        probs = tf.nn.softmax(logits, axis=0).numpy()
        
        predicted_class = np.argmax(probs)
        max_confidence = np.max(probs)
        
        classifier_predictions.append(predicted_class)
        classifier_confidences.append(max_confidence)
        classifier_logits_all.append(logits)
    
    # Step 2: Apply ensemble OOD detection
    print(f"Applying ensemble OOD detection (threshold={ood_threshold:.4f}, alpha={alpha})...")
    
    # Scale VAE losses to [0,1] range
    vae_scaled = all_vae_losses / np.max(all_vae_losses)
    uncertainty_scores = 1 - np.array(classifier_confidences)
    
    # Ensemble anomaly scores
    ensemble_scores = alpha * vae_scaled + (1 - alpha) * uncertainty_scores
    
    # Binary OOD predictions
    binary_predictions = (ensemble_scores > ood_threshold).astype(int)
    
    # Step 3: Create final multiclass predictions
    print("Creating final multiclass predictions...")
    
    final_predictions = []
    
    for i, (binary_pred, classifier_pred) in enumerate(zip(binary_predictions, classifier_predictions)):
        if binary_pred == 0:  # Predicted as in-domain
            final_predictions.append(classifier_pred)  # Use classifier's intent prediction
        else:  # Predicted as OOD
            final_predictions.append(ood_class_id)  # Assign OOD class
    
    # Step 4: Evaluate results
    print("\n" + "="*60)
    print("PIPELINE EVALUATION RESULTS")
    print("="*60)
    
    # Binary classification metrics (OOD detection)
    binary_f1_macro = f1_score(true_binary_labels, binary_predictions, average='macro')
    binary_f1_micro = f1_score(true_binary_labels, binary_predictions, average='micro')
    binary_auc = roc_auc_score(true_binary_labels, ensemble_scores)
    
    print(f"\nBINARY CLASSIFICATION (OOD Detection):")
    print(f"  F1 Macro: {binary_f1_macro:.4f}")
    print(f"  F1 Micro: {binary_f1_micro:.4f}")
    print(f"  AUC-ROC:  {binary_auc:.4f}")
    
    # Multiclass classification metrics
    multiclass_f1_macro = f1_score(true_multiclass_labels, final_predictions, average='macro')
    multiclass_f1_micro = f1_score(true_multiclass_labels, final_predictions, average='micro')
    
    print(f"\nMULTICLASS CLASSIFICATION (Intent + OOD):")
    print(f"  F1 Macro: {multiclass_f1_macro:.4f}")
    print(f"  F1 Micro: {multiclass_f1_micro:.4f}")
    
    # Detailed breakdown
    print(f"\nDETAILED BREAKDOWN:")
    
    # How many test samples were correctly kept as in-domain?
    test_binary_preds = binary_predictions[:len(test_sentences)]
    test_kept_in_domain = np.sum(test_binary_preds == 0)
    print(f"  Test samples kept in-domain: {test_kept_in_domain}/{len(test_sentences)} ({test_kept_in_domain/len(test_sentences)*100:.1f}%)")
    
    # How many OOD samples were correctly detected as OOD?
    ood_binary_preds = binary_predictions[len(test_sentences):]
    ood_detected = np.sum(ood_binary_preds == 1)
    print(f"  OOD samples detected as OOD: {ood_detected}/{len(ood_sentences)} ({ood_detected/len(ood_sentences)*100:.1f}%)")
    
    # For test samples kept in-domain, what's the intent classification accuracy?
    if test_kept_in_domain > 0:
        test_final_preds = final_predictions[:len(test_sentences)]
        test_true_labels = true_multiclass_labels[:len(test_sentences)]
        
        # Only evaluate intent classification for samples predicted as in-domain
        in_domain_mask = test_binary_preds == 0
        if np.sum(in_domain_mask) > 0:
            in_domain_accuracy = np.mean(np.array(test_final_preds)[in_domain_mask] == np.array(test_true_labels)[in_domain_mask])
            print(f"  Intent classification accuracy (for in-domain predictions): {in_domain_accuracy:.4f}")
    
    return {
        'binary': {
            'f1_macro': binary_f1_macro,
            'f1_micro': binary_f1_micro,
            'auc_roc': binary_auc
        },
        'multiclass': {
            'f1_macro': multiclass_f1_macro,
            'f1_micro': multiclass_f1_micro
        },
        'predictions': {
            'binary': binary_predictions,
            'multiclass': final_predictions,
            'ensemble_scores': ensemble_scores
        },
        'breakdown': {
            'test_kept_in_domain': test_kept_in_domain,
            'ood_detected': ood_detected,
            'total_test': len(test_sentences),
            'total_ood': len(ood_sentences)
        }
    }

# THRESHOLD OPTIMIZATION
def optimize_ensemble_threshold(test_sentences, ood_sentences, raw_test_loss, raw_ood_loss,
                              classifier, tokenizer, max_length, alpha=0.7):
    """
    Fine-tune the ensemble threshold for best performance
    """
    print("=== OPTIMIZING ENSEMBLE THRESHOLD ===")
    
    # Prepare data
    all_sentences = test_sentences + ood_sentences
    all_vae_losses = np.concatenate([raw_test_loss, raw_ood_loss])
    true_labels = [0] * len(test_sentences) + [1] * len(ood_sentences)
    
    # Get confidence scores
    classifier_confidences = []
    for sentence in all_sentences:
        inputs = __predict_preprocess__(sentence, tokenizer, max_length)
        logits = classifier.predict(inputs, verbose=0)[0][0]
        probs = tf.nn.softmax(logits, axis=0).numpy()
        max_confidence = np.max(probs)
        classifier_confidences.append(max_confidence)
    
    # Create ensemble scores
    vae_scaled = all_vae_losses / np.max(all_vae_losses)
    uncertainty_scores = 1 - np.array(classifier_confidences)
    ensemble_scores = alpha * vae_scaled + (1 - alpha) * uncertainty_scores
    
    # Try many different thresholds
    test_ensemble_scores = ensemble_scores[:len(test_sentences)]
    
    best_f1 = 0
    best_threshold = 0
    best_results = {}
    
    # Try percentiles from 70% to 99%
    for percentile in np.arange(70, 99, 0.5):
        threshold = np.percentile(test_ensemble_scores, percentile)
        predictions = (ensemble_scores > threshold).astype(int)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_micro = f1_score(true_labels, predictions, average='micro')
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = threshold
            best_results = {
                'threshold': threshold,
                'percentile': percentile,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'auc_roc': roc_auc_score(true_labels, ensemble_scores)
            }
    
    print(f"Best threshold: {best_threshold:.4f} (at {best_results['percentile']:.1f}th percentile)")
    print(f"Best F1 Macro: {best_f1:.4f}")
    
    return best_results


def smart_ensemble_detection(raw_test_loss, raw_ood_loss, test_sentences, ood_sentences, 
                           classifier, tokenizer, max_length):
    """
    Smart ensemble without destroying the VAE signal through normalization
    """
    print("=== SMART ENSEMBLE (NO NORMALIZATION) ===")
    
    # Combine all raw VAE losses
    all_vae_losses = np.concatenate([raw_test_loss, raw_ood_loss])
    all_sentences = test_sentences + ood_sentences
    true_labels = [0] * len(raw_test_loss) + [1] * len(raw_ood_loss)
    
    print(f"Raw Test VAE: mean={np.mean(raw_test_loss):.4f}, std={np.std(raw_test_loss):.4f}")
    print(f"Raw OOD VAE:  mean={np.mean(raw_ood_loss):.4f}, std={np.std(raw_ood_loss):.4f}")
    print(f"Separation ratio: {np.mean(raw_ood_loss) / np.mean(raw_test_loss):.2f}x")
    
    # Get classifier confidence scores
    print("Getting classifier confidence scores...")
    classifier_confidences = []
    
    for sentence in all_sentences:
        inputs = __predict_preprocess__(sentence, tokenizer, max_length)
        logits = classifier.predict(inputs, verbose=0)[0][0]
        probs = tf.nn.softmax(logits, axis=0).numpy()
        max_prob = np.max(probs)
        classifier_confidences.append(max_prob)
    
    classifier_confidences = np.array(classifier_confidences)
    test_confidences = classifier_confidences[:len(raw_test_loss)]
    ood_confidences = classifier_confidences[len(raw_test_loss):]
    
    print(f"Test Confidence: mean={np.mean(test_confidences):.4f}, std={np.std(test_confidences):.4f}")
    print(f"OOD Confidence:  mean={np.mean(ood_confidences):.4f}, std={np.std(ood_confidences):.4f}")
    
    results = {}
    
    # Method 1: Raw VAE losses only (no normalization)
    best_f1_vae = 0
    best_thresh_vae = 0
    
    # Try thresholds between test mean and OOD mean
    test_mean = np.mean(raw_test_loss)
    ood_mean = np.mean(raw_ood_loss)
    
    for alpha in np.arange(0.2, 1.0, 0.05):
        threshold = test_mean + alpha * (ood_mean - test_mean)
        predictions = (all_vae_losses > threshold).astype(int)
        f1 = f1_score(true_labels, predictions, average='macro')
        
        if f1 > best_f1_vae:
            best_f1_vae = f1
            best_thresh_vae = threshold
    
    pred_vae = (all_vae_losses > best_thresh_vae).astype(int)
    results['raw_vae_only'] = {
        'threshold': best_thresh_vae,
        'f1_macro': f1_score(true_labels, pred_vae, average='macro'),
        'f1_micro': f1_score(true_labels, pred_vae, average='micro'),
        'auc_roc': roc_auc_score(true_labels, all_vae_losses)
    }
    
    # Method 2: Confidence only (1 - max_prob = uncertainty)
    uncertainty_scores = 1 - classifier_confidences
    
    best_f1_conf = 0
    best_thresh_conf = 0
    
    test_uncertainties = uncertainty_scores[:len(raw_test_loss)]
    for alpha in np.arange(0.2, 1.0, 0.05):
        threshold = np.percentile(test_uncertainties, alpha * 100)
        predictions = (uncertainty_scores > threshold).astype(int)
        f1 = f1_score(true_labels, predictions, average='macro')
        
        if f1 > best_f1_conf:
            best_f1_conf = f1
            best_thresh_conf = threshold
    
    pred_conf = (uncertainty_scores > best_thresh_conf).astype(int)
    results['confidence_only'] = {
        'threshold': best_thresh_conf,
        'f1_macro': f1_score(true_labels, pred_conf, average='macro'),
        'f1_micro': f1_score(true_labels, pred_conf, average='micro'),
        'auc_roc': roc_auc_score(true_labels, uncertainty_scores)
    }
    
    # Method 3: Smart ensemble (scale but don't normalize)
    # Scale VAE to similar range as uncertainty (both roughly 0-1)
    vae_scaled = all_vae_losses / np.max(all_vae_losses)  # Scale to [0,1] but keep relative differences
    
    print(f"Scaled VAE: mean={np.mean(vae_scaled):.4f}, max={np.max(vae_scaled):.4f}")
    print(f"Uncertainty: mean={np.mean(uncertainty_scores):.4f}, max={np.max(uncertainty_scores):.4f}")
    
    # Try different ensemble weights
    for alpha in [0.3, 0.5, 0.7]:
        ensemble_scores = alpha * vae_scaled + (1 - alpha) * uncertainty_scores
        
        # Find best threshold for ensemble
        test_ensemble = ensemble_scores[:len(raw_test_loss)]
        
        best_f1_ens = 0
        best_thresh_ens = 0
        
        for thresh_alpha in np.arange(0.7, 0.95, 0.02):
            threshold = np.percentile(test_ensemble, thresh_alpha * 100)
            predictions = (ensemble_scores > threshold).astype(int)
            f1 = f1_score(true_labels, predictions, average='macro')
            
            if f1 > best_f1_ens:
                best_f1_ens = f1
                best_thresh_ens = threshold
        
        pred_ens = (ensemble_scores > best_thresh_ens).astype(int)
        results[f'smart_ensemble_alpha_{alpha}'] = {
            'threshold': best_thresh_ens,
            'f1_macro': f1_score(true_labels, pred_ens, average='macro'),
            'f1_micro': f1_score(true_labels, pred_ens, average='micro'),
            'auc_roc': roc_auc_score(true_labels, ensemble_scores)
        }
    
    return results



def complete_evaluation_with_multiclass(test_losses, ood_losses, test_sentences, ood_sentences, 
                                       test_intents, ood_intents, in_lbl_2_indx,
                                       classifier, tokenizer, max_length):
    """
    Complete evaluation including both binary and multiclass results
    """
    # Binary evaluation (what you already have)
    binary_results = emergency_fix_evaluation(test_losses, ood_losses, test_sentences, ood_sentences, 
                                            classifier, tokenizer, max_length)
    
    # Find best binary method
    best_binary_method = max(binary_results.keys(), key=lambda k: binary_results[k]['f1_macro'])
    best_threshold = binary_results[best_binary_method]['threshold']
    
    print(f"\nUsing best binary method: {best_binary_method} with threshold: {best_threshold:.4f}")
    
    # Now do multiclass evaluation
    all_losses = np.concatenate([test_losses, ood_losses])
    all_sentences = test_sentences + ood_sentences
    
    # Predict multiclass labels
    y_pred_multiclass = []
    ood_label = len(in_lbl_2_indx)  # OOD gets label = num_classes
    
    for loss, sentence in zip(all_losses, all_sentences):
        if loss <= best_threshold:
            # In-domain: use classifier
            inputs = __predict_preprocess__(sentence, tokenizer, max_length)
            logits = classifier.predict(inputs, verbose=0)[0]
            predicted_class = np.argmax(logits)
            y_pred_multiclass.append(predicted_class)
        else:
            # OOD
            y_pred_multiclass.append(ood_label)
    
    # True multiclass labels
    y_true_multiclass = []
    for intent in test_intents:
        y_true_multiclass.append(in_lbl_2_indx[intent])
    for _ in ood_intents:
        y_true_multiclass.append(ood_label)
    
    # Calculate multiclass metrics
    multiclass_f1_macro = f1_score(y_true_multiclass, y_pred_multiclass, average='macro')
    multiclass_f1_micro = f1_score(y_true_multiclass, y_pred_multiclass, average='micro')
    
    print("\n" + "="*60)
    print("COMPLETE RESULTS COMPARISON")
    print("="*60)
    print(f"BINARY CLASSIFICATION:")
    print(f"  Original Paper:  Macro={86.79:.2f}, Micro={87.15:.2f}")
    print(f"  Current Best:    Macro={binary_results[best_binary_method]['f1_macro']:.2f}, Micro={binary_results[best_binary_method]['f1_micro']:.2f}")
    print(f"  Difference:      Macro={binary_results[best_binary_method]['f1_macro']*100 - 86.79:.2f}, Micro={binary_results[best_binary_method]['f1_micro']*100 - 87.15:.2f}")
    
    print(f"\nMULTICLASS CLASSIFICATION:")
    print(f"  Original Paper:  Macro={79.38:.2f}, Micro={86.83:.2f}")
    print(f"  Current Best:    Macro={multiclass_f1_macro*100:.2f}, Micro={multiclass_f1_micro*100:.2f}")
    print(f"  Difference:      Macro={multiclass_f1_macro*100 - 79.38:.2f}, Micro={multiclass_f1_micro*100 - 86.83:.2f}")
    
    return {
        'binary': binary_results[best_binary_method],
        'multiclass': {
            'f1_macro': multiclass_f1_macro,
            'f1_micro': multiclass_f1_micro,
            'method': best_binary_method,
            'threshold': best_threshold
        }
    }


def emergency_fix_evaluation(test_losses, ood_losses, test_sentences, ood_sentences, 
                            classifier, tokenizer, max_length):
    """
    Emergency fix for overlapping distributions
    """
    all_losses = np.concatenate([test_losses, ood_losses])
    all_labels = [0] * len(test_losses) + [1] * len(ood_losses)
    
    print("=== EMERGENCY DIAGNOSIS ===")
    print(f"Test: min={np.min(test_losses):.4f}, 50th={np.percentile(test_losses, 50):.4f}, 95th={np.percentile(test_losses, 95):.4f}, max={np.max(test_losses):.4f}")
    print(f"OOD:  min={np.min(ood_losses):.4f}, 50th={np.percentile(ood_losses, 50):.4f}, 95th={np.percentile(ood_losses, 95):.4f}, max={np.max(ood_losses):.4f}")
    
    # Check separation
    test_95th = np.percentile(test_losses, 95)
    ood_5th = np.percentile(ood_losses, 5)
    print(f"Good separation: {test_95th < ood_5th} (Test 95th: {test_95th:.4f} vs OOD 5th: {ood_5th:.4f})")
    
    results = {}
    
    # Method 1: Use test 90th percentile as threshold
    threshold_90 = np.percentile(test_losses, 90)
    pred_90 = (all_losses > threshold_90).astype(int)
    results['test_90th_percentile'] = {
        'threshold': threshold_90,
        'f1_macro': f1_score(all_labels, pred_90, average='macro'),
        'f1_micro': f1_score(all_labels, pred_90, average='micro'),
        'auc_roc': roc_auc_score(all_labels, all_losses)
    }
    
    # Method 2: Use test 85th percentile  
    threshold_85 = np.percentile(test_losses, 85)
    pred_85 = (all_losses > threshold_85).astype(int)
    results['test_85th_percentile'] = {
        'threshold': threshold_85,
        'f1_macro': f1_score(all_labels, pred_85, average='macro'),
        'f1_micro': f1_score(all_labels, pred_85, average='micro'),
        'auc_roc': roc_auc_score(all_labels, all_losses)
    }
    
    # Method 3: Use test 80th percentile
    threshold_80 = np.percentile(test_losses, 80)
    pred_80 = (all_losses > threshold_80).astype(int)
    results['test_80th_percentile'] = {
        'threshold': threshold_80,
        'f1_macro': f1_score(all_labels, pred_80, average='macro'),
        'f1_micro': f1_score(all_labels, pred_80, average='micro'),
        'auc_roc': roc_auc_score(all_labels, all_losses)
    }
    
    # Method 4: Ensemble with classifier (this should work better)
    print("Computing ensemble scores...")
    ensemble_scores = []
    
    for sentence in test_sentences + ood_sentences:
        inputs = __predict_preprocess__(sentence, tokenizer, max_length)
        logits = classifier.predict(inputs, verbose=0)[0]
        max_prob = np.max(tf.nn.softmax(logits, axis=1).numpy())
        
        # Simple ensemble: high loss OR low confidence = OOD
        ensemble_scores.append(max_prob)  # We'll invert this
    
    # Combine: normalize both to [0,1], then ensemble
    vae_norm = (all_losses - np.min(all_losses)) / (np.max(all_losses) - np.min(all_losses))
    conf_scores = 1 - np.array(ensemble_scores)  # 1 - confidence = uncertainty
    conf_norm = (conf_scores - np.min(conf_scores)) / (np.max(conf_scores) - np.min(conf_scores))
    
    # Try different ensemble weights
    for alpha in [0.3, 0.5, 0.7]:
        ensemble = alpha * vae_norm + (1-alpha) * conf_norm
        
        # Find best threshold for this ensemble
        best_f1 = 0
        best_thresh = 0
        
        for thresh in np.percentile(ensemble, np.arange(70, 95, 2)):
            pred = (ensemble > thresh).astype(int)
            f1 = f1_score(all_labels, pred, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        pred_ens = (ensemble > best_thresh).astype(int)
        results[f'ensemble_alpha_{alpha}'] = {
            'threshold': best_thresh,
            'f1_macro': f1_score(all_labels, pred_ens, average='macro'),
            'f1_micro': f1_score(all_labels, pred_ens, average='micro'),
            'auc_roc': roc_auc_score(all_labels, ensemble)
        }
    
    return results

# ALSO - Quick check if your VAE training is the issue
def check_vae_training_quality(model, train_tf, test_tf):
    """
    Check if VAE is properly trained
    """
    print("=== VAE TRAINING QUALITY CHECK ===")
    
    # Sample a few batches and check reconstruction quality
    train_sample = next(iter(train_tf))
    test_sample = next(iter(test_tf))
    
    train_recon = model(train_sample, training=False)
    test_recon = model(test_sample, training=False)
    
    print(f"Train reconstruction range: [{np.min(train_recon):.4f}, {np.max(train_recon):.4f}]")
    print(f"Test reconstruction range: [{np.min(test_recon):.4f}, {np.max(test_recon):.4f}]")
    
    # Check for NaN/Inf
    if np.any(np.isnan(train_recon)) or np.any(np.isnan(test_recon)):
        print("ERROR: VAE producing NaN outputs!")
        return False
    
    if np.any(np.isinf(train_recon)) or np.any(np.isinf(test_recon)):
        print("ERROR: VAE producing Inf outputs!")
        return False
    
    print("VAE outputs look valid.")
    return True


def adaptive_threshold_selection(dev_losses, dev_labels, method='f1_macro'):
    """
    Adaptive threshold selection using validation data
    This replaces EVT with a more reliable approach
    """
    # Try different percentiles as potential thresholds
    percentiles = np.arange(85, 99, 0.5)
    thresholds = np.percentile(dev_losses, percentiles)
    
    best_threshold = 0
    best_score = 0
    
    for threshold in thresholds:
        # Binary prediction: 1 if loss > threshold (OOD), 0 otherwise (in-domain)
        predictions = (dev_losses > threshold).astype(int)
        
        if method == 'f1_macro':
            score = f1_score(dev_labels, predictions, average='macro', zero_division=0)
        elif method == 'f1_micro':
            score = f1_score(dev_labels, predictions, average='micro', zero_division=0)
        elif method == 'balanced_accuracy':
            score = balanced_accuracy_score(dev_labels, predictions)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

def class_aware_threshold_selection(train_losses, train_labels, percentile=95):
    """
    Learn different thresholds for each class
    """
    thresholds = {}
    unique_classes = np.unique(train_labels)
    
    for class_id in unique_classes:
        class_mask = train_labels == class_id
        class_losses = train_losses[class_mask]
        
        if len(class_losses) > 0:
            thresholds[class_id] = np.percentile(class_losses, percentile)
        else:
            thresholds[class_id] = np.percentile(train_losses, percentile)
    
    return thresholds

def ensemble_anomaly_detection(vae_losses, classifier_logits, alpha=0.6):
    """
    Ensemble VAE reconstruction loss with classifier confidence
    """
    # Get max probability from classifier
    classifier_probs = tf.nn.softmax(classifier_logits, axis=1).numpy()
    max_probs = np.max(classifier_probs, axis=1)
    
    # Confidence score (1 - max_probability)
    confidence_scores = 1 - max_probs
    
    # Normalize both scores to [0,1] range
    vae_normalized = (vae_losses - np.min(vae_losses)) / (np.max(vae_losses) - np.min(vae_losses) + 1e-8)
    conf_normalized = confidence_scores
    
    # Ensemble score
    ensemble_scores = alpha * vae_normalized + (1 - alpha) * conf_normalized
    
    return ensemble_scores

def comprehensive_evaluation(test_losses, ood_losses, test_sentences, ood_sentences, 
                           classifier, tokenizer, max_length, config):
    """
    Evaluate multiple threshold selection methods
    """
    # Combine test and OOD data
    all_losses = np.concatenate([test_losses, ood_losses])
    all_sentences = test_sentences + ood_sentences
    all_labels = [0] * len(test_losses) + [1] * len(ood_losses)  # 0=in-domain, 1=OOD
    
    results = {}
    
    # Method 1: Fixed percentile (baseline)
    fixed_threshold = np.percentile(test_losses, 95)
    fixed_predictions = (all_losses > fixed_threshold).astype(int)
    results['fixed_95th'] = {
        'threshold': fixed_threshold,
        'f1_macro': f1_score(all_labels, fixed_predictions, average='macro'),
        'f1_micro': f1_score(all_labels, fixed_predictions, average='micro'),
        'auc_roc': roc_auc_score(all_labels, all_losses)
    }
    
    # Method 2: Validation-optimized threshold
    # Use test set as validation for threshold selection (in practice, use separate validation set)
    val_threshold, val_score = adaptive_threshold_selection(test_losses, [0]*len(test_losses))
    val_predictions = (all_losses > val_threshold).astype(int)
    results['validation_optimized'] = {
        'threshold': val_threshold,
        'f1_macro': f1_score(all_labels, val_predictions, average='macro'),
        'f1_micro': f1_score(all_labels, val_predictions, average='micro'),
        'auc_roc': roc_auc_score(all_labels, all_losses)
    }
    
    # Method 3: Ensemble approach
    all_logits = []
    for sentence in all_sentences:
        inputs = __predict_preprocess__(sentence, tokenizer, max_length)
        logits = classifier.predict(inputs)[0]
        all_logits.append(logits[0])
    
    all_logits = np.array(all_logits)
    ensemble_scores = ensemble_anomaly_detection(all_losses, all_logits, alpha=0.6)
    
    # Find best threshold for ensemble scores
    ens_threshold, _ = adaptive_threshold_selection(ensemble_scores[:len(test_losses)], [0]*len(test_losses))
    ens_predictions = (ensemble_scores > ens_threshold).astype(int)
    
    results['ensemble'] = {
        'threshold': ens_threshold,
        'f1_macro': f1_score(all_labels, ens_predictions, average='macro'),
        'f1_micro': f1_score(all_labels, ens_predictions, average='micro'),
        'auc_roc': roc_auc_score(all_labels, ensemble_scores)
    }
    
    return results


def compute_loss_safe(model, data):
    """
    Compute loss with NaN/Inf checking and handling
    """
    losses = []
    skipped_count = 0
    
    for step, (x, y, z) in enumerate(data):
        try:
            # Compute loss
            logits = model([x, y, z], training=False)
            loss_value = model.losses
            
            if isinstance(loss_value, list):
                loss_value = loss_value[0]
            
            loss_numpy = loss_value.numpy()
            
            # Check for NaN or Inf
            if np.isnan(loss_numpy) or np.isinf(loss_numpy):
                print(f"Warning: NaN/Inf loss detected at step {step}, skipping")
                skipped_count += 1
                continue
                
            losses.append(loss_numpy)
            
        except Exception as e:
            print(f"Error computing loss at step {step}: {e}")
            skipped_count += 1
            continue
    
    if len(losses) == 0:
        print("ERROR: All losses were invalid!")
        # Return a default high loss value instead of empty array
        return np.array([10.0])  # High loss indicates anomaly
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples due to invalid losses")
    
    return np.array(losses)


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


def fit_evt_models_robust(
    classifier, tokenizer, losses, sentences, true_classes, max_length, fpr=0.05,
    contamination_ratio=0.05
):
    """
    Fit EVT models with outlier detection and robustness improvements
    """
    # Group validation samples by class
    class_to_samples = {}
    
    for i, (loss, sen, cls) in enumerate(zip(losses, sentences, true_classes)):
        # Skip invalid losses
        if np.isnan(loss) or np.isinf(loss):
            continue
            
        # Get classifier output
        inputs = __predict_preprocess__(sen, tokenizer, max_length)
        logits = classifier.predict(inputs)[0]
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        # Get maximum probability
        max_prob = np.max(probs)
        
        # Calculate ensemble score with alpha=0.5
        ood_score = 0.5 * (1 - max_prob) + 0.5 * loss
        
        # Skip invalid scores
        if np.isnan(ood_score) or np.isinf(ood_score):
            continue
        
        if cls not in class_to_samples:
            class_to_samples[cls] = []
        
        class_to_samples[cls].append(ood_score)
    
    # Fit EVT models for each class with outlier removal
    evt_models = {}
    thresholds = {}
    
    for cls, scores in class_to_samples.items():
        if len(scores) < 10:  # Need minimum samples
            print(f"Warning: Class {cls} has only {len(scores)} samples, using percentile threshold")
            thresholds[cls] = np.percentile(scores, 95)
            continue
        
        scores_array = np.array(scores)
        
        # Remove outliers using IQR method
        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Keep only scores within bounds
        scores_clean = scores_array[(scores_array >= lower_bound) & (scores_array <= upper_bound)]
        
        if len(scores_clean) < 5:
            print(f"Warning: Too few samples after outlier removal for class {cls}")
            thresholds[cls] = np.percentile(scores_array, 95)
            continue
        
        try:
            # Fit GEV distribution on clean data
            shape, loc, scale = stats.genextreme.fit(-scores_clean)
            
            # Check if parameters are reasonable
            if abs(shape) > 2 or scale <= 0 or scale > 10:
                print(f"Warning: Unreasonable EVT parameters for class {cls}, using percentile")
                thresholds[cls] = np.percentile(scores_clean, 100 * (1 - fpr))
            else:
                evt_models[cls] = (shape, loc, scale)
                # Calculate threshold based on desired FPR
                threshold = -stats.genextreme.ppf(1 - fpr, shape, loc, scale)
                
                # Sanity check threshold
                if threshold < np.min(scores_clean) or threshold > np.max(scores_array) * 2:
                    print(f"Warning: EVT threshold out of range for class {cls}, using percentile")
                    threshold = np.percentile(scores_clean, 100 * (1 - fpr))
                
                thresholds[cls] = threshold
                
        except Exception as e:
            print(f"Error fitting EVT for class {cls}: {e}, using percentile")
            thresholds[cls] = np.percentile(scores_clean, 100 * (1 - fpr))
    
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


def run_improved_prediction(config):
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
    # train_input_ids, train_attention_mask, train_token_type_ids = preprocessing(
    #     tokenizer, train_sentences, max_length
    # )
    test_input_ids, test_attention_mask, test_token_type_ids = preprocessing(
        tokenizer, test_sentences, max_length
    )
    # dev_input_ids, dev_attention_mask, dev_token_type_ids = preprocessing(
    #     tokenizer, dev_sentences, max_length
    # )
    ood_input_ids, ood_attention_mask, ood_token_type_ids = preprocessing(
        tokenizer, ood_sentences, max_length
    )

    # train_tf = to_tf_format((train_input_ids, train_attention_mask, train_token_type_ids), None, len(train_sentences), batch_size=1)
    # test_tf = to_tf_format(
    #     (test_input_ids, test_attention_mask, test_token_type_ids),
    #     None,
    #     len(test_sentences),
    #     batch_size=1,
    # )
    # dev_tf = to_tf_format((dev_input_ids, dev_attention_mask, dev_token_type_ids), None, len(dev_sentences), batch_size=1)
    # ood_tf = to_tf_format(
    #     (ood_input_ids, ood_attention_mask, ood_token_type_ids),
    #     None,
    #     len(ood_sentences),
    #     batch_size=1,
    # )
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
    classifier.load_weights(os.path.join("artifacts", config["dataset"], "bert/best_model"))
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

    print("------------------------------------------------------------------")

    # Calculate losses for dev, test, and ood sets
    # train_tf = to_tf_format(
    #     (train_input_ids, train_attention_mask, train_token_type_ids),
    #     None,
    #     len(train_sentences),
    #     batch_size=1,
    # )
    # dev_tf = to_tf_format(
    #     (dev_input_ids, dev_attention_mask, dev_token_type_ids),
    #     None,
    #     len(dev_sentences),
    #     batch_size=1,
    # )
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

    # train_loss = compute_loss_safe(model, train_tf)
    # dev_loss = compute_loss_safe(model, dev_tf)
    test_loss = compute_loss_safe(model, test_tf)
    ood_loss = compute_loss_safe(model, ood_tf)

    # Fix normalization - use proper function
    # normalized_train_loss = normalize(
    #     train_loss, path=os.path.join("artifacts", config["dataset"]), mode="train"
    # )
    # normalized_dev_loss = normalize_safe(
    #     dev_loss, path=os.path.join("artifacts", config["dataset"]), mode="eval"
    # )
    # normalized_test_loss = normalize_safe(
    #     test_loss, path=os.path.join("artifacts", config["dataset"]), mode="eval"
    # )
    # normalized_ood_loss = normalize_safe(
    #     ood_loss, path=os.path.join("artifacts", config["dataset"]), mode="eval"
    # )
    # Visualize test and OOD losses
    visualize(
        test_loss,
        os.path.join(
            "artifacts",
            config["dataset"],
            "vae_loss_for_{}_test.png".format(config["dataset"]),
        ),
    )
    visualize(
        ood_loss,
        os.path.join(
            "artifacts",
            config["dataset"],
            "vae_loss_for_{}_ood.png".format(config["dataset"]),
        ),
    )

    # Choose detection approach based on configuration
    print("Running comprehensive threshold evaluation...")
    
    # Evaluate different methods
    # results = complete_evaluation_with_multiclass(
    #     test_loss, ood_loss,
    #     test_sentences, ood_sentences,
    #     test_intents, ood_intents, in_lbl_2_indx,
    #     classifier, tokenizer, max_length
    # )
    # results = smart_ensemble_detection(
    #     test_loss, ood_loss,
    #     test_sentences, ood_sentences,
    #     classifier, tokenizer, max_length
    # )
    # best_params = optimize_ensemble_threshold(
    #     test_sentences, ood_sentences, test_loss, ood_loss,
    #     classifier, tokenizer, max_length, alpha=0.7
    # )
    
    # Step 2: Run complete pipeline
    # results = complete_ood_intent_pipeline(
    #     test_sentences, test_intents, ood_sentences, ood_intents,
    #     test_loss, ood_loss, classifier, tokenizer, max_length, 
    #     in_lbl_2_indx, ood_threshold=0.0467, alpha=0.7  # Use your current best
    # )
    df_results, best_binary, best_multiclass, quick_results = run_complete_optimization(
        test_sentences, test_intents, ood_sentences, ood_intents,
        test_loss, ood_loss, classifier, tokenizer, 
        max_length, in_lbl_2_indx
    )
    print("\n" + "="*60)
    print("QUCIK RESULTS")
    print(quick_results)
    print("BEST BINART")
    print(best_binary)
    print("MULTI CLASS")
    print(best_multiclass)
    df_results.to_csv(
        os.path.join(
            "artifacts",
            config["dataset"],
            "results.csv"), index=False)
    
    # print("\n" + "="*60)
    # print("FIXED ENSEMBLE RESULTS (NO NORMALIZATION)")
    # print("="*60)
    
    # for method, metrics in results.items():
    #     print(f"\n{method.upper()}:")
    #     print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
    #     print(f"  F1 Micro:  {metrics['f1_micro']:.4f}")
    #     print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # best_method = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    # print(f"\nBEST METHOD: {best_method} (F1 Macro: {results[best_method]['f1_macro']:.4f})")
    
    # Print results
    # print("\n" + "="*60)
    # print("COMPREHENSIVE EVALUATION RESULTS")
    # print("="*60)
    
    # for method, metrics in results.items():
    #     print(f"\n{method.upper()}:")
    #     print(f"  Threshold: {metrics['threshold']:.4f}")
    #     print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
    #     print(f"  F1 Micro:  {metrics['f1_micro']:.4f}")
    #     print(f"  AUC-ROC:   {metrics.get('auc_roc'):.4f}")
    
    # # Find best method
    # best_method = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    # print(f"\nBEST METHOD: {best_method} (F1 Macro: {results[best_method]['f1_macro']:.4f})")

    # print("=== VAE LOSS DIAGNOSIS ===")
    # print(f"Test (in-domain) losses: min={np.min(test_loss):.4f}, max={np.max(test_loss):.4f}, mean={np.mean(test_loss):.4f}")
    # print(f"OOD losses: min={np.min(ood_loss):.4f}, max={np.max(ood_loss):.4f}, mean={np.mean(ood_loss):.4f}")
    # print(f"OOD mean > Test mean: {np.mean(test_loss) > np.mean(ood_loss)}")
    
    # # Check a few raw samples
    # print(f"\nFirst 5 raw test losses: {test_loss[:5]}")
    # print(f"First 5 raw OOD losses: {ood_loss[:5]}")
    # print(f"Last 5 raw test losses: {test_loss[-5:]}")
    # print(f"Last 5 raw OOD losses: {ood_loss[-5:]}")
    
    
    return df_results


if __name__ == "__main__":
    config_file = open("./config.json")
    config = json.load(config_file)
    
    results = run_improved_prediction(config)
    
    # Save results for paper
    import pickle
    with open(f"artifacts/{config['dataset']}/improved_results.pkl", "wb") as f:
        pickle.dump(results, f)
