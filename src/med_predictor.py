# med_predictor.py
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input  # match training

PROJECT_CONFIG = {
    'model_name': 'ResNet50_Medical_XRay_final.h5',
    'threshold_file': 'decision_threshold.txt',  # will be created by training script
    'data_dir': './medical_data/test',           # contains Positive/Negative (case-insensitive ok)
    'image_size': 224,
    'class_names': ['Negative', 'Positive']
}

def load_threshold(default_t=0.5):
    t = default_t
    try:
        if os.path.exists(PROJECT_CONFIG['threshold_file']):
            with open(PROJECT_CONFIG['threshold_file'], 'r') as f:
                t = float(f.read().strip())
            print(f"Using tuned threshold: {t:.3f}")
        else:
            print(f"No threshold file found. Using default {default_t:.2f}")
    except Exception as e:
        print("Error reading threshold file, using default 0.5:", e)
    return t

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(PROJECT_CONFIG['image_size'], PROJECT_CONFIG['image_size']))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # IMPORTANT: same preprocess as training
    return arr

def find_dir_case_insensitive(base, name):
    # Try multiple casings, return the first that exists
    candidates = [name, name.lower(), name.upper(), name.capitalize()]
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.isdir(p):
            return p
    return None

def batch_predict(model_path, data_dir, class_names):
    # Load model
    if not os.path.exists(model_path):
        print(f"FATAL: Model file '{model_path}' not found.")
        sys.exit(1)
    model = load_model(model_path, compile=False)
    print(f"Loaded model: {model_path}")

    threshold = load_threshold(0.5)

    # Resolve directories (case-insensitive)
    pos_dir = find_dir_case_insensitive(data_dir, 'Positive')
    neg_dir = find_dir_case_insensitive(data_dir, 'Negative')
    test_dirs = {
        'Positive': pos_dir,
        'Negative': neg_dir
    }

    stats = dict(total=0, correct=0, incorrect=0,
                 false_positives=0, false_negatives=0,
                 true_positives=0, true_negatives=0)

    for actual_class_name, folder_path in test_dirs.items():
        if not folder_path:
            print(f"Warning: Test folder not found for class '{actual_class_name}' under {data_dir}")
            continue

        print(f"\nProcessing ACTUAL class: {actual_class_name} -> {folder_path}")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        actual_idx = class_names.index(actual_class_name)
        for fname in files:
            img_path = os.path.join(folder_path, fname)
            stats['total'] += 1

            arr = load_and_preprocess_image(img_path)
            prob_pos = float(model.predict(arr, verbose=0).ravel()[0])  # sigmoid output: P(Positive)
            pred_idx = 1 if prob_pos >= threshold else 0
            pred_name = class_names[pred_idx]
            conf = prob_pos if pred_idx == 1 else 1.0 - prob_pos

            correct = (pred_idx == actual_idx)
            if correct:
                stats['correct'] += 1
                if actual_class_name == 'Positive':
                    stats['true_positives'] += 1
                else:
                    stats['true_negatives'] += 1
            else:
                stats['incorrect'] += 1
                if actual_class_name == 'Negative' and pred_idx == 1:
                    stats['false_positives'] += 1
                elif actual_class_name == 'Positive' and pred_idx == 0:
                    stats['false_negatives'] += 1

            tag = "✅ CORRECT" if correct else "❌ INCORRECT"
            print(f"  [{tag}] {fname:<30} -> Pred: {pred_name} (conf={conf*100:.1f}%, prob_pos={prob_pos:.3f})")

    # Summary
    tp = stats['true_positives']
    tn = stats['true_negatives']
    fp = stats['false_positives']
    fn = stats['false_negatives']

    accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] else 0.0
    rec_pos = tp / (tp + fn + 1e-9)
    rec_neg = tn / (tn + fp + 1e-9)
    bal_acc = 0.5 * (rec_pos + rec_neg)
    precision_pos = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
    precision_neg = tn / (tn + fn + 1e-9) if (tn + fn) > 0 else 0.0

    print("\n" + "="*50)
    print("                BATCH PREDICTION SUMMARY")
    print("="*50)
    print(f"Total Images Tested:    {stats['total']}")
    print(f"Correct Predictions:    {stats['correct']}")
    print(f"Incorrect Predictions:  {stats['incorrect']}")
    print(f"Overall Accuracy:       {accuracy:.2f}%")
    print("-" * 50)
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"Recall Positive:        {rec_pos:.3f}")
    print(f"Recall Negative:        {rec_neg:.3f}")
    print(f"Balanced Accuracy:      {bal_acc:.3f}")
    print(f"Precision Positive:     {precision_pos:.3f}")
    print(f"Precision Negative:     {precision_neg:.3f}")
    print("="*50)

if __name__ == '__main__':
    batch_predict(
        PROJECT_CONFIG['model_name'],
        PROJECT_CONFIG['data_dir'],
        PROJECT_CONFIG['class_names']
    )
