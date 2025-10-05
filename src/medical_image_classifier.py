# medical_image_classifier.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Optional: enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"Found GPU(s): {gpus}")
    except Exception as e:
        print("Could not set memory growth:", e)
else:
    print("No GPU found. Running on CPU.")

# Reproducibility
SEED = 1337
tf.keras.utils.set_random_seed(SEED)

# --- Configuration ---
PROJECT_CONFIG = {
    'model_name': 'ResNet50_Medical_XRay_final.h5',
    'threshold_file': 'decision_threshold.txt',   # NEW: saved threshold after training
    'image_size': 224,                            # You can try 320 if VRAM allows
    'batch_size': 32,
    'train_dir': './medical_data/train',
    'validation_dir': './medical_data/val',
    'class_names': ['Negative', 'Positive']
}

# Training params (slightly adjusted)
PHASE_1_EPOCHS = 6        # head only
PHASE_2_EPOCHS = 12       # fine-tune
PHASE_1_LR = 3e-4         # safer than 1e-3
PHASE_2_LR = 1e-5         # very low for fine-tuning pretrained layers
DROPOUT = 0.5
LABEL_SMOOTHING = 0.02     # small smoothing helps generalization

def setup_data_generators(config):
    print("Setting up data generators with ResNet50 preprocess_input...")

    # Use preprocess_input (not rescale=1./255) to match ResNet50 expectations
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        config['train_dir'],
        target_size=(config['image_size'], config['image_size']),
        batch_size=config['batch_size'],
        class_mode='binary',
        classes=config['class_names'],  # enforce class order
        shuffle=True,
        seed=SEED
    )

    val_gen = val_datagen.flow_from_directory(
        config['validation_dir'],
        target_size=(config['image_size'], config['image_size']),
        batch_size=config['batch_size'],
        class_mode='binary',
        classes=config['class_names'],
        shuffle=False   # IMPORTANT for consistent evaluation/threshold finding
    )

    print("Class indices:", train_gen.class_indices)
    # Compute class weights to handle imbalance
    counts = np.bincount(train_gen.classes, minlength=2)
    total = counts.sum()
    neg, pos = counts[0], counts[1]
    w_neg = float(total / (2.0 * max(neg, 1)))
    w_pos = float(total / (2.0 * max(pos, 1)))
    class_weight = {0: w_neg, 1: w_pos}
    print(f"Train counts -> Negative: {neg}, Positive: {pos}")
    print(f"Class weights -> {class_weight}")

    return train_gen, val_gen, class_weight

def build_model(config):
    print("Building ResNet50 model...")
    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(config['image_size'], config['image_size'], 3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(DROPOUT)(x)
    # Ensure float32 output for numeric stability
    out = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs=base.input, outputs=out)
    return model, base

def find_best_threshold(model, val_gen):
    # Collect validation probabilities and labels
    steps = len(val_gen)
    y_prob = model.predict(val_gen, verbose=0).ravel()
    y_true = val_gen.classes[:len(y_prob)]  # aligned because shuffle=False

    best_t, best_bal_acc = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        y_hat = (y_prob >= t).astype(np.int32)
        tn = np.sum((y_true == 0) & (y_hat == 0))
        tp = np.sum((y_true == 1) & (y_hat == 1))
        fp = np.sum((y_true == 0) & (y_hat == 1))
        fn = np.sum((y_true == 1) & (y_hat == 0))
        rec_pos = tp / (tp + fn + 1e-9)
        rec_neg = tn / (tn + fp + 1e-9)
        bal_acc = 0.5 * (rec_pos + rec_neg)
        if bal_acc > best_bal_acc:
            best_bal_acc, best_t = bal_acc, t

    return best_t, best_bal_acc

def train_model():
    if not os.path.isdir(PROJECT_CONFIG['train_dir']) or not os.path.isdir(PROJECT_CONFIG['validation_dir']):
        print("CRITICAL ERROR: Training or validation directories not found.")
        print(f"Please ensure '{PROJECT_CONFIG['train_dir']}' and '{PROJECT_CONFIG['validation_dir']}' exist with 'Positive' and 'Negative' subfolders.")
        return

    train_gen, val_gen, class_weight = setup_data_generators(PROJECT_CONFIG)
    model, base = build_model(PROJECT_CONFIG)

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='prec'),
        tf.keras.metrics.Recall(name='rec'),
    ]

    # PHASE 1: train head
    print("\n" + "="*50)
    print(f"PHASE 1: Training the classification head for {PHASE_1_EPOCHS} epochs.")
    print("="*50)

    model.compile(
        optimizer=Adam(learning_rate=PHASE_1_LR),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=metrics
    )

    callbacks_phase1 = [
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_auc', mode='max', patience=4, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        train_gen,
        epochs=PHASE_1_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # PHASE 2: fine-tune backbone (keep BatchNorm layers frozen)
    print("\n" + "="*50)
    print(f"PHASE 2: Fine-tuning the model for {PHASE_2_EPOCHS} epochs.")
    print("="*50)

    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=PHASE_2_LR),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=metrics
    )

    checkpoint = ModelCheckpoint(
        PROJECT_CONFIG['model_name'],
        monitor='val_auc', mode='max',
        save_best_only=True, verbose=1
    )
    callbacks_phase2 = [
        checkpoint,
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        train_gen,
        epochs=PHASE_2_EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=callbacks_phase2,
        verbose=1
    )

    # Load best checkpoint (by val AUC)
    if os.path.exists(PROJECT_CONFIG['model_name']):
        print(f"Loading best model: {PROJECT_CONFIG['model_name']}")
        best = tf.keras.models.load_model(PROJECT_CONFIG['model_name'])
    else:
        best = model

    # Tune threshold on validation set and save it
    best_t, best_bal_acc = find_best_threshold(best, val_gen)
    with open(PROJECT_CONFIG['threshold_file'], 'w') as f:
        f.write(str(best_t))
    print(f"Saved decision threshold {best_t:.3f} to {PROJECT_CONFIG['threshold_file']} (val balanced acc={best_bal_acc:.4f})")

    # Final validation metrics print
    val_results = best.evaluate(val_gen, verbose=0, return_dict=True)
    print("Validation metrics:", val_results)

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    train_model()