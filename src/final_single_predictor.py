import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input

# --- Configuration (Matches your successful training setup) ---
MODEL_PATH = 'ResNet50_Medical_XRay_final.h5' 
OPTIMAL_THRESHOLD = 0.700  # Based on your successful batch prediction summary
IMG_SIZE = 224

def load_and_preprocess_image(img_path: str) -> np.ndarray or None:
    """
    Loads an image and preprocesses it to be compatible with ResNet50.
    This uses the same preprocess_input function as your training script.
    """
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at '{img_path}'")
        return None
        
    try:
        # Load the image, resizing to the required input size
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array
        arr = image.img_to_array(img)
        
        # Add a batch dimension (1, 224, 224, 3)
        arr = np.expand_dims(arr, axis=0)
        
        # Apply ResNet50 specific preprocessing (scaling and color channel adjustments)
        arr = preprocess_input(arr)
        
        return arr
        
    except Exception as e:
        print(f"Error processing image '{img_path}': {e}")
        return None

def predict_xray(image_path: str, model: tf.keras.Model):
    """Makes a prediction on a single X-ray image."""
    preprocessed_img = load_and_preprocess_image(image_path)
    
    if preprocessed_img is None:
        return "Prediction Failed (Check file path or format)", 0.0

    # Get the raw prediction probability (score for the Positive class)
    # The output is a single probability because you used 'sigmoid' activation.
    raw_prediction = model.predict(preprocessed_img, verbose=0)
    confidence_score = float(raw_prediction[0][0])
    
    # Apply the optimal decision threshold (0.700)
    if confidence_score >= OPTIMAL_THRESHOLD:
        predicted_label = "POSITIVE (VIRUS DETECTED)"
    else:
        # Note: 'Negative' in your dataset combines Normal and Bacterial cases.
        predicted_label = "NEGATIVE (NORMAL or BACTERIAL)"
        
    return predicted_label, confidence_score

def main():
    """Handles command-line arguments and runs the prediction."""
    # Ensure TensorFlow only logs errors
    tf.get_logger().setLevel('ERROR') 

    print("--- Medical X-Ray Classifier Initialization ---")
    
    # 1. Load the Model
    try:
        model = load_model(MODEL_PATH, compile=False)
        print(f"Successfully loaded model: '{MODEL_PATH}'")
    except Exception as e:
        print(f"\nFATAL ERROR: Could not load model. Check path and file integrity.")
        print(f"Details: {e}")
        sys.exit(1)
        
    print(f"Using Optimal Decision Threshold: {OPTIMAL_THRESHOLD}")
    print("-------------------------------------------------")

    # 2. Check for image path argument
    if len(sys.argv) < 2:
        print("\nUsage: python final_single_predictor.py <path_to_xray_image>")
        print("\nExample Test:")
        print(f"  > python final_single_predictor.py person14_virus_44.jpeg")
        sys.exit(0)
        
    # 3. Get image path from command line
    input_image_path = sys.argv[1]

    # 4. Run Prediction
    print(f"Classifying image: '{input_image_path}'...")
    label, score = predict_xray(input_image_path, model)
    
    # 5. Output Results
    print("\n" + "="*40)
    print(f"CLASSIFICATION RESULT")
    print("="*40)
    print(f"Input Image: {input_image_path}")
    print(f"Predicted Class: {label}")
    print(f"P(Positive): {score:.4f}")
    print(f"Decision Threshold: {OPTIMAL_THRESHOLD}")
    print("="*40)

if __name__ == "__main__":
    main()
