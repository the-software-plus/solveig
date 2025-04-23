# app/services/model_service.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Global variable to hold the model
model = None
class_names = ["healthy", "bacterial_blight", "powdery_mildew"]  # Update based on your dataset
treatment_dict = {
    "healthy": "No treatment needed.",
    "bacterial_blight": "Apply copper-based fungicide and remove infected leaves.",
    "powdery_mildew": "Use sulfur-based fungicide and improve air circulation."
}

def load_model():
    global model
    try:
        model = tf.keras.models.load_model("model/plant_disease_model.h5")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def predict_disease(image):
    try:
        # Preprocess image
        image = image.resize((224, 224))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        disease = class_names[np.argmax(prediction)]
        treatment = treatment_dict.get(disease, "Consult an expert.")
        
        return {"disease": disease, "treatment": treatment}
    except Exception as e:
        print(f"Error predicting disease: {str(e)}")
        return {"disease": "error", "treatment": "Unable to predict"}