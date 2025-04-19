import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BinaryClassifier:
    def __init__(self):
        try:
            self.mlp = joblib.load("MainWebsite/mlp_model.pkl")
            self.scaler = joblib.load("MainWebsite/scaler.pkl")
            logger.debug("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def divide_and_average(self, arr, n=7):
        """Divide array into n parts and calculate average for each part"""
        part_size = len(arr) // n
        return [np.mean(arr[i * part_size : (i + 1) * part_size]) for i in range(n)]

    def tsbtc_features(self, image, n=7):
        """Extract TSBTC features from image"""
        b, g, r = cv2.split(image)
        return (self.divide_and_average(np.sort(r.ravel()), n) + 
                self.divide_and_average(np.sort(g.ravel()), n) + 
                self.divide_and_average(np.sort(b.ravel()), n))

    def glcm_features(self, image, distances=[1], angles=[3*np.pi/4]):
        """Extract GLCM features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        glcm = graycomatrix(gray, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        
        properties = ['contrast', 'dissimilarity', 'homogeneity', 
                     'energy', 'correlation', 'ASM']
        return [graycoprops(glcm, prop).flatten()[0] for prop in properties]

    def extract_features(self, image):
        """Extract all features from image"""
        tsbtc = self.tsbtc_features(image)
        glcm = self.glcm_features(image)
        return tsbtc + glcm

    def predict(self, image_path):
        """
        Predict if image shows benign or malignant lesion
        Returns: tuple (prediction, confidence)
        """
        try:
            # Read image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                # Handle byte array or similar
                file_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Could not load image")

            # Extract features
            features = self.extract_features(image)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction
            prediction = self.mlp.predict(features_scaled)[0]
            probabilities = self.mlp.predict_proba(features_scaled)[0]
            
            # Get confidence score
            confidence = probabilities[prediction]
            
            # Convert to string label
            label = "Malignant" if prediction == 1 else "Benign"
            
            logger.debug(f"Prediction: {label}, Confidence: {confidence}")
            
            return (label, confidence)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

def predict_binary(image_path):
    """
    Wrapper function to maintain compatibility with existing code
    """
    classifier = BinaryClassifier()
    return classifier.predict(image_path)