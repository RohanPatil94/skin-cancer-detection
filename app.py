from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import google.generativeai as genai
import time
import tensorflow as tf
import uuid
import numpy as np
from werkzeug.utils import secure_filename
time.clock = time.time

# Load the AI/ML Models
from binary_classifier import predict_binary
from database import init_db, add_user, verify_user, get_user_by_id
from auth import login_required
import secrets

secret_key = secrets.token_hex(16)
app = Flask(__name__)
app.secret_key = secret_key
init_db()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained MobileNet model (update path if needed)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MobileNetV2_final_v5.h5")
mobilenet_model = load_model(model_path)

class_names = ['Melanoma', 'Nevus', 'Benign Keratosis-like Lesions',
               'Basal Cell Carcinoma', 'Actinic Keratoses',
               'Vascular Lesions', 'Dermatofibroma']

def model_prediction_mobilenet(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

    preds = mobilenet_model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]

    return predicted_class

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionary to temporarily store images for detailed analysis
temp_images = {}

@app.route("/login", methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = verify_user(email, password)
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['name']  # Store user's name for display
            flash('Successfully logged in!')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    # If user is already logged in, redirect to home
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('register.html')
        
        if add_user(name, email, password):
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        else:
            flash('Email already exists')
    
    return render_template('register.html')

@app.route("/logout")
def logout():
    session.clear()
    flash('Successfully logged out')
    return redirect(url_for('login'))

# Update existing routes to require login
@app.route("/model")
@login_required
def mlmodel():
    return render_template("mlindex.html")

@app.route("/detailed-analysis")
@login_required
def detailed_analysis():
    image_id = request.args.get('image_id', None)
    return render_template("detailed_analysis.html", image_id=image_id)

@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html")

@app.route("/home")
@login_required
def home():
    return render_template("index.html")

@app.context_processor
def inject_user():
    if 'user_id' in session:
        return {'user_name': session.get('user_name', '')}
    return {'user_name': ''}

@app.route("/")
def index():
    if 'user_id' in session:
        return render_template("index.html")
    return redirect(url_for('login'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

def gemini(user_input):
    KEY = "AIzaSyCeGnFecFpMBqBA6Mv2M2K7GT-MD6n46vc"
    genai.configure(api_key=KEY)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(user_input)
    return response.text

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return gemini(userText).replace('*', '')

def save_image(file):
    """Save the uploaded image and return the file path."""
    if not file:
        raise ValueError("No file provided")
    
    filename = file.filename
    if not filename:
        raise ValueError("No filename provided")
    
    # Ensure filename is secure
    filename = secure_filename(filename)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({"error": "No file part in the request"}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            try:
                # Save the uploaded image
                filepath = save_image(file)
                
                # First, perform binary classification by reopening the saved file
                with open(filepath, 'rb') as f:
                    result, confidence = predict_binary(f)
                
                # Format confidence as percentage
                confidence_pct = f"{confidence * 100:.2f}%"
                
                if result == "Benign":
                    # If benign, no need to store for detailed analysis
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        
                    return jsonify({
                        "result": "Benign",
                        "confidence": confidence_pct,
                        "message": "The lesion appears to be benign.",
                        "needs_detailed": False
                    })
                else:
                    # If malignant, store the image path with a unique ID for detailed analysis
                    image_id = str(uuid.uuid4())
                    temp_images[image_id] = filepath
                    
                    # Set a timeout to clean up this image after 10 minutes
                    # In a production app, you'd want a better cleanup mechanism
                    
                    return jsonify({
                        "result": "Malignant",
                        "confidence": confidence_pct,
                        "message": "The lesion appears to be malignant. Further analysis is recommended.",
                        "needs_detailed": True,
                        "image_id": image_id
                    })
                    
            except Exception as e:
                # Clean up on error
                if 'filepath' in locals() and os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({"error": str(e)}), 500
                    
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return "Please upload an image."

@app.route('/get-stored-image', methods=['GET'])
def get_stored_image():
    image_id = request.args.get('image_id')
    if not image_id or image_id not in temp_images:
        return jsonify({"error": "Image not found or expired"}), 404
    
    # Return the path to the stored image
    return jsonify({"status": "success", "image_exists": True})

@app.route('/detailed-predict', methods=['POST'])
def detailed_predict():
    if request.method == 'POST':
        try:
            # Check if we're using a stored image or new upload
            image_id = request.form.get('image_id')
            
            if image_id and image_id in temp_images:
                # Use the stored image
                filepath = temp_images[image_id]
            else:
                # No stored image, check for new upload
                if 'image' not in request.files:
                    return jsonify({"error": "No image provided"}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({"error": "No selected file"}), 400
                
                # Save the newly uploaded image
                filepath = save_image(file)

            try:
                # Step 1: Reopen saved image file for binary prediction
                with open(filepath, "rb") as f:
                    binary_result, confidence = predict_binary(f)
                confidence_pct = f"{confidence * 100:.2f}%"

                # Step 2: If Benign, no further classification needed
                if binary_result == "Benign":
                    return jsonify({
                        "result": "Benign",
                        "confidence": confidence_pct,
                        "message": "The lesion appears to be benign.",
                        "needs_detailed": False
                    })

                # Step 3: Use MobileNet model to classify cancer type
                cancer_type = model_prediction_mobilenet(filepath)
                return jsonify({
                    "result": "Malignant",
                    "confidence": confidence_pct,
                    "type": cancer_type,
                    "message": f"The lesion is malignant. Classified as: {cancer_type}",
                    "needs_detailed": True
                })

            finally:
                # Clean up the image if it was a one-time upload
                if not image_id and 'filepath' in locals() and os.path.exists(filepath):
                    os.remove(filepath)
                # If it was a stored image, remove it from the temp dictionary
                elif image_id and image_id in temp_images:
                    # Keep the file for now in case of page refresh
                    # You could implement a more sophisticated cleanup strategy
                    pass

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return "Please upload an image."

if __name__ == "__main__":
    app.run(debug=True)