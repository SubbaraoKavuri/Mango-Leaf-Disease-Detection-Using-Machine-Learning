from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import time  # Import time module for calculating prediction time

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("mango_leaf_disease_model.h5")

# Define class labels with symptoms, treatment, organic pesticides, and fertilizers information
class_labels = {
    0: {
        "name": "Anthracnose",
        "symptoms": "Dark lesions on leaves, fruit, and stems. Leaves develop necrotic lesions.",
        "treatment": "Use fungicides and remove infected plant material.",
        "organic_pesticides": "Neem oil or insecticidal soap can be used as organic pesticides.",
        "fertilizers": "Use balanced fertilizers with adequate nitrogen, phosphorus, and potassium."
    },
    1: {
        "name": "Bacterial Canker",
        "symptoms": "Water-soaked lesions on leaves, bark, and fruit. Dark brown streaks.",
        "treatment": "Use copper-based bactericides and prune affected areas.",
        "organic_pesticides": "Copper-based organic fungicides can be used.",
        "fertilizers": "Fertilize with organic compost to strengthen the plant's immune system."
    },
    2: {
        "name": "Cutting Weevil",
        "symptoms": "Tiny holes on leaves, yellowing or wilting of mango trees.",
        "treatment": "Insecticide application to control weevils.",
        "organic_pesticides": "Diatomaceous earth or neem oil can be used as natural insecticides.",
        "fertilizers": "Use balanced fertilizers to promote healthy growth and improve resistance."
    },
    3: {
        "name": "Die Back",
        "symptoms": "Dead branches, brownish leaf tips, and twigs.",
        "treatment": "Prune dead branches and apply fungicides.",
        "organic_pesticides": "Neem oil can help in controlling fungal growth.",
        "fertilizers": "Apply organic compost to improve plant strength and soil health."
    },
    4: {
        "name": "Gall Midge",
        "symptoms": "Deformation of leaves, galls forming on young tissues.",
        "treatment": "Use insecticides to control gall midges.",
        "organic_pesticides": "Garlic spray or neem oil can control gall midge infestation.",
        "fertilizers": "Use nitrogen-rich fertilizers for better growth and stress tolerance."
    },
    5: {
        "name": "Healthy",
        "symptoms": "No symptoms of disease. The plant appears healthy with no abnormal lesions.",
        "treatment": "Continue regular care for the tree. Ensure proper watering and nutrients.",
        "organic_pesticides": "No pesticides needed. Keep the plant well-cared.",
        "fertilizers": "Use organic compost or slow-release fertilizers for healthy growth."
    },
    6: {
        "name": "Powdery Mildew",
        "symptoms": "White powdery spots on leaves and stems.",
        "treatment": "Use fungicides and improve air circulation around the tree.",
        "organic_pesticides": "Sulfur or neem oil can be used to control powdery mildew organically.",
        "fertilizers": "Fertilize with phosphorus-rich fertilizers to promote plant health."
    },
    7: {
        "name": "Sooty Mould",
        "symptoms": "Black fungal growth on leaves and branches.",
        "treatment": "Control the sap-sucking insects that cause the mold.",
        "organic_pesticides": "Insecticidal soap or neem oil to target the insects.",
        "fertilizers": "Use balanced fertilizers to support plant recovery."
    }
}

# Configure a folder for uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a threshold for classification confidence
CONFIDENCE_THRESHOLD = 0.5  # Adjust this threshold as needed (e.g., 50%)

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    predicted_class = None
    prediction_time = None
    prediction_confidence = None
    symptoms = None
    treatment = None
    organic_pesticides = None
    fertilizers = None

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            error = 'No file part'
            return render_template('index.html', error=error)

        file = request.files['file']

        if file.filename == '':
            error = 'No selected file'
            return render_template('index.html', error=error)

        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the image
            try:
                start_time = time.time()  # Start timer to measure prediction time

                img = Image.open(file_path)
                img = img.resize((224, 224))  # Resize to model's input size
                img = np.array(img) / 255.0  # Normalize the image
                img = np.expand_dims(img, axis=0)  # Add batch dimension

                # Make predictions
                predictions = model.predict(img)
                predicted_class_idx = np.argmax(predictions)
                predicted_class_confidence = np.max(predictions)  # Get the highest prediction probability

                # Stop the timer and calculate prediction time
                end_time = time.time()
                prediction_time = round(end_time - start_time, 3)  # Time in seconds (rounded to 3 decimals)
                prediction_confidence = round(predicted_class_confidence * 100, 2)  # Confidence as percentage

                # If the confidence is below the threshold, return error
                if predicted_class_confidence < CONFIDENCE_THRESHOLD:
                    error = "No plant leaf detected in the image."
                    return render_template('index.html', error=error)

                # Get the predicted disease information
                predicted_class = class_labels[predicted_class_idx]["name"]
                symptoms = class_labels[predicted_class_idx]["symptoms"]
                treatment = class_labels[predicted_class_idx]["treatment"]
                organic_pesticides = class_labels[predicted_class_idx]["organic_pesticides"]
                fertilizers = class_labels[predicted_class_idx]["fertilizers"]

            except Exception as e:
                # If any error occurs, return this message
                error = "It is not a mango leaf images."
                return render_template('index.html', error=error)

            # Display the predicted class, confidence, and prediction time
            return render_template(
                'index.html',
                predicted_class=predicted_class,
                prediction_confidence=prediction_confidence,
                prediction_time=prediction_time,
                symptoms=symptoms,
                treatment=treatment,
                organic_pesticides=organic_pesticides,
                fertilizers=fertilizers
            )

    return render_template('index.html', error=error, predicted_class=predicted_class, prediction_confidence=prediction_confidence, prediction_time=prediction_time, symptoms=symptoms, treatment=treatment, organic_pesticides=organic_pesticides, fertilizers=fertilizers)


if __name__ == '__main__':
    app.run(debug=True)
