import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import joblib
import cv2
from dataProccessing import calculate_texture_features,calculate_channel_stats,get_contour_properties

# Load the CSV file with image data
data = pd.read_csv('image_data.csv')
 
# Create a Tkinter GUI window
window = tk.Tk()
window.title("Bill Detection App")

# Load models
svm_model = joblib.load('models/svm_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')
perceptron_model = joblib.load('models/perceptron_model.pkl')


def preprocess_image(image_path):
    """Preprocess the uploaded image."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    means, std_devs = calculate_channel_stats(image_rgb)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast, energy, homogeneity, correlation = calculate_texture_features(image_gray)
    area, perimeter, aspect_ratio = get_contour_properties(image_gray)
    features = np.concatenate([means, std_devs, [contrast, energy, homogeneity, correlation, area, perimeter, aspect_ratio]])
    return features.reshape(1, -1)


def predict_bill(filepath):
    """Predict the bill using the selected model."""
    try:
        features = preprocess_image(filepath)
        if model_choice.get() == "SVM":
            predicted_label = svm_model.predict(features)
        elif model_choice.get() == "KNN":
            predicted_label = knn_model.predict(features)
        elif model_choice.get() == "Perceptron":
            predicted_label = perceptron_model.predict(features)
        else:
            raise ValueError("Please select a model.")
        messagebox.showinfo("Prediction", f"The predicted label is: {predicted_label[0]}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def open_file():
    """Handle file selection."""
    filepath = filedialog.askopenfilename()
    if filepath:
        predict_bill(filepath)

# Create a frame for model selection
model_frame = tk.Frame(window)
model_frame.pack(pady=10)

model_label = tk.Label(model_frame, text="Select Model:")
model_label.pack(side=tk.LEFT)

model_choice = tk.StringVar()
svm_radio = tk.Radiobutton(model_frame, text="SVM", variable=model_choice, value="SVM")
svm_radio.pack(side=tk.LEFT)
knn_radio = tk.Radiobutton(model_frame, text="KNN", variable=model_choice, value="KNN")
knn_radio.pack(side=tk.LEFT)
perceptron_radio = tk.Radiobutton(model_frame, text="Perceptron", variable=model_choice, value="Perceptron")
perceptron_radio.pack(side=tk.LEFT)

# Create a button to upload image
upload_button = tk.Button(window, text="Upload Image", command=open_file)
upload_button.pack(pady=10)

# Run the GUI application
window.mainloop()
