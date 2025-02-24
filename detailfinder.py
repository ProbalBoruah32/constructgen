import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tkinter import Tk, filedialog
from PIL import Image

# Load Pretrained Model (EfficientNetB0 for feature extraction)
model = EfficientNetB0(weights='imagenet')

# Function to Select an Image (Replaces Google Colab Upload)
def select_image():
    Tk().withdraw()  # Hide the root tkinter window
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        print("No file selected. Exiting...")
        exit()
    return Image.open(file_path)

# Function to Preprocess Image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to model input size
    img = img.convert('RGB')  # Convert image to RGB (removes alpha channel if present)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to Analyze Image
def analyze_image(img):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    decoded_preds = tf.keras.applications.efficientnet.decode_predictions(predictions, top=3)[0]
    return decoded_preds

# Function to Handle User Queries After Analysis
def user_query(results):
    while True:
        query = input("\nAsk a question about the building (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting query mode.")
            break
        else:
            print(f"\nProcessing query: {query}...")
            print("Based on the analysis, the system suggests:")
            for i, (imagenet_id, label, score) in enumerate(results):
                print(f"{i+1}. {label} ({score * 100:.2f}%)")
            print("For more details, consider consulting an architect or engineer.")

# Main Execution
print("Select an image of a building for analysis:")
img = select_image()

# Display Selected Image
plt.imshow(img)
plt.axis("off")
plt.show()

# Analyze and Print Results
results = analyze_image(img)
print("\n🏗️ Construction Insights:")
for i, (imagenet_id, label, score) in enumerate(results):
    print(f"{i+1}. {label} ({score * 100:.2f}%)")

# Allow User to Ask for Information After Analysis
user_query(results)
