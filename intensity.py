import cv2
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the pixel values
    return img

# Define a function to predict the disease from an input image
def predict_disease(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    # Assuming the model predicts probabilities for multiple classes, use argmax to get the predicted class
    predicted_class = np.argmax(prediction)

    # Add your own logic to map the predicted class to the corresponding disease label
    disease_labels = ['acne pustular', 'Eczema', 'Melasma']
    predicted_disease = disease_labels[predicted_class]

    return predicted_disease
def compare_images(image1, image2):
    # Load the images
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    # Convert the images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate the Structural Similarity Index (SSIM)
    similarity = ssim(img1_gray, img2_gray)

    return similarity

def detect_improvement(past_image, present_image, threshold=0.3):
    # Compare the images
    improvement = False
    similarity = compare_images(past_image, present_image)
    # Determine if there is an improvement based on the similarity threshold
    if similarity >= threshold:
        improvement = True
    else:
        improvement = False
    if similarity == 1.0:
        improvement = False 
    return improvement
model = load_model('keras_model.h5')
# Specify the paths to the past and present images
past_image_path = '29.jpg'
present_image_path = '1.jpg'

# Detect improvement
is_improvement = detect_improvement(past_image_path, present_image_path)

# Print the result
if is_improvement:
    print("Improvement detected!")
else:
    print("No improvement detected.")
# Path to the input image you want to predict
input_image_path = 'Dataset\Resized\Eczema\eczema-hand-116.jpg'

# Call the predict_disease function to get the predicted disease
predicted_disease = predict_disease(input_image_path)

# Print the predicted disease
print("Predicted disease:", predicted_disease)
if predicted_disease=='acne pustular':
    print("Remedies:")
    print("1.Topical Treatments")
    print("2.Oral Medications")
    print("3.Corticosteroid Injections")
    print("4.Skincare Routine")
    print("5.Lifestyle and Home Care")
if predicted_disease=="Eczema":
    print("Remedies:")
    print("1.Moisturize regularly")
    print("2.Avoid triggers")
    print("3.Use mild soaps and detergents")
    print("4.Take shorter, lukewarm baths or showers")
    print("5.Wear soft and breathable fabrics")
    print("6.Avoid scratching")
    print("7.Topical corticosteroids")
    print("8.Wet wrap therapy")
    print("9.Antihistamines")
    print("10.Stress management")
if predicted_disease=="Melasma":
    print("Remedies:")
    print("1.Sun Protection")
    print("2.Topical Depigmenting Agents")
    print("3.Chemical Peels")
    print("4.Microdermabrasion")
    print("5.Laser Treatments")
    print("6.Cosmetics and Camouflage")
