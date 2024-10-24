import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import zipfile

# -------------------------------------
# 1. Download and Unzip Kaggle Dataset
# -------------------------------------

# Replace with the path to your downloaded Kaggle dataset zip file
#zip_path = 'path_to_downloaded_kaggle_dataset.zip'
#extract_to_path = 'dataset/'

#with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#  zip_ref.extractall(extract_to_path)

# After extracting, your dataset folder should be organized like this:
# dataset/
#     train/
#         Fresh/
#             img1.jpg
#         Spoiled/
#             img2.jpg
#     validation/
#         Fresh/
#             img3.jpg
#         Spoiled/
#             img4.jpg

# --------------------------
# 2. Image Processing with OpenCV
# --------------------------

def detect_spoilage_opencv(img_path):
    img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    green_pixel_count = np.sum(mask > 0)
    
    if green_pixel_count > 500:
        return True  # Spoiled
    else:
        return False  # Fresh

# --------------------------
# 3. ML Model for Spoilage Detection (CNN)
# --------------------------

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Define the CNN model with Input layer
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Check for the image data generators
if len(train_generator) > 0 and len(validation_generator) > 0:
    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=10
    )

    # Save the trained model
    model.save('spoilage_detection_model.h5')
else:
    print("Error: No images found in the training or validation data.")

# --------------------------
# 4. Combining OpenCV Detection with ML Prediction
# --------------------------

def predict_spoilage_with_ml(img_path):
    model = tf.keras.models.load_model('spoilage_detection_model.h5')
    
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    
    if prediction < 0.5:
        return False  # Fresh
    else:
        return True  # Spoiled

# --------------------------
# 5. Full Spoilage Detection System
# --------------------------

def full_spoilage_detection_system(img_path):
    if detect_spoilage_opencv(img_path):
        print("Spoilage detected by OpenCV method.")
    else:
        result = predict_spoilage_with_ml(img_path)
        if result:
            print("Prediction: The vegetable/fruit is spoiled.")
        else:
            print("Prediction: The vegetable/fruit is fresh.")

# Test the system with an image
img_path = 'img_path.2.jpg'
full_spoilage_detection_system(img_path)