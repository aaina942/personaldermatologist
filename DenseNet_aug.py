import os
import random
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Set your dataset directory and other parameters
dataset_directory = "data\Resized"
train_directory = "data\etrain"
validation_directory = "data\eval"
image_size = (224, 224)
batch_size = 32
epochs = 150
num_classes = 3
# Function to split dataset into train and validation directories
def split_dataset(dataset_dir, train_dir, validation_dir, validation_split=0.2):
    classes = os.listdir(dataset_dir)
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        validation_class_dir = os.path.join(validation_dir, class_name)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(validation_class_dir, exist_ok=True)
        
        images = os.listdir(class_dir)
        num_validation = int(len(images) * validation_split)
        validation_images = random.sample(images, num_validation)
        
        for image in images:
            src_path = os.path.join(class_dir, image)
            if image in validation_images:
                dst_path = os.path.join(validation_class_dir, image)
            else:
                dst_path = os.path.join(train_class_dir, image)
            shutil.copy(src_path, dst_path)

# Split dataset into training and validation directories
split_dataset(dataset_directory, train_directory, validation_directory)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Create and compile the DenseNet model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Change num_classes to the number of classes in your dataset

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Save the trained model
model.save("skin_disease_model.h5")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()