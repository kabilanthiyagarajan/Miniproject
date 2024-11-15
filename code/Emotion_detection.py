import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from mtcnn import MTCNN
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter
# Directory paths
train_dir = '/kaggle/input/emotion-dta-set123/test'
val_dir = '/kaggle/input/emotion-dta-set123/test'

# Custom callback for logging
class EpochEndCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} completed. Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, "
              f"Validation Loss: {logs['val_loss']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}")

# Function to build the emotion model
def emotion_model():
    model = Sequential()
    
    # 1st CNN layer
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # 2nd CNN layer
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # 3rd CNN layer
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Flattening the model to prepare for fully connected layers
    model.add(Flatten())
    
    # Fully connected 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    # Fully connected 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    # Fully connected 3rd layer
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    # Output layer for 7 emotion classes
    model.add(Dense(7, activation='softmax'))
    
    return model

# Balance the dataset
def balance_dataset(train_dir):
    classes = os.listdir(train_dir)
    data = [(os.path.join(cls_path, img), cls) for cls in classes for img in os.listdir(os.path.join(train_dir, cls))]
    df = pd.DataFrame(data, columns=['image', 'label'])
    max_count = df['label'].value_counts().max()
    balanced_data = pd.concat([resample(df[df['label'] == cls], replace=True, n_samples=max_count, random_state=42) for cls in df['label'].unique()]).reset_index(drop=True)
    return balanced_data

balanced_data = balance_dataset(train_dir)

# Setup ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_dataframe(balanced_data, x_col='image', y_col='label', target_size=(48, 48), batch_size=64, color_mode="grayscale", class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(val_dir, target_size=(48, 48), batch_size=64, color_mode="grayscale", class_mode='categorical')

# Compile and train model
emotion_model_instance = emotion_model()
emotion_model_instance.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Train the model and save history
history = emotion_model_instance.fit(train_generator, steps_per_epoch=train_generator.n // train_generator.batch_size, epochs=120, validation_data=validation_generator, validation_steps=validation_generator.n // validation_generator.batch_size, callbacks=[early_stopping, EpochEndCallback()])
emotion_model_instance.save('/content/models/emotion_model.h5')
emotion_model_instance.save_weights('/content/models/emotion_model.weights.h5')

# Visualizations
# 1. Training and Validation Accuracy/Loss Curves
epochs = range(1, len(history.history['accuracy']) + 1)

# Plot Training and Validation Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Training Loss')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 2. Confusion Matrix
# Load model and set up for detection
emotion_model = load_model('/content/models/emotion_model.h5')
emotion_model.load_weights('/content/models/emotion_model.weights.h5')
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Assuming you have y_test and y_pred for confusion matrix
y_test = validation_generator.classes
y_pred = emotion_model_instance.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_dict.values(), yticklabels=emotion_dict.values())
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Detect and display emotions in a sample image
image = cv2.imread('/kaggle/input/1234-img1/img4.jpg')
if image is not None:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_detector = MTCNN()
    faces = face_detector.detect_faces(rgb_image)

    for face in faces:
        x, y, w, h = face['box']
        roi_gray = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        prob = emotion_prediction[0][maxindex]
        detected_emotion = emotion_dict[maxindex]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, f"{detected_emotion}: {prob:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    plt.imshow(cv2.resize(image, (300, 400)))
    plt.axis('off')
    plt.show()
