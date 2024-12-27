import os
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Dataset Paths
train_dir = "./data/cifar10/train"
test_dir = "./data/cifar10/test"

# Dynamically fetch class names from the folder structure
CATEGORIES = os.listdir(train_dir)
print(f"Classes: {CATEGORIES}")

# Function to prepare data with a limit
def prepare_data(dataset_path, limit):
    data = []
    for category in CATEGORIES:
        label = CATEGORIES.index(category)  # Assign a unique integer to each class
        path = os.path.join(dataset_path, category)
        img_files = os.listdir(path)[:limit // len(CATEGORIES)]  # Limit files per category
        for img_file in img_files:
            img = cv.imread(os.path.join(path, img_file), 1)
            img_resized = cv.resize(img, (150, 150))
            data.append((img_resized, label))
    random.shuffle(data)
    X, y = zip(*data)
    return np.array(X).astype('float32') / 255.0, np.array(y)

# Prepare training and testing data
train_X, train_y = prepare_data(train_dir, limit=2000)  # Use 2000 images for training
test_X, test_y = prepare_data(test_dir, limit=1000)    # Use 1000 images for testing

# Convert labels to one-hot encoding
train_y = to_categorical(train_y, num_classes=len(CATEGORIES))
test_y = to_categorical(test_y, num_classes=len(CATEGORIES))

# Visualize some training images
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_X[i])
    plt.title(CATEGORIES[np.argmax(train_y[i])])
    plt.axis('off')
plt.show()

# Model creation
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(CATEGORIES), activation='softmax')  # Multi-class classification
])

# Model summary
model.summary()

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=5,              # Stop if no improvement after 5 epochs
    restore_best_weights=True  # Restore the weights of the best epoch
)

# Train the model with early stopping
history = model.fit(
    train_X, train_y,
    epochs=10,               # Set a high number of epochs
    batch_size=32,
    validation_split=0.2,    # Use a split of the training data for validation
    callbacks=[early_stopping]  # Include the early stopping callback
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_X, test_y)
print(f"Test Accuracy: {test_acc:.2f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Predict the classes for the test set
y_pred = model.predict(test_X)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability
y_true = np.argmax(test_y, axis=1)          # True labels from one-hot encoding

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CATEGORIES)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()
