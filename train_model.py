import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Step 1: Load Data
data = []
labels = []

categories = ['with_mask', 'without_mask']

for category in categories:
    path = f'dataset/{category}'
    label = categories.index(category)

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 100))
            data.append(img)
            labels.append(label)
        except:
            pass  # skip unreadable images

# Step 2: Prepare Data
data = np.array(data) / 255.0
labels = to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Step 3: Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train Model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Step 5: Save the Trained Model
if not os.path.exists('model'):
    os.mkdir('model')

model.save('model/mask_detector_model.h5')
print("✅ Model trained and saved successfully!")
