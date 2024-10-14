import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2

from src.data import train_dataset, test_dataset, validation_dataset

model = Sequential([
    # Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    # Conv2D(64, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.1),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(), 
    Dropout(0.2),
    
    # Conv2D(512, (3, 3), activation='relu', padding='same'), 
    # Dropout(0.2),
    
    # Conv2D(64, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    # Dropout(0.2),

    Conv2D(256, (3, 3), activation='relu', padding='same'), 
    BatchNormalization(),
    Dropout(0.1),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'), 
    BatchNormalization(),
    Dropout(0.1),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'), 
    BatchNormalization(),
    Dropout(0.1),
    
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    # Dropout(0.5),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.summary()

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# from tensorflow.keras.optimizers import SGD
# model.compile(optimizer=SGD(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


 

history = model.fit(
    train_dataset,
    epochs=30,
    # validation_data=test_dataset,
    validation_data=validation_dataset,  
    batch_size = 64
)


model.save('model/emotion_detection_model.h5')
print("Modèle sauvegardé sous 'emotion_detection_model.h5'")

# Tracer les courbes de précision
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.savefig('accuracy_plot.png')

plt.show()

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Perte sur le test set: {test_loss}')
print(f'Précision sur le test set: {test_accuracy}')