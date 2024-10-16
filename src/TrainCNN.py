import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import inspect

# from src.model import model_1, model_2, model_3
# from src.data import train_dataset, test_dataset, validation_dataset

from model import model_1, model_2, model_3, model_combined
from data import train_dataset, test_dataset, validation_dataset

model = model_combined
model.summary()

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 

# model.compile(optimizer=SGD(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    train_dataset,
    epochs=30,
    # validation_data=test_dataset,
    validation_data=validation_dataset,  
    batch_size = 64,
    callbacks=[early_stopping]
)


model.save('src/model/emotion_detection_model.h5')
print("Modèle sauvegardé sous 'emotion_detection_model.h5'")

# Tracer les courbes de précision
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.savefig('monitor/accuracy_plot.png')

plt.show()

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Perte sur le test set: {test_loss}')
print(f'Précision sur le test set: {test_accuracy}')



def get_variable_name(variable):
    # Récupère le cadre d'appel actuel (frame)
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    # Recherche du nom correspondant à la valeur de la variable
    return [name for name, val in callers_local_vars if val is variable]


f = open("src/model/model_name.txt", "w")
f.write(f"{get_variable_name(model)[0]}")
f.close()

