import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from model import SEBlock

# Chargement des émotions
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Charger le modèle déjà entraîné
model = load_model('src/model/emotion_detection_model.h5', custom_objects={'SEBlock': SEBlock})

# Fonction pour prétraiter l'image
def preprocess_image(image_path):
    # Lire l'image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Lire en niveaux de gris
    if img is None:
        raise ValueError(f"L'image n'a pas pu être chargée à partir de {image_path}")
    
    # Redimensionner l'image à 48x48
    # img = cv2.resize(img, (48, 48))
    img = cv2.resize(img, (64, 64))

    # Normalisation et centrage des données
    # img = img / 255.0
    img = tf.cast(img, tf.float32) / 255.0
    img = img - 1.0

    # Ajouter une dimension supplémentaire pour correspondre aux dimensions du modèle
    img = np.expand_dims(img, axis=-1)  # Ajouter un canal pour les images en niveaux de gris
    img = np.expand_dims(img, axis=0)  # Ajouter la dimension batch
    
    return img

# Fonction pour prédire l'émotion
def predict_emotion(image_path):
    # Prétraiter l'image
    img = preprocess_image(image_path)

    # Faire une prédiction
    predictions = model.predict(img)
    
    # Obtenir l'index de l'émotion avec la plus haute probabilité
    emotion_index = np.argmax(predictions)
    emotion_label = emotions[emotion_index]
    
    return emotion_label, predictions[0]

# Chemin de l'image que tu as prise
# image_path = "data/joie.jpg"
image_path = "data/colere.jpg"
# image_path = "data/tristesse.jpg"
# image_path = "data/deg2.jpeg"
# image_path = "data/fear.jpg"

# Prédire l'émotion sur l'image
emotion, confidence = predict_emotion(image_path)
print(f"Émotion prédite : {emotion} avec confiance : {confidence}")