import tensorflow as tf
import deeplake

f = open("src/model/model_name.txt", "r")
model = f.read()

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
ds = deeplake.load('hub://activeloop/fer2013-train')
ds_test = deeplake.load('hub://activeloop/fer2013-public-test')
ds_validation = deeplake.load('hub://activeloop/fer2013-public-test')

def augment(image):
    # image = tf.image.random_flip_left_right(image)  # Flip horizontal aléatoire
    image = tf.image.random_brightness(image, max_delta=0.2)  # Variations de luminosité
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Variations de contraste
    # image = tf.image.random_rotation(image, angles=0.1)  # Rotation aléatoire
    return image

from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
def convert_to_rgb(image):
    # Vérifiez si l'image a un seul canal (niveaux de gris)
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)  # Convertir de (64, 64, 1) à (64, 64, 3)
    return image

def preprocess_data(item):
    image = item['images']
    
    # Vérifier les dimensions de l'image et ajouter un canal si nécessaire
    if len(image.shape) == 2:  # Supposant des images en niveaux de gris
        image = tf.expand_dims(image, axis=-1)
    
    if model == 'model_3':
        image = convert_to_rgb(image)
    
    # S'assurer que l'image est en float32 pour la normalisation
    # image = tf.image.resize(image, [48, 48])
    image = tf.image.resize(image, [64, 64])

    image = tf.cast(image, tf.float32) / 255.0
    image = image - 1.0  # Centrer les données

    image = augment(image)

    label = item['labels']
    label = tf.cast(label, tf.int32)

    return image, label

# Création du dataset TensorFlow
# train_dataset = ds.tensorflow().map(preprocess_data).batch(32)
train_dataset = ds.tensorflow().map(preprocess_data).batch(32).shuffle(buffer_size=1000)
test_dataset = ds_test.tensorflow().map(preprocess_data).batch(32)
validation_dataset = ds_validation.tensorflow().map(preprocess_data).batch(32)