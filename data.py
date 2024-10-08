from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "path_to_training_data"
validation_dir = "path_to_validation_data"

# Préparation des données
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Chargement des données
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='sparse')

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='sparse')