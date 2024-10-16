import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Multiply
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Layer, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16

class SEBlock(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense_1 = Dense(self.filters // self.ratio, activation='relu')
        self.dense_2 = Dense(self.filters, activation='sigmoid')
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        se = self.global_avg_pool(inputs)
        se = self.dense_1(se)
        se = self.dense_2(se)
        return Multiply()([inputs, se])

model_1 = Sequential([
    # Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    # Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(2*64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    # Conv2D(64, (3, 3), activation='relu'),
    Conv2D(2*2*64, (3, 3), activation='relu',padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    # Conv2D(64, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(256, (3, 3), activation='relu', padding='same'), 
    BatchNormalization(),
    Dropout(0.2),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'), 
    BatchNormalization(),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'), 
    BatchNormalization(),
    Dropout(0.2),
    
    Flatten(),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    # Dropout(0.5),
    Dropout(0.2),
    
    # Sortie:
    Dense(7, activation='softmax')
])

model_2 = Sequential([
    # Conv Block 1
    Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # Conv Block 2
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Conv Block 3
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Conv Block 4 + SE Block
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    SEBlock(),  # Ajout du SE block ici
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Fully Connected Layers
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
# Pour commencer, ne pas entraîner la partie convolutionnelle
base_model.trainable = False  
model_3 = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

### Combinaison des model avec openface pur 



from keras.utils import CustomObjectScope

with CustomObjectScope({'tf': tf}):
    openface_model = load_model('nn4.small2.v1.h5')

# openface_model = load_model('openface_model_converted.h5')
# openface_model = tf.keras.models.load_model('openface_model_converted.h5')

# Geler les couches du modèle OpenFace !!
for layer in openface_model.layers:
    layer.trainable = False

# Ajouter les couches supplémentaires pour la classification des émotions
input_layer = Input(shape=(64, 64, 1))
x = openface_model(input_layer)

# Ajout des couches convolutionnelles définies dans ton modèle `model_2`
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.4)(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = SEBlock()(x)  # Ajout du SE block
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.4)(x)

# Couches Fully Connected
x = Flatten()(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.5)(x)
output_layer = Dense(7, activation='softmax')(x)

# Définir le modèle final
model_combined = Model(inputs=input_layer, outputs=output_layer)