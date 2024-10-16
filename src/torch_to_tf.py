import torch
import onnx
import torch.onnx 
from onnx_tf.backend import prepare
import tensorflow as tf

# Charger le modèle en Torch
model = torch.load('nn4.small2.v1.t7')
model.eval()
dummy_input = torch.randn(1, 3, 96, 96)

torch.onnx.export(model, dummy_input, "openface_model.onnx")

model_onnx = onnx.load("openface_model.onnx")

# Convertir vers le format TensorFlow
tf_rep = prepare(model_onnx)
tf_rep.export_graph("openface_model.pb")
# Charger le modèle TensorFlow
model = tf.saved_model.load("openface_model.pb")
# Sauvegarder au format Keras
model.save("src/model/openface_model_converted.h5")