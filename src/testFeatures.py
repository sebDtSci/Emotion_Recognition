import os

model_path = 'src/nn4.small2.v1.h5'
if os.path.exists(model_path):
    print("existe")
else:
    print("n'existe pas")