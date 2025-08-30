import tensorflow as tf
from tensorflow.keras import models
import os
import platform

from flowers_cnn import flowers_cnn

if platform.system() == 'Windows':
    print("Running on Windows")
    saved_dir = r'C:\python_work\tensorFlow\wsl_venv\Udacity\flowers\saved_models'
else:
    saved_dir = '/mnt/c/python_work/tensorFlow/wsl_venv/Udacity/flowers/saved_models'

model_path = os.path.join(saved_dir, 'loss50_a82.keras')
prediction = flowers_cnn.flowers_predict(model_path, './predict/daisy.jpg')
print(prediction)