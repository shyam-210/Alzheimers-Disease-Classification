from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
def convert_image(file):
    img = Image.open(file)
    img = img.convert('RGB')
    img = img.resize((180,180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array