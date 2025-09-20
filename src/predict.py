
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

model = tf.keras.models.load_model("../models/best_model.h5")
classes = ["NonWeapon", "Weapon"]  # adjust if you have more classes

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    return classes[np.argmax(preds)]

if __name__ == "__main__":
    print(predict_image("../data/test/Weapons/example.jpg"))
