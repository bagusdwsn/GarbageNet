import tensorflow as tf
import numpy as np

def classify(image, model, class_names):
    img_width = 224
    img_height = 224
    # Preprocess
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [img_width, img_height])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
