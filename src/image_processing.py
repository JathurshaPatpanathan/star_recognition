import tensorflow as tf


def preprocess_image(image, label):
    image = tf.image.resize(image, [64, 64])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label


def prepare_datasets(train_data, test_data):
    train_data = train_data.map(preprocess_image).cache().shuffle(
        10000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    test_data = test_data.map(preprocess_image).cache().batch(
        32).prefetch(tf.data.experimental.AUTOTUNE)
    return train_data, test_data
