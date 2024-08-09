import tensorflow_datasets as tfds


def load_dataset():
    dataset, info = tfds.load('mnist', with_info=True,
                              as_supervised=True, split='test')
    return dataset, info
