import tensorflow_datasets as tfds


def load_dataset():
    # Load the dataset with info separately
    dataset = tfds.load('mnist', as_supervised=True, split='test')
    info = tfds.builder('mnist').info
    return dataset, info
