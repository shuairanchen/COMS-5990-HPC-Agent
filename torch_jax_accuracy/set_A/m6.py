import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds  # Required for loading datasets
import flax.linen as nn
from flax.training import train_state
import matplotlib.pyplot as plt
import numpy as np

def load_cifar10(batch_size=64):
    ds = tfds.load('cifar10', split='train', as_supervised=True)
    
    def preprocess(image, label):
        image = jax.image.resize(jnp.array(image), (32, 32))  # // MODIFIED: Ensure image is in the correct JAX format
        image = jnp.array(image) / 255.0  # // MODIFIED: Normalize to [0, 1]
        return image, label
    
    ds = ds.map(preprocess)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)  # Improve performance with prefetching
    
    return tfds.as_numpy(ds)  # // MODIFIED: Convert the dataset to NumPy arrays

def main():
    try:
        batch_size = 64  # Example batch size, adjust as necessary
        cifar10_data = load_cifar10(batch_size)
        
        # Example of iterating through the dataset and displaying images
        for images, labels in cifar10_data:
            # Show images - this is where you would handle displaying or further processing
            print(images.shape, labels.shape)  # // MODIFIED: Print shapes to show output
            break  # Remove break to process all batches
        
    except Exception as e:
        print("An error occurred:", e)

if __name__ == '__main__':
    main()  # Entry point to the program