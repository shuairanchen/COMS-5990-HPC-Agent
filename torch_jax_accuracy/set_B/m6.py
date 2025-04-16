import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# Simulate loading and transforming CIFAR-10 data
def load_data():
    # Placeholder for CIFAR-10 data: random images and labels
    key = jax.random.PRNGKey(42)
    images = jax.random.uniform(key, shape=(64, 32, 32, 3), minval=0,
                                maxval=1)  # Simulate CIFAR-10 images (64 batch size)
    labels = jax.random.randint(key, shape=(64,), minval=0, maxval=10)  # Simulate CIFAR-10 labels (10 classes)
    return images, labels


# Data Augmentation functions
def random_horizontal_flip(images):
    # Randomly flip images horizontally
    flip_mask = jax.random.randint(jax.random.PRNGKey(0), shape=(images.shape[0],), minval=0, maxval=2)  # 0 or 1 flip
    return jnp.array([jnp.fliplr(img) if flip == 1 else img for img, flip in zip(images, flip_mask)])


def random_crop(images, crop_size=32, padding=4):
    # Randomly crop images with padding (e.g., 32x32 image with padding 4)
    pad_images = jnp.pad(images, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant',
                         constant_values=0)
    crop_start = jax.random.randint(jax.random.PRNGKey(1), shape=(images.shape[0], 2), minval=0, maxval=padding)
    return jnp.array([img[start[0]:start[0] + crop_size, start[1]:start[1] + crop_size, :] for img, start in
                      zip(pad_images, crop_start)])


def normalize(images, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    # Normalize images
    return (images - np.array(mean)) / np.array(std)


# Apply transformations
def apply_transformations(images):
    images = random_horizontal_flip(images)
    images = random_crop(images)
    images = normalize(images)
    return images


# Display a batch of images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Main function to load data and apply transformations
def main():
    images, labels = load_data()  # Simulate loading CIFAR-10 data
    transformed_images = apply_transformations(images)  # Apply transformations

    # Display a batch of transformed images
    imshow(transformed_images[0])  # Display the first image in the batch


if __name__ == "__main__":
    main()
