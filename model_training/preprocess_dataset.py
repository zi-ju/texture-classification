import os
import cv2
import numpy as np
from PIL import Image


DATASET_PATH = "dataset"
CATEGORIES = ["stone", "brick", "wood"]
IMG_SIZE = (256, 256)


def preprocess_image(image):
    # Convert to RGBA if the image has transparency or is in palette mode
    if image.mode == 'P':  # If the image is in palette mode
        image = image.convert('RGBA')
    elif image.mode != 'RGBA':
        image = image.convert('RGBA')
    # Convert image to grayscale
    image = np.array(image)
    grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    # Resize and normalize
    grayscale_image = cv2.resize(grayscale_image, IMG_SIZE)  # Resize
    grayscale_image = grayscale_image / 255.0  # Normalize pixel values (0-1)
    return grayscale_image


def load_images():
    images = []
    labels = []

    for category_idx, category in enumerate(CATEGORIES):
        category_path = os.path.join(DATASET_PATH, category)
        image_files = os.listdir(category_path)[:100]

        for img_name in image_files:
            img_path = os.path.join(category_path, img_name)
            image = Image.open(img_path)
            grayscale_image = preprocess_image(image)

            images.append(grayscale_image)
            labels.append(category_idx)  # Assign numeric labels (0=stone, 1=brick, 2=wood)

    return np.array(images), np.array(labels)
