#resources.py
import os
import cv2
import logging


def load_images_from_folder(folder_path):
    """
    Loads and returns a list of images from the specified folder.

    If the folder does not exist, logs an error and returns an empty list.
    """
    if not os.path.exists(folder_path):
        logging.error(f"Folder not found: {folder_path}")
        return []

    images = []
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
        else:
            logging.warning(f"Failed to load image: {img_path}")
    if not images:
        logging.warning(f"No images loaded from {folder_path}")
    return images
