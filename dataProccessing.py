import cv2
import numpy as np
import os
import csv
from skimage.feature import graycomatrix, graycoprops


def calculate_channel_stats(image):
    """Calculate mean and standard."""
    # Split image into color channels
    channels = cv2.split(image)
    # Calculate mean and standard deviation for each channel
    means = [np.mean(channel) for channel in channels]
    std_devs = [np.std(channel) for channel in channels]
    return means, std_devs


def calculate_texture_features(image_gray):
    """Calculate texture features using Gray-Level Co-occurrence Matrix (GLCM)."""
    # Calculate GLCM
    glcm = graycomatrix(
        image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    # Calculate texture features
    contrast = graycoprops(glcm, "contrast")[0][0]
    energy = graycoprops(glcm, "energy")[0][0]
    homogeneity = graycoprops(glcm, "homogeneity")[0][0]
    correlation = graycoprops(glcm, "correlation")[0][0]
    return contrast, energy, homogeneity, correlation


def get_contour_properties(image_gray):
    """Extract properties of the largest contour found in the image."""
    # Find contours
    contours, _ = cv2.findContours(
        image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Get properties of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h != 0 else 0
    else:
        # Set default values if no contours found
        area = 0
        perimeter = 0
        aspect_ratio = 0
    return area, perimeter, aspect_ratio


def process_image(img_dir, label, writer):
    """Process each image in the specified directory and write its data to CSV."""
    # Get list of image files in the directory
    image_files = os.listdir(img_dir)
    image_paths = [os.path.join(img_dir, img_file) for img_file in image_files]

    # Process each image in the directory
    for path in image_paths:
        # Load image
        image = cv2.imread(path)
        # Convert to RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Calculate channel statistics
        means, std_devs = calculate_channel_stats(image_rgb)
        # Calculate texture features
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast, energy, homogeneity, correlation = calculate_texture_features(
            image_gray
        )
        # Get contour properties
        area, perimeter, aspect_ratio = get_contour_properties(image_gray)
        # Write image data to CSV
        os.path.basename(path)
        writer.writerow(
            [label]
            + means
            + std_devs
            + [
                contrast,
                energy,
                homogeneity,
                correlation,
                area,
                perimeter,
                aspect_ratio,
            ]
        )


def main():
    # Directories containing the images along with their corresponding labels
    img_dirs = [
        ("data/train/data5Augmentation/", "5"),
        ("data/train/data10Augmentation/", "10"),
        ("data/train/data20Augmentation/", "20"),
        ("data/train/data50Augmentation/", "50"),
    ]

    # Create a CSV file to store the data
    csv_file = "image_data.csv"

    # Open CSV file for writing
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Label",
                "Red Mean",
                "Green Mean",
                "Blue Mean",
                "Red StdDev",
                "Green StdDev",
                "Blue StdDev",
                "Contrast",
                "Energy",
                "Homogeneity",
                "Correlation",
                "Area",
                "Perimeter",
                "Aspect Ratio",
            ]
        )

        # Process each directory
        for img_dir, label in img_dirs:
            process_image(img_dir, label, writer)


if __name__ == "__main__":
    main()
