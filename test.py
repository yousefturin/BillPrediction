import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to plot image and its histogram
def plot_image_and_histogram(axs, image, title, row):
    # Plot image
    axs[row, 0].imshow(image)
    axs[row, 0].set_title(title)

    # Plot RGB histogram
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        axs[row, 1].plot(histogram, color=col)
        axs[row, 1].set_xlim([0, 256])
    axs[row, 1].set_title('RGB Histogram')

# Load the banknote images
image_paths = [
    'ApProject/data/train/5/5.JPG',
    'ApProject/data/train/10/5.JPG',
    'ApProject/data/train/20/5.JPG',
    'ApProject/data/train/50/5.JPG'
]

# Create a single figure
fig, axs = plt.subplots(len(image_paths), 2, figsize=(10, 8))

# Process each image
for i, path in enumerate(image_paths):
    # Load image
    image = cv2.imread(path)
    # Convert to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Plot image and histogram
    plot_image_and_histogram(axs, image_rgb, title=f'Image: {path}', row=i)

plt.tight_layout()
plt.show()
