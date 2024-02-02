import os
import json
import matplotlib.pyplot as plt
import numpy as np

def show_speed_distribution(path):
    """
    Load and display the distribution of wind speed for each hurricane.

    Parameters
    ----------
    path : str
        Root path containing folders for each hurricane with associated label JSON files.

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(5, 6, figsize=(25, 25))  # 5 rows, 6 columns for 30 images
    axs = axs.ravel()

    dic_list = os.listdir(path)
    for i in range(len(dic_list)):
        wind_speed = []
        label_list = [file for file in os.listdir(dic_list[i]) if file.endswith('_label.json')]
        for label_name in label_list:
            with open(os.path.join(dic_list[i], label_name), 'r') as f:
                label = json.load(f)
            wind_speed.append(float(label['wind_speed']))
        axs[i].hist(wind_speed, bins=20, edgecolor='black')
        axs[i].set_title(f"ID: {dic_list[i]}")
        axs[i].set_xlabel('Wind Speed')
        axs[i].set_ylabel('Frequency')

def show_one_image_per_storm(root_dir, rows=3, cols=10):
    """
    This function displays the first image from each storm in a dataset of storm images.
    The images are displayed in a grid layout specified by the rows and columns parameters.
    
    Parameters:
    - root_dir (str): The root directory containing subdirectories for each storm, 
                      where each subdirectory contains images for that storm.
    - rows (int): The number of rows in the grid of images to display.
    - cols (int): The number of columns in the grid of images to display.
    
    The function will display the images in a matplotlib figure with each image
    in its subplot. Titles for the subplots are the names of the storms.
    """
    
    storms = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    fig, axs = plt.subplots(rows, cols, figsize=(20, 6))  # Adjust the figure size as needed
    axs = axs.flatten()
    
    for ax in axs:
        ax.axis('off')
    
    for i, storm in enumerate(storms):
        if i >= rows * cols:
            break  # Stop if the maximum number of images for the grid is reached
        
        storm_dir = os.path.join(root_dir, storm)
        image_files = [file for file in sorted(os.listdir(storm_dir)) if file.endswith('.jpg')]
        
        if not image_files:
            continue
        
        img_name = os.path.join(storm_dir, image_files[0])
        image = Image.open(img_name)
        axs[i].imshow(image, cmap='gray')
        axs[i].set_title(storm)  # Set the title of the subplot to the storm's name
        
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# Function to load and display the first 100 images and their associated features and labels for a specific storm
def show_samples(dataset, storm_id):
    """
    Display the first 100 images and their associated features and labels for a specific storm.

    Parameters:
    - dataset: The dataset object containing the storm images, features, and labels.
    - storm_id: A string representing the unique identifier of the storm in the dataset.
    
    # Example usage of the function
    # storm_id should be replaced with an actual storm ID from the dataset
    storm_id = 'pjj'
    show_samples(dataset, storm_id)

    The function will filter out the files from the dataset corresponding to the specified storm_id,
    plot them in a grid, and show associated features like the storm ID, relative time from the
    start of the storm sequence, and the wind speed as the label.
    """

    storm_files = [file for file in dataset.files if file.startswith(storm_id)]
    num_images = min(100, len(storm_files))
    
    cols = 10  # Number of columns in the grid
    rows = num_images // cols + (num_images % cols > 0)  # Number of rows in the grid
    
    fig, axs = plt.subplots(rows, cols, figsize=(25, 2.5 * rows))  # Figure size may need to be adjusted
    axs = axs.ravel()  # Flatten the array of axes for easy iteration
    
    for i in range(num_images):
        idx = dataset.files.index(storm_files[i])
        image_tensor, features, label = dataset[idx]
        image = ToPILImage()(image_tensor)
        axs[i].imshow(image, cmap='gray')  # Display the image in grayscale
        axs[i].set_title(f"ID: {features['storm_id']}\nTime: {features['relative_time']}\nWind Speed: {label['wind_speed']} km/h")
        axs[i].axis('off')  # Hide the axis for a cleaner look
        
    for j in range(num_images, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()