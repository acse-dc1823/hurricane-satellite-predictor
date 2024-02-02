import matplotlib
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
import subprocess
import numpy as np


# use this to avoid graphs from showing
matplotlib.use('Agg')

# using jupyter nbconvert --to script task2_workflow.ipynb

# Define the command to convert the notebook to a script

command = "jupyter nbconvert --to script task2_workflow.ipynb"

# Execute the command using subprocess
try:
    subprocess.run(command, shell=True, check=True)
    print("Notebook converted to script successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error: Failed to convert notebook to script: {e}")

import notebook.Task2 as Task2  # change filename as appropriate

# Generate a random images for testing
folder_name = "fake_images2"
os.makedirs(folder_name, exist_ok=True)
width, height = 256, 256
size = 15
for i in range(size):

    random_image = np.random.randint(0, 256, (height, width, 3),
                                     dtype=np.uint8)
    random_image = Image.fromarray(random_image)

    # Save the random image in the folder
    image_path = os.path.join(folder_name, "random_image" + str(i).zfill(2) + ".jpg")
    random_image.save(image_path)
    json_data = {"wind_speed": "25"}
    json_file_name = "random_image" + str(i).zfill(2) + "_label.json"
    json_file_path = os.path.join(folder_name, json_file_name)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file)

    print(f"Random image saved at: {image_path}")


transformer = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_path = 'fake_images2'

dataset = Task2.StormDataset(root_dir=data_path, start_idx=0, end_idx=-10,
                             transform=transformer)
print(len(dataset))

model = Task2.ConvNet()


def test_import():
    assert Task2


def test_dataset():

    assert len(dataset) >= 0
    assert len(dataset[0]) == 2
    assert isinstance(dataset[0], tuple)
    assert dataset[0][0].shape == torch.Size([3, 128, 128])
    assert isinstance(dataset[0][0], torch.Tensor)
    assert isinstance(dataset[0][1], torch.Tensor)


def test_predict():
    device = "cpu"
    predictions = Task2.predict(model, DataLoader(dataset), device)

    assert isinstance(predictions, list)
    assert len(predictions) == 3

    for speed in predictions:
        assert isinstance(speed, np.float32)


def test_wind_speed_loader():

    unseen_path = 'Selected_Storms_curated/rml/'
    json_path = [f for f in os.listdir(unseen_path)
                 if f.endswith('_label.json')]
    json_path.sort()

    for file in json_path[-10:]:

        speed = Task2.load_wind_speed(os.path.join(unseen_path, file))

        assert isinstance(speed, float)
        assert speed > 0

    for i in range(15):

        os.remove(os.path.join(folder_name, "random_image" + str(i).zfill(2) + ".jpg"))
        os.remove(os.path.join(folder_name, "random_image" + str(i).zfill(2) + "_label.json"))

    os.rmdir(folder_name)
    # os.rmdir("task2_workflow.py")
    print("Images and folder deleted.")

