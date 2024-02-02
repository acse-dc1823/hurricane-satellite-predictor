import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import matplotlib
import storm_predict.models.image_generate_model as Task1
# use this to avoid graphs from showing
matplotlib.use('Agg')

# using jupyter nbconvert --to script Task1-Kevin-New.ipynb


# Generate a random images for testing
folder_name = "fake_images"
os.makedirs(folder_name, exist_ok=True)
width, height = 256, 256

for i in range(10):

    random_image = np.random.randint(0, 256, (height, width, 3),
                                     dtype=np.uint8)
    random_image = Image.fromarray(random_image)

    # Save the random image in the folder
    image_path = os.path.join(folder_name, "random_image" + str(i).zfill(2)
                              + ".jpg")
    random_image.save(image_path)
    print(f"Random image saved at: {image_path}")


transform = transforms.Compose([
    transforms.Resize((366, 366)),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.2))])


dataset = Task1.ImageSequenceDataset(
    directory='fake_images',
    transform=transform,
    window_size=10,
    prediction_size=3)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
prediction_size = 1
prediction_shape = (prediction_size, 366, 366, 1)
model = Task1.ConvLSTMModel(prediction_shape)


def test_import():
    assert Task1


def test_nn_model():

    assert isinstance(model, nn.Module)
    assert len(list(model.parameters())) > 0


def test_optimizer_and_output_shape():
    prediction_size = 1
    prediction_shape = (prediction_size, 366, 366, 1)
    model = Task1.ConvLSTMModel(prediction_shape=prediction_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Capture parameter values before optimization
    params_before_optimization = [param.clone() for param
                                  in model.parameters()]

    # Perform optimization step
    input_data = torch.randn(1, 7, 1, 366, 366)
    output = model(input_data)
    criterion = Task1.WeightedMSELoss(weight_increase=1)
    loss = criterion(output, target=torch.randn(1, 3, 1, 366, 366))
    loss.backward()
    optimizer.step()

    assert output.shape == torch.Size([1, 1, 366, 366, 1])

    # Capture parameter values after optimization
    params_after_optimization = [param.clone() for param in model.parameters()]

    # Assert that parameter values have been updated
    for param_before, param_after in zip(params_before_optimization,
                                         params_after_optimization):
        assert not torch.equal(param_before, param_after)


def test_dataset():

    assert len(dataset) > 1
    assert len(dataset[0]) == 2
    assert isinstance(dataset[0], tuple)
    assert dataset[0][0].shape == torch.Size([7, 1, 366, 366])
    assert dataset[0][1].shape == torch.Size([3, 1, 366, 366])
    assert isinstance(dataset[0][0], torch.Tensor)
    assert isinstance(dataset[0][1], torch.Tensor)


def test_dataload():

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    first_batch = next(iter(dataloader))
    X_first, y_first = first_batch

    assert len(first_batch) == 2

    assert X_first.shape == torch.Size([2, 7, 1, 366, 366])
    assert y_first.shape == torch.Size([2, 3, 1, 366, 366])


def test_custom_loss_function():
    ones_1d = torch.ones(5)
    zeros_1d = torch.zeros(5)
    criterion = Task1.WeightedMSELoss(weight_increase=1.0)

    loss = criterion(ones_1d, zeros_1d)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() == 2.0

    for i in range(10):

        os.remove(os.path.join(folder_name, "random_image" + str(i).zfill(2)
                  + ".jpg"))
