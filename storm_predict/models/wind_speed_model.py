import os
import json
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from livelossplot import PlotLosses
from ..visualization.predict_visualize import plot_wind_speed_difference


class ConvNet(nn.Module):
    """
    Convolutional Neural Network for wind speed prediction.

    Attributes:
        conv_layer (nn.Sequential): Convolutional layers.
        linear_layer (nn.Sequential): Linear layers for final prediction.
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten()
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(64*128*128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Forward pass of the ConvNet.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Model predictions.
        """
        x = self.conv_layer(x)
        x = self.linear_layer(x)
        return x


def train_model(model, device, train_loader, val_loader, criterion,
                optimizer, num_epochs=10, save_path='best_model.pth'):
    """
    Train the given model.

    Parameters
    ----------
    model : nn.Module
        Model to be trained.
    device : torch.device
        Device to perform training on.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : torch.nn.modules.loss._Loss
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for model training.
    num_epochs : int, optional
        Number of training epochs. Default is 10.
    save_path : str, optional
        Path to save the best model. Default is 'best_model.pth'.

    Returns
    -------
    nn.Module
        Best-trained model.
    """
    liveplot = PlotLosses()
    best_val_loss = -1
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        train_iterator = tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{num_epochs}',
            dynamic_ncols=True
            )
        train_loss = 0.0
        for inputs, labels in train_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iterator.set_postfix({'Train Loss': loss.item()},
                                       refresh=True)

        # Evaluate the model on a validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs,
                                      labels.unsqueeze(1).float()).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        liveplot.update({
            'loss': train_loss,
            'val_loss': val_loss
        })
        liveplot.draw()

        # Save the model if it has the best validation loss
        if best_val_loss == -1 or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            torch.save(best_model.state_dict(), save_path)

        print(f'Validation Loss: {val_loss}')

    return best_model


def predict(model, data_loader, device):
    """
    Predict wind speeds using the trained model.

    Parameters
    ----------
    model : nn.Module
        Trained model for wind speed prediction.
    data_loader : DataLoader
        DataLoader for input data.
    device : torch.device
        Device to perform predictions on.

    Returns
    -------
    list
        List of predicted wind speeds.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Flattened and converted to numpy array
            flattened_outputs = outputs.view(-1).cpu().numpy()
            predictions.extend(flattened_outputs)

    return predictions


def load_wind_speed(json_path):
    """
    Load wind speed from a JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing wind speed information.

    Returns
    -------
    float
        Wind speed value.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
        return float(data['wind_speed'])


def show_difference(unseen_path, model, predict_loader, device, start=-10):
    """
    Show the difference between predicted and actual wind speeds.

    Parameters
    ----------
    unseen_path : str
        Path to the folder containing unseen data.
    model : nn.Module
        Trained model for wind speed prediction.
    predict_loader : DataLoader
        DataLoader for prediction data.
    device : torch.device
        Device to perform predictions on.
    start : int, optional
        Starting index for visualization. Default is -10.
    """
    actual_speeds = []
    predicted_speeds = predict(model, predict_loader, device)

    json_path = [f for f in os.listdir(unseen_path)
                 if f.endswith('_label.json')]
    json_path.sort()

    # Loop through the folder
    for file in json_path[start:]:
        # Load actual wind speed
        actual_speed = load_wind_speed(os.path.join(unseen_path, file))
        actual_speeds.append(actual_speed)

    plot_wind_speed_difference(predicted_speeds, actual_speeds)


def predict_unknown(folder_path, model, device,
                    predict_num=13, transformer=None):
    """
    Predict wind speeds for unknown data in a folder
    and update corresponding JSON files.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing images and JSON files.
    model : nn.Module
        Trained model for wind speed prediction.
    device : torch.device
        Device to perform predictions on.
    predict_num : int, optional
        Number of predictions to make. Default is 13.
    transformer : torchvision.transforms.Transform, optional
        Image transformation. Default is None.

    Returns
    -------
    list
        List of predicted wind speeds.
    """
    # Get all the .jpg file names in the folder
    img_files = [img for img in sorted(os.listdir(folder_path))
                 if img.endswith('.jpg')]
    # Generate corresponding .json file names
    json_files = [img.replace('.jpg', '_label.json') for img in img_files]
    # Number of pictures of known wind speeds
    k = len(img_files)-predict_num
    # Predict result
    result = []

    for i in range(predict_num):
        img1 = Image.open(os.path.join(folder_path,
                                       img_files[i+k-2])).convert('L')
        img2 = Image.open(os.path.join(folder_path,
                                       img_files[i+k-1])).convert('L')
        img3 = Image.open(os.path.join(folder_path,
                                       img_files[i+k])).convert('L')

        # Image transformation
        img1 = transformer(img1)
        img2 = transformer(img2)
        img3 = transformer(img3)

        label1_path = os.path.join(folder_path, json_files[i+k-2])
        label3_path = os.path.join(folder_path, json_files[i+k])
        with open(label1_path, 'r') as f:
            label1 = json.load(f)['wind_speed']

        # Combine images into a 4-channel tensor
        eye_tensor = torch.eye(128).unsqueeze(0)
        inputs = torch.cat([float(label1)*eye_tensor, img1, img2, img3], dim=0)

        model.eval()
        inputs = inputs.to(device)
        outputs = model(inputs.unsqueeze(0))
        result.append(outputs.item())

        # Write a new json file
        new_data = {
            'wind_speed': outputs.item(),
        }
        with open(label3_path, 'w') as new_json:
            json.dump(new_data, new_json, indent=4)

    return result
