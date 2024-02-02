import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

from livelossplot import PlotLosses
from torch.optim.lr_scheduler import StepLR


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class ImageSequenceDataset(Dataset):
    """
    A PyTorch dataset class for loading sequences of images for time series
    prediction tasks,such as forecasting future frames. This dataset provides
    sequences of images (X) and their corresponding future images (y) based
    on specified window and prediction sizes. Each item in the dataset is
    a tuple (X, y), where X is the sequence of 'window_size - prediction_size'
    input images and y is the sequence of 'prediction_size' future images that
    follow X.

    Parameters:
    -------------
    directory (str): The path to the directory containing the image files.
    transform (callable, optional): An optional transform to be applied on
                                    each image.
    window_size (int): The total number of images in each sequence,
                        including both input and prediction frames.
    prediction_size (int): The number of future images to predict at
    the end of each sequence.
    filename(str): the filename of images
    num_sequences(int): the length of sequences
    inference(bool): flag to indicate whether we are in inference phase or not.
    If yes, it will set the X sequence to the end of the dataset.

    """

    def __init__(
        self,
        directory,
        transform=None,
        window_size=10,
        prediction_size=3,
        inference=False,
    ):
        self.directory = directory
        self.transform = transform
        self.window_size = window_size
        self.prediction_size = prediction_size
        self.inference = inference

        # List of sorted filenames
        self.filenames = sorted(
            [f for f in os.listdir(directory) if f.endswith(".jpg")]
        )

        # Total number of sequences that can be generated given the window
        # and prediction sizes
        self.num_sequences = len(self.filenames) - window_size + 1

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Loads at most just the sequence into memory instead
        of the whole dataset.
        """
        sequence_images = []

        # Calculate start and end index of the sequence
        start_idx = idx
        end_idx = idx + self.window_size

        # Load and transform each image in the sequence
        for i in range(start_idx, end_idx):
            image_path = os.path.join(self.directory, self.filenames[i])
            image = Image.open(image_path).convert("L")
            if self.transform:
                image = self.transform(image)
            sequence_images.append(image)

        # Convert list of images to tensor
        sequence_tensor = torch.stack(sequence_images)

        # Split the sequence tensor into X (input) and y (target)
        if not self.inference:
            X = sequence_tensor[: self.window_size - self.prediction_size]
            y = sequence_tensor[self.window_size - self.prediction_size:]
        else:
            X = sequence_tensor[self.prediction_size:]
            y = sequence_tensor[: self.prediction_size]
            # Not actually used at all

        return X, y


class ImageSequenceDatasetDelta(Dataset):
    """
    A dataset class for loading and transforming sequences of image data
    intended for use in predicting future states of a sequence.

    The dataset generates pairs of image sequences (X)
    and target delta sequences (y).
    Each sequence X consists of 'window_size' consecutive images,
    and the corresponding
    target y is the sequence of 'prediction_size' delta images
    calculated as the pixel-wise difference from the last image
    in X to the subsequent images.

    Parameters:
    ---------------
    directory (str): Directory path containing the image files.
    transform (callable, optional): Optional transform to be applied
                                    on each image.
    window_size (int): Number of images in each sequence of X.
    prediction_size (int): Number of delta images to predict for each sequence.

    Returns:
    ---------------
    Tuple of Tensors: (X, y) where X is a tensor of input deltas with shape
    (window_size - 1 - prediction_size, height, width) and y is a tensor of
    target
    deltas with shape (prediction_size, height, width).
    """

    def __init__(self, directory, transform=None, window_size=10,
                 prediction_size=3):
        self.directory = directory
        self.transform = transform
        self.window_size = window_size
        self.prediction_size = prediction_size

        # List of sorted filenames
        self.filenames = sorted(
            [f for f in os.listdir(directory) if f.endswith(".jpg")]
        )

        '''Total number of sequences that can be generated
        given the window and prediction sizes'''
        self.num_sequences = len(self.filenames) - window_size + 1

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        sequence_images = []

        # Calculate start and end index of the sequence
        start_idx = idx
        end_idx = idx + self.window_size

        # Load and transform each image in the sequence
        for i in range(start_idx, end_idx):
            image_path = os.path.join(self.directory, self.filenames[i])
            image = Image.open(image_path).convert("L")
            if self.transform:
                image = self.transform(image)
            sequence_images.append(image)

        # Convert list of images to tensor
        sequence_tensor = torch.stack(sequence_images)

        # Compute deltas for X (input deltas)
        X_deltas = [
            sequence_tensor[i + 1] - sequence_tensor[i]
            for i in range(self.window_size - 1 - self.prediction_size)
        ]
        X = torch.stack(X_deltas)

        '''Compute deltas for y (target deltas) relative to the
        last image in X sequence'''
        last_X_image = sequence_tensor[self.window_size - 1 -
                                       self.prediction_size]
        y_deltas = [
            sequence_tensor[self.window_size -
                            self.prediction_size + i] - last_X_image
            for i in range(self.prediction_size)
        ]
        y = torch.stack(y_deltas)

        return X, y


class ConvLSTMModel(nn.Module):
    """
    A convolutional LSTM model that combines convolutional layers
    and an LSTM layer for processing sequential image data.

    The model consists of a series of convolutional layers followed by an
    LSTM layer and a fully connected layer. The convolutional layers extract
    spatial features from each frame in the sequence, while the LSTM layer
    captures temporal dependencies between the frames.
    The output of the LSTM layer is then passed through a fully connected
    layer to generate predictions of image shape for each sequence.

    Attributes:
    --------------
    conv_layers (nn.Sequential): Convolutional layers for feature extraction
                                from images.
    lstm (nn.LSTM): LSTM layer for capturing temporal relationships between
                    features extracted by convolutional layers.
    fc (nn.Linear): Fully connected layer for mapping LSTM outputs
                    to the desired prediction shape.
    prediction_shape (tuple): Shape of the output predictions.

    Args:
    --------------
    prediction_shape (tuple): The shape of the output predictions.
    This should include dimensions for batch size, sequence length,
    and any spatial dimensions of the output.

    The input to the model should be a 5D tensor with dimensions
    [batch size, sequence length, channels, height, width].
    """

    def __init__(self, prediction_shape):
        super(ConvLSTMModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(input_size=128 * 45 * 45, hidden_size=256,
                            batch_first=True)
        self.fc = nn.Linear(256, np.prod(prediction_shape))
        self.prediction_shape = prediction_shape

    def forward(self, x):
        """
        Forward pass of the ConvLSTMModel.

        Args:
        -----------
        x (Tensor): Input tensor of shape [batch size, sequence length,
        channels, height, width],representing a batch of image sequences.

        Returns:
        -----------
        Tensor: The model's predictions, reshaped to the
        specified prediction shape.
        """
        batch_size, seq_len, C, H, W = x.shape
        c_in = x.view(batch_size * seq_len, C, H, W)
        c_out = self.conv_layers(c_in)
        r_in = c_out.view(batch_size, seq_len, -1)
        r_out, _ = self.lstm(r_in)
        r_out2 = self.fc(r_out[:, -1, :])
        return r_out2.view(-1, *self.prediction_shape)


class WeightedMSELoss(nn.Module):
    """
    Custom weighted mean squared error loss that
    penalizes larger errors more heavily.

    The loss is calculated by weighting the squared difference between
    the input and the target.
    Larger differences are given more weight,
    which is controlled by the `weight_increase` parameter.

    Attributes:
    ---------------
    weight_increase (float): A scaling factor that determines
                            how much larger errors are penalized.
                            A higher value increases the penalty on
                            larger errors.

    Methods:
    ---------------
    forward(input, target): Computes the weighted mean squared error
                            between input and target.

    Example:
    ---------------
        >>> criterion = WeightedMSELoss(weight_increase=0.5)
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.randn(3)
        >>> loss = criterion(input, target)
        >>> loss.backward()
    """

    def __init__(self, weight_increase=1.0):
        """
        Initializes the WeightedMSELoss class with a specified weight
        increase for larger errors.

        Parameters:
        ------------
        weight_increase (float): A factor to control the weighting of
        larger errors. Defaults to 1.0.
        """
        super().__init__()
        self.weight_increase = (
            weight_increase
            # Determines how much more to penalize larger errors
        )

    def forward(self, input, target):
        """
        Compute the weighted mean squared error loss.

        Parameters:
        ------------
        input (Tensor): The predicted values.
        target (Tensor): The ground truth values.

        Returns:
        ------------
        Tensor: The computed weighted mean squared error loss.
        """
        # Calculate the square of the differences
        diff = input - target
        squared_diff = diff**2

        # Weight errors: larger errors are given more weight
        weights = 1 + self.weight_increase * torch.abs(diff)

        # Calculate the weighted mean squared error
        weighted_squared_diff = weights * squared_diff
        loss = torch.mean(weighted_squared_diff)
        return loss


def train_epoch(model, dataloader, criterion, optimizer,
                device, prediction_shape):
    """
    Conducts a single epoch of training on the given model.

    Processes batches from the dataloader, performs forward and
    backward passes, and updates model weights.

    Parameters:
    --------------
    model (nn.Module): The neural network model to be trained.
    dataloader (DataLoader): Provides batches of data for training.
    criterion (function): Loss function to measure model performance.
    optimizer (Optimizer): Algorithm to update model's weights.
    device (torch.device): The device to run the training on (CPU or GPU).
    prediction_shape(int): images we can produce

    Returns:
    --------------
    float: The average loss over the epoch.
    """
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.view(-1, *prediction_shape))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    epochs=10,
    device=device,
    plot_losses=False,
    prediction_shape=(3, 366, 366, 1),
):
    """
    Runs the training process for the specified number of epochs.

    Each epoch involves training the model with data provided
    by the dataloader,calculating loss,
    and updating the model parameters.

    Optionally, it can plot the training loss after each epoch.

    Parameters:
    -------------
    model (nn.Module): The neural network model to be trained.
    dataloader (DataLoader): Provides batches of data for training.
    criterion (function): Loss function to measure model performance.
    optimizer (Optimizer): Algorithm to update model's weights.
    epochs (int): Number of epochs to train the model.
    device (torch.device): The device to run the training on (CPU or GPU).
    plot_losses (bool): Flag to turn on live plotting of losses.

    Prints:
    --------------
    Epoch number and the corresponding average loss.
    """
    liveloss = PlotLosses() if plot_losses else None
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    ps = prediction_shape

    for epoch in range(epochs):
        logs = {}
        avg_loss = train_epoch(model, dataloader, criterion,
                               optimizer, device, ps)

        logs["loss"] = avg_loss

        # Update the live plot with the latest logs
        if plot_losses:
            liveloss.update(logs)
            liveloss.send()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")


def generate_images(model, input_sequence, num_images=3):
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.to(device)
        predictions = model(input_sequence)
        # Assume that the model outputs a flattened image sequence,
        # which needs to be reshaped to match the dimensions of the image.
        predictions = predictions.view(
            -1, num_images, 366, 366, 1
        )  # Adjust to the shape of the predicted image
        # reverse_normalize(predictions)
        return predictions.cpu().numpy()


def show_images(images, title="Images", figsize=(15, 5), show_colorbar=False):
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:  # If there's only one image
        axes = [axes]
    for img, ax in zip(images, axes):
        im = ax.imshow(
            img.squeeze(), cmap="gray"
        )  # Squeeze to remove singleton dimensions
        ax.axis("off")
        if show_colorbar:
            fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046,
                         pad=0.04)
            # Adjust the fraction and pad to change the size and
            # spacing of the colorbar if needed
    plt.suptitle(title)
    plt.show()


def load_images_from_folder(folder, length_sequence, image_size=(366, 366)):
    # Retrieve all jpg images in the folder
    images = glob.glob(os.path.join(folder, "{}_*.jpg".format(folder)))
    images.sort(
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )  # Sort images by number

    # Initialize an empty tensor for storing images
    data_tensor = torch.empty((1, length_sequence, 1, *image_size))

    # Load each image, resize it, convert it to tensor and store
    # it in the data tensor
    for i, img_path in enumerate(
        images[:length_sequence]
    ):  # Only take the first j images
        with Image.open(img_path) as img:
            print(img_path)
            # Resize image and convert to grayscale
            img = img.resize(image_size).convert("L")
            img_tensor = torch.unsqueeze(
                torch.tensor(np.array(img)), 0
            )  # Convert to tensor and add channel dimension
            data_tensor[:, i, :, :, :] = img_tensor

    return data_tensor
