import numpy as np
from matplotlib import pyplot as plt

def plot_wind_speed_difference(predicted_speeds, actual_speeds):
    """
    Plot the difference between predicted and actual wind speeds.

    Parameters
    ----------
    predicted_speeds : list
        List of predicted wind speeds.
    actual_speeds : list
        List of actual wind speeds.

    Returns
    -------
    None
    """
    # Calculate differences and difference percentages
    differences = np.array(predicted_speeds) - np.array(actual_speeds)
    difference_percentages = np.abs(differences) / np.array(actual_speeds) * 100

    rmse = np.sqrt(np.mean(np.square(differences)))
    print('rmse:',rmse)
    # Plotting
    plt.figure(figsize=(16, 9))
    plt.subplot(2, 2, 1)
    plt.plot(actual_speeds, label='Actual Speeds', color='blue')
    plt.plot(predicted_speeds, label='Predicted Speeds', color='red')
    plt.ylabel('Wind Speed')
    plt.title('Actual vs Predicted Wind Speeds')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(difference_percentages, label='Difference Percentages', color='green')
    plt.xlabel('Sample Index')
    plt.ylabel('Difference Percentage (%)')
    plt.title('Difference Percentages Between Predicted and Actual Speeds')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.scatter(range(len(differences)), differences, label='Error')
    plt.axhline(y=0, color='red', linestyle='--', label='Ground truth')
    plt.xlabel('Sample Index')
    plt.ylabel('Error')
    plt.title('Predict Error')
    plt.legend()

    plt.tight_layout()
    plt.show()