# The Day After Tomorrow

### Link to the training dataset
https://drive.google.com/drive/folders/1tFqoQl-sdK6qTY1vNWIoAdudEsG6Lirj?usp=drive_link

### Link to the briefing slides
https://imperiallondon.sharepoint.com/:p:/r/sites/TrialTeam-EA/Shared%20Documents/General/TheDayAfterTomorrow-presentation%202.pptx?d=wdf1d9e0210264eab88858e2353a36242&csf=1&web=1&e=XoU1Am

## Predicting tropical storm behaviour through Deep Learning

**NOTE**: This project is currently a work-in-progress. If you spot any errors, please reach out.

### 1. Overview

Hurricanes can cause upwards of 1,000 deaths and $50 billion in damages in a single event and have been responsible for well over 160,000 deaths globally in recent history. During a tropical cyclone, humanitarian response efforts hinge on accurate risk approximation models that can help predict optimal emergency strategic decisions.

Therefore, we are committed to building two deep-learning models to address the following two tasks.

- Task 1: Generate an ML/DL-based solution that can generate 3 future image predictions based on the given satellite images.
- Task 2: Train a network to predict wind speeds based on the training data provided.

### 2. Documentation

The detailed project explanation and function API can be found [here](https://ese-msc-2023.github.io/acds-the-day-after-tomorrow-debi/).

This web page is automatically created by a Github action.

### 3. Environment Setup Guide

The main progress of this project can be explored using **notebook**:
- `/notebooks/xxx.ipynb`

To run these notebooks, follow these steps to download the `storm_predict` project and set up your environment:

1. clone this repository by running `git clone <url>` in your chosen directory.
2. In the cloned directory, create a new virtual environment. For example, using `conda`, run the following in the terminal:
    ```
    conda env create -f environment.yml
    conda activate acds_debi
    ```

    **NOTE**: Python version >= 3.11 is required for this project.

    Once activated, if you run the command `pip list`, you should see the following:

    ```
    Package    Version
    ---------- -------
    pip        23.3
    ...
    ```

### 4. Project Structure

See module-level docstrings for module details.


```
├── storm_predict
│   ├── models
│       ├── wind_speed_model.py
│       └── image_generate_model.py
│   ├── resources
│       ** static data **
│   ├── visualisation
│       ├── analysis.py
│       └── predict_visual.py
│   ├── tests
│       ├── __init__.py
│       ├── test_geo.py
│       └── test_tool.py
│   ├── notebooks
│       ├── read_data.ipynb
│       └── result.ipynb
│   └── tools    
│       └── tool.py
├── requirements.txt
└── environment.yml
```

### 5. Overall Performance

Generated images contain added blurry noise...

Generated windspeeds

### 6. Future Improvements

A tried implementation for the project was instead of training on specific image sequences, i.e. the methodology used for the final model, to train the image generation model using image difference and subsequent addition of the generated deltas to the final image in the sequence. Although theoretically strong in base, this was unable to outperform our final model, particularly on storms with significant storm development between images. With added development time...

### 7. Team Contacts

For any problems with the project, please feel free to contact the team leader of the Debi group.
- **[David Colomer Matachana]** - Project Coordinator - [david.colomer-matachana23@imperial.ac.uk]

Here are the contact information of the other team members:
- **[Jiangnan Meng]** - [jiangnan.meng23@imperial.ac.uk]
- **[Rory Cornelius Smith]** - [rory.cornelius-smith23@imperial.ic.uk]
- **[Iona Y Chadda]** - [iona.chadda23@imperial.ac.uk]
- **[Tianzi Zhang]** - [tianzi.zhang23@imperial.ac.uk]
- **[Alex N Njeumi]**  - [alex.njeumi23@imperial.ac.uk]
- **[Yibin Gao]** - [yibin.gao23@imperial.ac.uk]
- **[Zeyu Zhao]** - [zeyu.zhao23@imperial.ac.uk]

We welcome your input and look forward to collaborating with the wider community!
