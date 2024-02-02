## Predicting tropical storm behaviour through Deep Learning

**NOTE**: This project is currently a work-in-progress. If you spot any errors, please reach out to the contributors or raise an issue.

### 1. Overview

Hurricanes can cause upwards of 1,000 deaths and $50 billion in damages in a single event and have been responsible for well over 160,000 deaths globally in recent history. During a tropical cyclone, humanitarian response efforts hinge on accurate risk approximation models that can help predict optimal emergency strategic decisions.

Therefore, we are committed to building two deep-learning models to address the following two tasks.

- Task 1: Generate an ML/DL-based solution that can generate 3 future image predictions based on the given satellite images.
- Task 2: Train a network to predict wind speeds based on the training data provided.

### 2. Documentation

The detailed project explanation and function API can be found [here](https://ese-msc-2023.github.io/acds-the-day-after-tomorrow-debi/).

This web page is automatically updated by a Github action.

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

**N.B.** Please ensure that the dataset "Selected_Storms_curated" is located in the repo before utilising the workflow for additional training. This can be done in the local directory after cloning.

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

An attempted approach for the project involved training the image generation model using image differences and adding the generated deltas to the final image in the sequence, as opposed to the methodology used for the final model, which focused on training specific image sequences. While this alternative method had a strong theoretical base, it ultimately failed to surpass the performance of our final model, especially in cases involving storms with substantial development between images. **As shown below** for the final test case, storm translation (left to right) was inconsistent with expectations. Furthermore, graininess and slightly reduced image resolution in comparison with the final model were a factor in the decision against using this model. It is noted that when feeding the generated images from the final model as well as the deltas, there was low variation in predicted windspeeds, indicating not only a robust CNN for the wind prediction but also a low perceived pixel variation between both generation models. With additional development time...

Included in the repository are some extra experimental models namely, a ConvLSTM with time encoding included. With added development time, code reviewing, testing and packaging, the accuracy of the model would be able to be ratified. **As seen below** image sequence generation from this model performed with higher image resolution and combining this into the wind prediction CNN model may be able to produce more accurate results. It is noted, however, that this model produces less perceived rotation and considering the final stage of the surprise storm, this is an important factor.

### 7. Team Contacts

If you have any problems with the project, please feel free to contact the team leader of the Debi group.
- **[David Colomer Matachana]** - Project Coordinator - [david.colomer-matachana23@imperial.ac.uk]

Here are the contact information of the other team members:
- **[Jiangnan Meng]** - [jiangnan.meng23@imperial.ac.uk]
- **[Rory Cornelius Smith]** - [rory.cornelius-smith23@imperial.ac.uk]
- **[Iona Y Chadda]** - [iona.chadda23@imperial.ac.uk]
- **[Tianzi Zhang]** - [tianzi.zhang23@imperial.ac.uk]
- **[Alex N Njeumi]**  - [alex.njeumi23@imperial.ac.uk]
- **[Yibin Gao]** - [yibin.gao23@imperial.ac.uk]
- **[Zeyu Zhao]** - [zeyu.zhao23@imperial.ac.uk]

We welcome your input and look forward to collaborating with the wider community!
