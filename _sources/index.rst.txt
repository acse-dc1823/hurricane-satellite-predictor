Project 2: The Day After Tomorrow
======================================================

Synopsis
========

"The Day After Tomorrow" challenge is a mock exercise designed to enhance
emergency protocols under hurricane threats. It simulates a scenario where
FEMA (Federal Emergency Management Agency) in the US has opened a competition
for teams of ML specialists to forecast the evolution of tropical cyclones 
in real time. The challenge involves developing solutions to accurately predict 
hurricane trajectories and impacts using advanced machine learning techniques.


Problem Definition
==================

The problem posed in this challenge is significant and urgent, considering 
the devastating impact of hurricanes. Hurricanes can cause massive damage, 
including loss of lives and property. The project aims to leverage machine 
learning in developing predictive models that can accurately forecast the 
trajectory and intensity of tropical cyclones. This initiative seeks to 
improve emergency response and preparedness, ultimately saving lives and 
reducing economic losses.

The project specifically addresses two topics through leveraging deep learning namely,
storm sequence image prediction to gain visual representations of storm evolution
based upon prior images of storms at relative times and generation of wind speed predictions 
based upon these generated images.


User Guide
==========

A full model pipeline for image generation of the next 3 storm evolutions for a new storm, 
alongside wind prediction from these generated images is included in the repo under notebooks `here <https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-debi/tree/main/notebooks>`_.

Please utilise this notebook alongside all comments and markdown text in the notebooks as a user
reference.


Usage of AI Tools
=================

ChatGPT generated docstrings for most functions, however, these were ratified with common sense
in coding review.

We utilized ChatGPT  for parts of the image generation process as well as the image-to-wind prediction,
but again this was used primarily for "playing" with ideas getting baseline implementations to build
upon, ultimately we didn't directly use the answers, we modified a lot as this was almost always necessary
to return functioning code for the required use case.

We used a sphinx theme "read-the-docs" for documentation.


References
==========

[1] Image Generation:
            <1> `Patient Subtyping via Time-Aware LSTM Networks. <https://dl.acm.org/doi/pdf/10.1145/3097983.3097997>`_

            <2> `Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. <https://arxiv.org/pdf/1506.04214.pdf>`_

            <3> `Andrea Palazzi (ndrplz) - ConvLSTM_pytorch, MIT Licensed. <https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py>`_

[2] Wind Speed Prediction: 
            <1> `TKnutson, T., McBride, J., Chan, J. et al. Tropical cyclones and climate change. <https://www.nature.com/articles/ngeo779%22%22>`_

            <2> `Use Keras and OpenCV to predict age, gender and emotion in real-time. <https://cloud.tencent.com/developer/article/2011061>`_

[3] Miscellaneous:
            <1> `Sphinx Document theme RTD. <https://sphinx-rtd-theme.readthedocs.io/en/stable/index.html>`_


Image Generation Function API
=============================
.. automodule:: image_generate_model
  :members:
  :imported-members:


Wind Prediction Function API
============================
.. automodule:: speed_loader
  :members:
  :imported-members:

.. automodule:: wind_speed_model
  :members:
  :imported-members:
