# research_practice
Plant disease classification
==============================

Classification of plant diseases using image data and neural networks

This repository contains the code and relevant analysis used to train several 
deep convolutional neural networks (CNN) to identify 
14 crop species and 26 diseases.

The models were trained using a public dataset of 54,306 images of diseased 
and healthy plant leaves collected under controlled conditions and
made available by the PlantVillage project.

Three different approaches were evaluated to improve the baseline accuracy
reported by Mohanty et al. in the research paper, "Using Deep Learning
for Image-Based Plant Disease Detection" in which CNN models were also
used to classifiy plant diseases using the same dataset. The three
approaches investigated are Transfer Learning, Single Image
Super-Resolution and Hierarchical Superclass Learning, all of which
focus on a particular component that is unique to this dataset or image
classification problems in general.

Project Organization
------------

    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks for analyzing database and results.
    │
    └── src                <- Source code used for training models and running experiments



--------

