# QuickDraw-ML

This repository is dedicated to training machine learning models using the QuickDraw dataset. It encompasses a range of utilities from classes for model training to functions designed for dataset setup. Moreover, the repository includes numerous notebooks that demonstrate the application of these classes and techniques.


One of the best models that has been fine-tuned is a Vision Transformer and has been pushed on hugging face 🤗: [ilyesdjerfaf/vit-base-patch16-224-in21k-quickdraw](https://huggingface.co/ilyesdjerfaf/vit-base-patch16-224-in21k-quickdraw)

## Introduction

The QuickDraw-ML repository is crafted to support the development of machine learning models utilizing the QuickDraw dataset, a comprehensive collection of hand-drawn sketches gathered globally. This dataset offers a unique opportunity to train and refine models capable of recognizing and reproducing diverse sketches.

## Features

### Training Classes
The repository contains several classes designed for the training of machine learning models on the QuickDraw dataset. These classes facilitate tasks such as data loading, preprocessing, and augmentation.

### Dataset Setup Functions
A suite of functions are provided to streamline the process of preparing the QuickDraw dataset. These include downloading, splitting into train/test sets, and preprocessing.

### Demo Notebooks
A collection of notebooks within the repository serve as practical demonstrations for the implemented classes and techniques. These demos provide insights into how the training classes are applied and evaluate the efficacy of the models trained.

## Project Structure

Below is the project structure, detailing the main folders and their contents:

| Folder/File      | Description                                               |
|------------------|-----------------------------------------------------------|
| `data/`           | Contains modules for generating and managing the dataset. |
| `mlruns/`         | Includes all experiments done.           |
| `schema/`     | Contains all classes for training models in various ways. |
| `requirements.txt` | Lists all dependencies required by the project.          |

## Getting Started

To begin using QuickDraw-ML, follow these instructions:

1. **Clone the repository** to your local machine.
```bash
git clone https://github.com/mlengineershub/QuickDraw-ML
```
2. **Install required dependencies** listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```
3. **Run the dataset setup script** to download and preprocess the QuickDraw dataset. (Ask someone for the `data/size_10000/` folder)
```bash
python data/dataset_builder.py data/size_10000 data --max_images 250 --labels data/labels.txt
```
