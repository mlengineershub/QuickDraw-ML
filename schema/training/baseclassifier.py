# Standard library imports
from abc import ABC, abstractmethod

# Helper library imports
import numpy as np
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Transformers and datasets imports
from transformers import TrainingArguments
from datasets import DatasetDict

# XGBoost import
from xgboost import XGBClassifier

# Typing imports
from typing import Any, Dict, Union, Final, final

# PyTorch imports
import torch
from torchvision.models.resnet import ResNet
from torchvision.models.vgg import VGG
from torchvision.models.densenet import DenseNet
from torchvision.models.alexnet import AlexNet
from torchvision.models.squeezenet import SqueezeNet
from torchvision.models.googlenet import GoogLeNet
from torchvision.models.shufflenetv2 import ShuffleNetV2
from torchvision.models.mobilenet import MobileNetV2, MobileNetV3


SklearnModel = Union[GaussianNB,
                     MultinomialNB,
                     RandomForestClassifier,
                     GradientBoostingClassifier,
                     AdaBoostClassifier,
                     ExtraTreesClassifier,
                     XGBClassifier,
                     LogisticRegression,
                     SGDClassifier,
                     SVC,
                     KNeighborsClassifier,
                     DecisionTreeClassifier]


TorchModel = Union[ResNet,
                   VGG,
                   DenseNet,
                   AlexNet,
                   SqueezeNet,
                   GoogLeNet,
                   ShuffleNetV2,
                   MobileNetV2,
                   MobileNetV3]


class BaseClassifier(ABC):
    """
    This abstract class defines the methods that a classifier should implement.
    Either a feature extraction classifier, fine tuning classifier...
    """

    evaluation_experiment_name: Final[str] = "Benchmark"

    model_name: str
    output_dir: str
    device: str
    seed: int
    data_path: str
    training_args: Union[Dict[str, Any], TrainingArguments]
    data: DatasetDict
    label2idx: Dict[str, int]
    idx2label: Dict[int, str]
    run_name: str
    experiment_name: str
    metrics: Dict[str, Any]

    def __init__(self,
                 model_name: str,
                 output_dir: str,
                 device: str,
                 data_path: str,
                 label2idx: Dict[str, int],
                 seed: int,
                 **kwargs) -> None:

        """
        The constructor of the BaseClassifier class

        param moel_name {str} the name of the model
        param output_dir {str} the output directory
        param device {str} the device to use
        param data_path {str} the path to the data
        param label2idx {Dict[str, int]} the mapping of labels to indices
        param seed {int} the seed to use
        """

        self.data = None
        self.label2idx = None
        self.idx2label = None
        self.run_name = None
        self.training_args = None
        self.metrics = {}

        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.seed = seed
        self.data_path = data_path

        self._set_seed()
        self._set_data(data_path, label2idx)
        self._set_training_args(**kwargs)
        self._set_device()

    @final
    def _set_seed(self) -> None:
        """
        Set the seed for reproducibility
        """

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @abstractmethod
    def _set_device(self) -> None:
        """
        Set the device to use
        """

        pass

    @abstractmethod
    def _set_data(self,
                  data_path: str,
                  label2idx: Dict[str, int]) -> None:
        """
        Set the data to use

        param data_path {str} the path where images are stored
        param label2idx {Dict[str, int]} the mapping of labels to indices
        """

        pass

    @abstractmethod
    def _set_training_args(self,
                           **kwargs) -> None:
        """
        Set the training arguments
        """

        pass

    @abstractmethod
    def train(self) -> None:
        """
        Train the model and log the results in MLflow
        """

        pass

    @abstractmethod
    def compute_metrics(self,
                        split: str) -> Dict[str, float]:
        """
        Compute the metrics of the model during training and evaluation

        param split {str} the split to compute the metrics on
        return {Dict[str, float]} the metrics of the model
        """

        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model and log the results in MLflow

        return {Dict[str, float]} the metrics of the model
        """

        pass

    @abstractmethod
    def plot_confusion_matrix(self,
                              split: str) -> plt.Figure:
        """
        Plot the confusion matrix

        param split {str} the split to plot the confusion matrix on
        return {plt.Figure} the confusion matrix
        """

        pass

    @abstractmethod
    def save_classifier(self) -> None:
        """
        Save the classifier model
        """

        pass
