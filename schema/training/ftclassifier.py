# local imports
from .baseclassifier import BaseClassifier, SklearnModel, TorchModel

# external imports
# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
import os
import re
from time import perf_counter

# Helper library imports
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

# Sklearn imports
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# Transformers and datasets imports
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

# Typing imports
from typing import Any, Dict, Union, Optional

# PyTorch imports
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.datasets import ImageFolder


class FTClassifier(BaseClassifier, ABC):
    """
    This class defines the methods that a Vision Transformer model should implement for performing
    classification tasks
    """

    
    def __init__(self,
                 model_name: str, 
                 output_dir: str,
                 device: str,
                 data_path: str,
                 seed: int) -> None:
        """
        Constructor for the FTClassifier class
        """

        super().__init__(model_name, 
                         output_dir, 
                         device, 
                         data_path, 
                         {}, 
                         seed)

    
class TransformersFTClassifier(FTClassifier):
    """
    This class aims to classify images using a Vision Transformer model(ViT) for feature extraction
    """

    def __init__(self,
                 model_name: str,
                 output_dir: str,
                 device: str,
                 data_path: str,
                 seed: int) -> None:
        """
        Constructor for the TransformersFTClassifier class
        """

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        super().__init__(model_name, 
                         output_dir, 
                         device, 
                         data_path, 
                         seed)
        
        self.model.label2id = self.label2idx
    
    @staticmethod
    def collate_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method defines how to collate the data
        """

        return {
            'pixel_values': torch.stack(x['pixel_values'] for x in batch),
            'labels': torch.tensor([x['labels'] for x in batch])
        }
    
    def __transform(self,
                  example_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method transforms the data with the feature extractor
        """

        inputs = self.feature_extractor([x for x in example_batch['image']],
                                        return_tensors='pt')
        
        inputs['labels'] = example_batch['labels']

        return inputs
        

    def _set_data(self, 
                  data_path: str,
                  label2idx: Dict[str, int]) -> None:
        """
        Set the data to use
        param data_path {str} the path where images are stored
        param label2idx {Dict[str, int]} the mapping of labels to indices (not used here)
        """

        data_train = ImageFolder(data_path + "/train", transform=None)
        data_val = ImageFolder(data_path + "/val", transform=None)
        data_test = ImageFolder(data_path + "/test", transform=None)

        assert data_train.class_to_idx == data_val.class_to_idx == data_test.class_to_idx

        self.label2idx = data_train.class_to_idx

        datasets = DatasetDict({
            split: Dataset.from_dict({"image": [image for image, _ in data], "labels": [label for _, label in data]})
            for split, data in [('train', data_train), ('val', data_val), ('test', data_test)]
        })

        self.data = datasets.with_transform(self.__transform)

    @staticmethod
    def stream_metrics(preds: torch.Tensor) -> Dict[str, Any]:
        """
        Compute the metrics of the model during training and evaluation
        """

        pass
    
    def _set_training_args(self,
                           **kwargs) -> None:
        """
        Set the training arguments
        param kwargs {Dict[str, Any]} the training arguments for Vision Transformer
        """

        self.model = AutoModelForImageClassification.from_pretrained(self.model_name,
                                                                     ignore_mismatched_sizes=True,
                                                                     num_labels=len(self.label2idx),
                                                                     label2id=self.label2idx,
                                                                     id2label={v: k for k, v in self.label2idx.items()})
        
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            **kwargs
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.data['train'],
            eval_dataset=self.data['val'],
            data_collator=self.collate_fn,
            compute_metrics=self.stream_metrics
        )
