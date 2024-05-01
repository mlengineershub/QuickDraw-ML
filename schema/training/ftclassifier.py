# local imports
from .baseclassifier import BaseClassifier, TorchModel

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
from transformers import (
    AutoImageProcessor,
    # AutoFeatureExtractor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
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

    experiment_name = "Fine_Tuning"
    model: Union[TorchModel, AutoModelForImageClassification]
    
    def __init__(self,
                 model_name: str, 
                 output_dir: str,
                 device: str,
                 data_path: str,
                 seed: int,
                 **kwargs) -> None:
        """
        Constructor for the FTClassifier class

        param model_name {str} the name of the model
        param output_dir {str} the output directory
        param device {str} the device to use
        param data_path {str} the path to the data
        param seed {int} the seed to use
        """

        super().__init__(model_name, 
                         output_dir, 
                         device, 
                         data_path, 
                         {}, 
                         seed,
                         **kwargs)

    def _set_device(self) -> None:
        """
        Set the device to use
        """
        
        try:
            self.model.to(self.device)
        except:
            print(f"Device {self.device} not found, using cpu instead")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

    
class TransformersFTClassifier(FTClassifier):
    """
    This class aims to classify images using a Vision Transformer model(ViT) for feature extraction
    """

    feature_extractor: AutoImageProcessor

    def __init__(self,
                 model_name: str,
                 output_dir: str,
                 device: str,
                 data_path: str,
                 seed: int,
                 **kwargs) -> None:
        """
        Constructor for the TransformersFTClassifier class

        param model_name {str} the name of the model
        param output_dir {str} the output directory
        param device {str} the device to use
        param data_path {str} the path to the data
        param seed {int} the seed to use
        """

        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)

        super().__init__(model_name, 
                         output_dir, 
                         device, 
                         data_path, 
                         seed,
                         **kwargs)
        
    
    @staticmethod
    def collate_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method defines how to collate the data
        """

        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }
    
    def __transform(self,
                    example_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method transforms the data with the feature extractor

        param example_batch {Dict[str, Any]} the input batch of images
        return {Dict[str, Any]} the processed and converted images
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
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        datasets = DatasetDict({
            split: Dataset.from_dict({"image": [image for image, _ in data], "labels": [label for _, label in data]})
            for split, data in [('train', data_train), ('val', data_val), ('test', data_test)]
        })

        self.data = datasets.with_transform(self.__transform)

    @staticmethod
    def stream_metrics(preds: EvalPrediction) -> Dict[str, float]:
        """
        Compute the metrics of the model during training and evaluation

        param preds {EvalPrediction} the predictions of the model
        return {Dict[str, float]} the metrics of the model
        """

        labels = preds.label_ids
        preds = preds.predictions.argmax(-1)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "weighted_precision": precision_score(labels, preds, average="weighted"),
            "weighted_recall": recall_score(labels, preds, average="weighted"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
            "macro_precision": precision_score(labels, preds, average="macro"),
            "macro_recall": recall_score(labels, preds, average="macro"),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "micro_precision": precision_score(labels, preds, average="micro"),
            "micro_recall": recall_score(labels, preds, average="micro"),
            "micro_f1": f1_score(labels, preds, average="micro"),
            "mcc": matthews_corrcoef(labels, preds)
        }
        
        return metrics
    
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
                                                                     id2label=self.idx2label)
        
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
            tokenizer=self.feature_extractor,
            compute_metrics=self.stream_metrics
        )

    def compute_metrics(self,
                        split: str) -> Dict[str, float]:
        """
        Compute the metrics of the model for the given split

        param split {str} the split to compute the metrics for
        return {Dict[str, float]} the metrics of the model
        """

        metrics = self.trainer.evaluate(eval_dataset=self.data[split])

        time = metrics.pop("eval_runtime") / len(self.data[split])
        metrics = {re.sub(r"eval", split, key): value for key, value in metrics.items()}

        metrics[split + "_inference_time"] = time

        self.metrics.update(metrics)

        return metrics
    

    def train(self) -> None:
        """
        Train the model
        """

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            print(f"Creating experiment: {self.experiment_name}")
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            print(f"Using MLflow experiment: {self.experiment_name}")
            experiment_id = experiment.experiment_id

        self.run_name = f"{self.model_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

        with mlflow.start_run(run_name=self.run_name, experiment_id=experiment_id):
            self.trainer.train()
            mlflow.log_params(self.training_args.to_dict())
            mlflow.log_metrics(self.compute_metrics('train'))
            mlflow.log_metrics(self.compute_metrics('val'))
            
            mlflow.log_metrics({'train_length': len(self.data['train']),
                                'val_length': len(self.data['val']),
                                'test_length': len(self.data['test']),
                                'num_labels': len(self.label2idx)})
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model and log the results in MLflow

        return {Dict[str, float]} the metrics
        """

        experiment = mlflow.get_experiment_by_name(self.evaluation_experiment_name)
        if experiment is None:
            print(f"Creating MLflow experiment: {self.evaluation_experiment_name}")
            experiment_id = mlflow.create_experiment(self.evaluation_experiment_name)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(run_name=self.run_name, experiment_id=experiment_id):
            metrics = self.compute_metrics('test')
            mlflow.log_metrics(metrics)
            mlflow.log_metrics({'test_length': len(self.data['test']),
                                'num_labels': len(self.label2idx)})
            
        return metrics

    def plot_confusion_matrix(self,
                              split: str) -> plt.Figure:
        """
        Plot the confusion matrix of the model for the given split

        param split {str} the split to plot the confusion matrix on
        return {plt.Figure} the confusion matrix plot
        """

        preds = self.trainer.predict(self.data[split])

        labels = preds.label_ids
        preds = preds.predictions.argmax(-1)

        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=False, fmt=".2f", cmap="Blues", xticklabels=self.idx2label.values(), yticklabels=self.idx2label.values())
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{split.capitalize()} Confusion Matrix")
        plt.show()

    def save_classifier(self) -> None:
        """
        Save the classifier model
        """

        self.trainer.save_model(self.output_dir)
