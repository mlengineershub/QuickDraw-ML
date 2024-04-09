# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime
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
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Transformers and datasets imports
from transformers import AutoFeatureExtractor, AutoModel
from datasets import Dataset, DatasetDict

# XGBoost import
from xgboost import XGBClassifier

# Typing imports
from typing import Any, Dict, Union

# PyTorch imports
import torch


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


class BaseClassifier(ABC):
    """
    This abstract class defines the methods that a classifier should implement.
    Either a feature extraction classifier, fine tuning classifier...
    """

    evaluation_experiment_name = "Benchmark"

    model_name: str
    output_dir: str
    device: str
    seed: int
    data_path: str
    training_args: Dict[str, Any]
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
                 seed: int) -> None:

        """
        The constructor of the BaseClassifier class
        param moel_name {str} the name of the model
        param output_dir {str} the output directory
        param device {str} the device to use
        param data_path {str} the path to the data
        param label2idx {Dict[str, int]} the mapping of labels to indices
        param seed {int} the seed to use
        """
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.seed = seed
        self.data_path = data_path

        self._set_seed()
        self._set_device()
        self._set_data(data_path, label2idx)
        self._set_training_args()

        self.metrics = {}
    
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
                  label2idx: Dict[str, int],
                  *args,
                  **kwargs) -> None:
        """
        Set the data to use
        param data_path {str} the path where images are stored
        param label2idx {Dict[str, int]} the mapping of labels to indices
        """

        pass

    @abstractmethod
    def _set_training_args(self,
                           *args,
                           **kwargs) -> None:
        """
        Set the training arguments
        """

        pass

    @abstractmethod
    def train(self,
              *args,
              **kwargs) -> None:
        """
        Train the model and log the results in MLflow
        """

        pass

    @abstractmethod
    def compute_metrics(self,
                        *args,
                        **kwargs) -> Dict[str, Any]:
        """
        Compute the metrics of the model during training and evaluation
        """

        pass


class FEClassifier(BaseClassifier, ABC):
    """
    This class defines the methods that a classifier using feature extraction should implement
    """

    _regex = re.compile(r"(.+?)\(")

    embedding_model_name: str
    classifier_model_name: str
    classifier_model: SklearnModel # type: ignore
    embedding_model: Any

    def __init__(self,
                 embedding_model_name: str,
                 classifier_model: SklearnModel, # type: ignore
                 output_dir: str,
                 data_path: str,
                 label2idx: Dict[str, int],
                 device: str,
                 seed: int) -> None:
        """
        The constructor of the FEClassifier class
        param embedding_model_name {str} the name of the embedding model
        param classifier_model {SklearnModel} the head classifier model
        param output_dir {str} the output directory
        param data_path {str} the path to the data
        param label2idx {Dict[str, int]} the mapping of labels to indices
        param device {str} the device to use
        param seed {int} the seed to use
        """
        
        self.experiment_name = "Feature_Extraction"
        self.classifier_model = classifier_model
        self.classifier_model_name = self._get_model_name()
        self.embedding_model_name = embedding_model_name
    
        super().__init__(model_name=f"{self.embedding_model_name}_{self.classifier_model_name}",
                         output_dir=output_dir,
                         data_path=data_path,
                         label2idx=label2idx,
                         device=device,
                         seed=seed)
            
    def _set_device(self) -> None:
        """
        Set the device to use
        """
        
        try:
            self.embedding_model.to(self.device)
        except:
            print(f"Device {self.device} not found, using cpu instead")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model.to(self.device)
    
    def _get_model_name(self) -> str:
        """
        Get the name of the classifier model
        return {str} the name of the classifier model
        """

        return self._regex.match(str(self.classifier_model)).group(1)
    
    @abstractmethod
    def embed(self,
              *args,
              **kwargs) -> None:
        """
        Embed the images of the dataset
        """

        pass


class TransformersFEClassifier(FEClassifier):
    """
    This class aims to classify images using a Vision Transformer model(ViT) for feature extraction
    """

    feature_extractor: AutoFeatureExtractor

    def __init__(self,
                 embedding_model_name: str,
                 classifier_model: SklearnModel, # type: ignore
                 output_dir: str,
                 data_path: str,
                 label2idx: Dict[str, int],
                 device: str,
                 seed: int,
                 **kwargs) -> None:
        """
        The constructor of the TransformersFEClassifier class
        param embedding_model_name {str} the name of the embedding model
        param classifier_model {SklearnModel} the head classifier model
        param output_dir {str} the output directory
        param data_path {str} the path to the data
        param label2idx {Dict[str, int]} the mapping of labels to indices
        param device {str} the device to use
        param seed {int} the seed to use
        """

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name, **kwargs)            
        super().__init__(embedding_model_name=embedding_model_name,
                         classifier_model=classifier_model,
                         output_dir=output_dir,
                         data_path=data_path,
                         label2idx=label2idx,
                         device=device,
                         seed=seed)

    def _set_data(self,
                  data_path: str,
                  label2idx: Dict[str, int]) -> None:
        """
        Set the data to use
        param data_path {str} the path where images are stored
        param label2idx {Dict[str, int]} the mapping of labels to indices
        """
    
        data = {"train": {"image": [], "labels" : []},
                    "val": {"image": [], "labels" : []},
                    "test": {"image": [], "labels" : []}}
        
        for key in data.keys():
            classes = os.listdir(f"{data_path}/{key}")
            images = []
            labels = []
            for class_name in tqdm(classes, desc=f"Loading images from {key}"):
                class_path = os.path.join(f"{data_path}/{key}", class_name)
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    images.append(Image.open(image_path))
                    labels.append(label2idx[class_name])
            data[key]["image"] = images
            data[key]["labels"] = labels
            data[key] = Dataset.from_dict(data[key])
        
        self.label2idx = label2idx
        self.idx2label = {v: k for k, v in label2idx.items()}
        self.data = DatasetDict(data)
    
    def _set_training_args(self,
                           **kwargs) -> None:
        """
        Set the training arguments
        param kwargs {Dict[str, Any]} the training arguments
        """
          
        self.training_args = kwargs
        self.classifier_model.set_params(**kwargs)
    
    def __embed_batch(self, 
                      batch) -> Dict[str, np.ndarray]:
        """
        Embed a batch of images
        param batch {List[PngImageFile]} the batch of images
        return {Dict[str, np.ndarray]} the embeddings
        """

        with torch.inference_mode():
            features = self.feature_extractor(images=batch, return_tensors="pt")
            output = self.embedding_model(**features)
            return {"embeddings": output.pooler_output.cpu().numpy().squeeze()} 

    def embed(self) -> None:
        """
        Embed the images of the dataset
        """   
        
        self.data = self.data.map(lambda x: self.__embed_batch(x["image"]),
                                  batched=False)
        
    def compute_metrics(self,
                        split: str) -> Dict[str, Any]:
        """
        Compute the metrics of the model for the given split
        param split {str} the split to compute the metrics on
        return {Dict[str, Any]} the metrics
        """

        metrics = {}

        if split not in self.data:
            raise ValueError(f"Key {split} not found in data splits.")

        if split == "val" or split == "test":
            times = []
            samples = self.data[split].select(range(2))
            for i in range(len(samples)):
                start = perf_counter()
                _ = self.__embed_batch(samples[i]["image"])
                end = perf_counter()
                times.append(end-start)
            time = np.mean(times)
            
        y_true = self.data[split]["labels"]
        start = perf_counter()
        y_pred = self.classifier_model.predict(self.data[split]["embeddings"])
        end = perf_counter()

        if split == "val" or split == "test":
            metrics.update({f"{split}_inference_time": time + (end-start)/len(y_true)})

        metrics.update({f"{split}_accuracy": accuracy_score(y_true, y_pred),
                        f"{split}_weighted_precision": precision_score(y_true, y_pred, average="weighted"),
                        f"{split}_weighted_recall": recall_score(y_true, y_pred, average="weighted"),
                        f"{split}_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
                        f"{split}_macro_precision": precision_score(y_true, y_pred, average="macro"),
                        f"{split}_macro_recall": recall_score(y_true, y_pred, average="macro"),
                        f"{split}_macro_f1": f1_score(y_true, y_pred, average="macro"),
                        f"{split}_micro_precision": precision_score(y_true, y_pred, average="micro"),
                        f"{split}_micro_recall": recall_score(y_true, y_pred, average="micro"),
                        f"{split}_micro_f1": f1_score(y_true, y_pred, average="micro"),
                        f"{split}_mcc": matthews_corrcoef(y_true, y_pred)})

        self.metrics.update(metrics)
        return metrics

    def plot_confusion_matrix(self,
                              split: str) -> plt.Figure:
        """
        Plot the confusion matrix of the model for the given split
        param split {str} the split to plot the confusion matrix on
        return {plt.Figure} the confusion matrix plot
        """    

        y_true = self.data[split]["labels"]
        y_pred = self.classifier_model.predict(self.data[split]["embeddings"])
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=False, fmt=".2f", cmap="Blues", xticklabels=self.idx2label.values(), yticklabels=self.idx2label.values())
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"{split.capitalize()} confusion matrix")
        
        return fig

    def train(self) -> None:
        """
        Train the model and log the results in MLflow
        """

        print("Training model")

        if "embeddings" not in self.data["train"].column_names:
            print('No embeddings found, computing them')
            self.embed()

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            print(f"Creating MLflow experiment: {self.experiment_name}")
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            print(f"Using MLflow experiment: {self.experiment_name}")
            experiment_id = experiment.experiment_id

        try:
            self.run_name = f"{self.model_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            with mlflow.start_run(run_name=self.run_name, experiment_id=experiment_id):
                self.classifier_model.fit(self.data["train"]["embeddings"], self.data["train"]["labels"])
                mlflow.log_params(self.training_args)
                mlflow.log_metrics(self.compute_metrics("train"))
                mlflow.log_metrics(self.compute_metrics("val"))

                fig_train = self.plot_confusion_matrix("train")
                train_confusion_matrix_path = "train_confusion_matrix.png"
                fig_train.savefig(train_confusion_matrix_path)
                mlflow.log_artifact(train_confusion_matrix_path)
                os.remove(train_confusion_matrix_path)

                fig_val = self.plot_confusion_matrix("val")
                val_confusion_matrix_path = "val_confusion_matrix.png"
                fig_val.savefig(val_confusion_matrix_path)
                mlflow.log_artifact(val_confusion_matrix_path)
                os.remove(val_confusion_matrix_path)

                mlflow.log_metrics({"train_length": len(self.data["train"]),
                                    "val_length": len(self.data["val"]),
                                    "num_labels": len(self.label2idx)})

        except Exception as e:
            self.run_name = None
            raise e

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model and log the results in MLflow
        return {Dict[str, Any]} the metrics
        """

        experiment = mlflow.get_experiment_by_name(self.evaluation_experiment_name)
        if experiment is None:
            print(f"Creating MLflow experiment: {self.evaluation_experiment_name}")
            experiment_id = mlflow.create_experiment(self.evaluation_experiment_name)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(run_name=self.run_name, experiment_id=experiment_id):
            metrics = self.compute_metrics("test")
            mlflow.log_metrics(metrics=metrics)
            fig_test = self.plot_confusion_matrix("test")
            test_confusion_matrix_path = "test_confusion_matrix.png"
            fig_test.savefig(test_confusion_matrix_path)
            mlflow.log_artifact(test_confusion_matrix_path)
            os.remove(test_confusion_matrix_path)

            mlflow.log_metrics({"test_length": len(self.data["test"]),
                                "num_labels": len(self.label2idx)})

        return metrics
