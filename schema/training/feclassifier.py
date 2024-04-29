# local imports
from .baseclassifier import BaseClassifier, SklearnModel, TorchModel

# external imports
# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
import warnings
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
from sklearn.model_selection import cross_val_score

# Transformers and datasets imports
from transformers import AutoFeatureExtractor, AutoModel
from datasets import Dataset, DatasetDict

# Typing imports
from typing import Any, Dict, Union, Literal, List, Tuple

# PyTorch imports
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.datasets import ImageFolder

# Pickle
import pickle

# Optuna
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.trial import Trial


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
    
    def _set_training_args(self,
                           **kwargs) -> None:
        """
        Set the training arguments

        param kwargs {Dict[str, Any]} the training arguments
        """
          
        self.training_args = kwargs
        self.classifier_model.set_params(**kwargs)
    
    def _get_model_name(self) -> str:
        """
        Get the name of the classifier model

        return {str} the name of the classifier model
        """

        return self._regex.match(str(self.classifier_model)).group(1)
    
    def compute_metrics(self,
                        split: str) -> Dict[str, float]:
        """
        Compute the metrics of the model for the given split

        param split {str} the split to compute the metrics on
        return {Dict[str, float]} the metrics
        """

        metrics = {}

        if split not in self.data:
            raise ValueError(f"Key {split} not found in data splits.")

        if split == "val" or split == "test":
            if isinstance(self.data, DatasetDict):
                times = []
                samples = self._get_random_images(30)
                for i in range(len(samples)):
                    start = perf_counter()
                    _ = self._embed_batch(samples[i]["image"])
                    end = perf_counter()
                    times.append(end-start)
                time = np.mean(times)
            else:
                samples = self._get_random_images(30)
                start = perf_counter()
                _ = self._embed_batch(samples)
                end = perf_counter()
                time = (end-start)/30

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

        if isinstance(self.data, DatasetDict) and ("embeddings" not in self.data["train"].column_names):
            print('No embeddings found, computing them')
            self.embed()

        elif isinstance(self.data, dict) and ("embeddings" not in self.data["train"]):
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
                df_tmp = pd.DataFrame({"embeddings": self.data["train"]["embeddings"], 
                                       "labels": self.data["train"]["labels"]})
                df_tmp = df_tmp.sample(frac=1, random_state=self.seed)
                self.classifier_model.fit(df_tmp["embeddings"].tolist(), df_tmp["labels"].tolist())
                mlflow.log_params(self.training_args)
                mlflow.log_metrics(self.compute_metrics("train"))
                mlflow.log_metrics(self.compute_metrics("val")) 

                if isinstance(self.data, DatasetDict):
                    mlflow.log_metrics({"train_length": len(self.data["train"]),
                                        "val_length": len(self.data["val"]),
                                        "test_length": len(self.data["test"]),
                                        "num_labels": len(self.label2idx)})
                else:
                    mlflow.log_metrics({"train_length": len(self.data["train"]["labels"]),
                                        "val_length": len(self.data["val"]["labels"]),
                                        "test_length": len(self.data["test"]["labels"]),
                                        "num_labels": len(self.label2idx)})

        except Exception as e:
            self.run_name = None
            print(e)
    
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
            metrics = self.compute_metrics("test")
            mlflow.log_metrics(metrics=metrics)

            if isinstance(self.data, DatasetDict):
                mlflow.log_metrics({"test_length": len(self.data["test"]),
                                    "num_labels": len(self.label2idx)})
            else:
                mlflow.log_metrics({"test_length": len(self.data["test"]["labels"]),
                                    "num_labels": len(self.label2idx)})

        return metrics
    
    def optimize(self,
                 objective_metric: Literal["accuracy", "weighted_precision", 
                                            "weighted_recall", "weighted_f1", 
                                            "macro_precision", "macro_recall", 
                                            "macro_f1", "micro_precision", 
                                            "micro_recall", "micro_f1", 
                                            "mcc"],
                 params: Dict[str, Union[Tuple[int, int], Tuple[float, float], List[str]]],
                 cv: int,
                 n_trials: int,
                 n_jobs: int = -1) -> pd.DataFrame:
        """
        Optimize the classifier model using Optuna and cross-validation

        param objective_metric {Literal["accuracy", "weighted_precision", 
                    "weighted_recall", "weighted_f1", "macro_precision", ...]} the metric to maximize
        param params {Dict[str, Union[Tuple[int, int], Tuple[float, float], List[str]]} the parameters to optimize
        param cv {int} the number of folds for cross-validation
        param n_trials {int} the number of trials to run
        param n_jobs {int} the number of jobs to run in parallel
        return {pd.DataFrame} the results of the optimization

        Example of params:
        for a RandomForestClassifier model:
            params = {
                "n_estimators": (100, 1000),
                "max_depth": (1, 10),
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 10),
                "max_features": ["auto", "sqrt", "log2"]
            }
        for a SVC model:
            params = {
                "C": (0.1, 1.0),
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "degree": (1, 5),
                "gamma": ["scale", "auto"],
                "coef0": (0.0, 1.0)
            }
        """

        print(f"Optimizing {len(params)} of a {self.classifier_model_name} model using {objective_metric} as the objective metric")

        if isinstance(self.data, DatasetDict):
            if "embeddings" not in self.data["train"].column_names:
                print('No embeddings found, computing them')
                self.embed()
        elif isinstance(self.data, dict):
            if "embeddings" not in self.data["train"]:
                print('No embeddings found, computing them')
                self.embed()

        model = self.classifier_model.__class__(**self.classifier_model.get_params())

        df_tmp = pd.DataFrame({"embeddings": np.concatenate([self.data["train"]["embeddings"], self.data["val"]["embeddings"]], axis=0),
                               "labels": np.concatenate([self.data["train"]["labels"], self.data["val"]["labels"]], axis=0)})
        
        df_tmp = df_tmp.sample(frac=1, random_state=self.seed)

        def objective(trial: Trial) -> float:
            """
            Objective function for the Optuna optimization

            param trial {Trial} the trial to run
            return {float} the mean of the cross-validation scores
            """

            trial_params = {}
            for key, value in params.items():
                try:
                    if isinstance(value, tuple) and len(value) == 2:
                        if all(isinstance(v, float) for v in value):
                            trial_params[key] = trial.suggest_float(key, value[0], value[1], log=False)
                        elif all(isinstance(v, int) for v in value):
                            trial_params[key] = trial.suggest_int(key, value[0], value[1])
                    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                        trial_params[key] = trial.suggest_categorical(key, value)
                    else:
                        raise ValueError(f"Unsupported type or structure for parameter {key}: {value}")
                except ValueError as e:
                    warnings.warn(str(e))
                    continue
            
            if len(trial_params) == 0:
                raise ValueError("No valid parameters were provided to configure the model.")

            model.set_params(**trial_params)
            scores = cross_val_score(model, df_tmp["embeddings"].tolist(), df_tmp["labels"].tolist(), cv=cv, n_jobs=n_jobs, scoring=objective_metric)
            
            return scores.mean()

        study = create_study(direction="maximize", sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials)

        study_results = study.trials_dataframe()

        return study_results

    @abstractmethod
    def embed(self,
              *args,
              **kwargs) -> None:
        """
        Embed the images of the dataset
        """

        pass

    @abstractmethod
    def _embed_batch(self,
                     batch) -> Dict[str, np.ndarray]:
            """
            Embed a batch of images

            param batch {List[PngImageFile]} the batch of images
            return {Dict[str, np.ndarray]} the embeddings
            """
    
            pass
    
    @abstractmethod
    def _get_random_images(self,
                           n: int) -> Union[DataLoader, Dataset]:
        """
        Get the first n images of a dataset

        param n {int} the number of images to get
        return {Union[DataLoader, Dataset]} the first n images from the image folder
        """

        pass

    def save_classifier(self) -> None:
        """
        Save the classifier model
        """

        with open(f"{self.output_dir}/{self.model_name}.pkl", "wb") as f:
            pickle.dump(self.classifier_model, f)


class TorchFEClassifier(FEClassifier):
    """
    This class aims to classify images using a Torch model for feature extraction
    """

    tranform = Compose([
        Resize((224, 224)), 
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self,
                 embedding_model: TorchModel,
                 classifier_model: SklearnModel, # type: ignore
                 output_dir: str,
                 data_path: str,
                 device: str,
                 seed: int,
                 **kwargs) -> None:
        """
        The constructor of the TorchFEClassifier class

        param embedding_model_name {str} the name of the embedding model
        param classifier_model {SklearnModel} the head classifier model
        param output_dir {str} the output directory
        param data_path {str} the path to the data
        param label2idx {Dict[str, int]} the mapping of labels to indices
        param device {str} the device to use
        param seed {int} the seed to use
        """

        self.embedding_model = embedding_model
        super().__init__(embedding_model_name=embedding_model.__class__.__name__,
                         classifier_model=classifier_model,
                         output_dir=output_dir,
                         data_path=data_path,
                         label2idx={},
                         device=device,
                         seed=seed)
        
    def _set_data(self,
                  data_path: str,
                  label2idx: Dict[str, int]) -> None:
        """
        Set the data to use

        param data_path {str} the path where images are stored
        param label2idx {Dict[str, int]} the mapping of labels to indices (here {})
        """
        
        data_train = ImageFolder(root=f"{data_path}/train", transform=self.tranform)
        data_val = ImageFolder(root=f"{data_path}/val", transform=self.tranform)
        data_test = ImageFolder(root=f"{data_path}/test", transform=self.tranform)

        assert data_train.class_to_idx == data_val.class_to_idx == data_test.class_to_idx

        self.label2idx = data_train.class_to_idx
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        dataloader_train = DataLoader(data_train, batch_size=32, shuffle=False)
        dataloader_val = DataLoader(data_val, batch_size=32, shuffle=False)
        dataloader_test = DataLoader(data_test, batch_size=32, shuffle=False)

        self.data = {"train": dataloader_train,
                     "val": dataloader_val,
                     "test": dataloader_test }
    
    
    def embed(self) -> None:
        """
        Embed the images of the dataset
        """

        self.data = {split: self._embed_batch(batch) for split, batch in self.data.items()}

    def _embed_batch(self,
                     batch: DataLoader) -> Dict[str, Union[np.ndarray, np.ndarray]]:
            """
            Embed a batch of images

            param data {DataLoader} the data to embed
            return {Dict[str, np.ndarray]} the embeddings
            """
    
            with torch.inference_mode():
                embeddings = []
                all_labels = []
                for images, labels in tqdm(batch, desc="Embedding images"):
                    embeddings.append(self.embedding_model(images.to(self.device)).cpu().detach().numpy())
                    all_labels.extend(labels.cpu().detach().numpy())

                return {"embeddings": np.concatenate(embeddings),
                        "labels": np.array(all_labels)}
    
    def _get_random_images(self, 
                           n: int) -> DataLoader:
            """
            Get the first n images of a DataLoader

            param n {int} the number of images to get
            return {DataLoader} the first n images from the image folder
            """

            images_dataset = ImageFolder(root=f"{self.data_path}/train", transform=self.tranform)

            images_dataset.samples = images_dataset.samples[:n]

            return DataLoader(images_dataset, batch_size=1, shuffle=False)
    

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
    
    def _embed_batch(self, 
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
        
        self.data = self.data.map(lambda x: self._embed_batch(x["image"]),
                                  batched=False)
        
    def _get_random_images(self,
                            n: int) -> Dataset:
        """
        Get the first n images of a dataset

        param n {int} the number of images to get
        return {Dataset} the first n images from the image folder
        """

        return self.data["train"].select(range(n))
    