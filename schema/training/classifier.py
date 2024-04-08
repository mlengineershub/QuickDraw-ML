from abc import abstractmethod, ABC
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from transformers import pipeline, AutoFeatureExtractor, AutoModel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
from time import perf_counter
import seaborn as sns
import os
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from tqdm.auto import tqdm
import pandas as pd
from datetime import datetime
import re
from datasets import Dataset, DatasetDict
import mlflow
import torch
import numpy as np


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

    evaluation_experiment_id = "Benchmark"

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
    experiment_id: str

    def __init__(self, 
                 model_name: str,
                 output_dir: str,
                 device: str,
                 data_path: str,
                 label2idx: Dict[str, int],
                 seed: int) -> None:
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.seed = seed
        self.data_path = data_path

        self._set_seed()
        self._set_device()
        self._set_data(data_path, label2idx)
        self._set_training_args()
    
    def _set_seed(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    @abstractmethod
    def _set_device(self) -> None:
        pass

    @abstractmethod
    def _set_data(self, 
                  data_path: str,
                  label2idx: Dict[str, int]) -> None:
        pass

    @abstractmethod
    def _set_training_args(self,
                           **kwargs) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def compute_metrics(self) -> Dict[str, Any]:
        pass
    
    # we will see step by step which methods we will need to implement in the mother class



class FEClassifier(BaseClassifier, ABC):

    _regex = re.compile(r"(.+?)\(")

    # all attributes listed below
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
        
        self.experiment_id = "Feature Extraction"
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
        
        try:
            self.embedding_model.to(self.device)
        except:
            print(f"Device {self.device} not found, using cpu instead")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model.to(self.device)
    

    def _get_model_name(self) -> str:
        return self._regex.match(str(self.classifier_model)).group(1)
    
    @abstractmethod
    def embed(self) -> None:
        pass
        

class TransformersFEClassifier(FEClassifier):


    def __init__(self,
                 embedding_model_name: str,
                 classifier_model: SklearnModel, # type: ignore
                 output_dir: str,
                 data_path: str,
                 label2idx: Dict[str, int],
                 device: str,
                 seed: int,
                 **kwargs) -> None:

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name, **kwargs)            
        super().__init__(embedding_model_name=embedding_model_name,
                         classifier_model=classifier_model,
                         output_dir=output_dir,
                         data_path=data_path,
                         label2idx=label2idx,
                         device=device,
                         seed=seed)

        self.__time_mean_val = 0
        self.__time_mean_test = 0
    

    def _set_data(self,
                 data_path: str,
                 label2idx: Dict[str, int]) -> None:
    
        data = {"train": {"image": [], "labels" : []},
                    "val": {"image": [], "labels" : []},
                    "test": {"image": [], "labels" : []}}
        
        for key in data.keys():
            classes = os.listdir(f"{data_path}/{key}")
            images = []
            labels = []
            for class_name in tqdm(classes, desc=f"Loading {key} from {key}"):
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
          
          self.training_args = kwargs
          self.classifier_model.set_params(**kwargs)

    def embed(self) -> None:

        # FIX THIS METHOD USE MAP FOR DATASETDICT
        
        times_test = []
        times_val = []
        for key in self.data.keys():
            embeddings = []
            for image in tqdm(self.data[key]["image"], desc=f"Embedding {key}"):
                if key == "test" or key == "val":
                    start_time = perf_counter()
                with torch.inference_mode():
                    features = self.feature_extractor(images=image, return_tensors="pt")
                    if key == "test":
                        times_test.append(perf_counter() - start_time)
                    if key == "val":
                        times_val.append(perf_counter() - start_time)
                    output = self.embedding_model(**features)
                    embeddings.append(output.pooler_output.cpu().numpy())    
            self.data[key]["embeddings"] = embeddings
        
        self.__time_mean_test = np.mean(times_test)
        self.__time_mean_val = np.mean(times_val)
    
    def compute_metrics(self,
                        is_training: bool = True) -> Dict[str, Any]:


        # FIX BENCHMARKING TIME, PASS THROUGH THE EMB MODEL SOME SAMPLESA AND GET THE TIME ONLY FOR VAL SET
        metrics = {}
        for key in self.data.keys():
            if is_training and key == "test":
                continue
            if key == "test" or key == "val":
                start_time = perf_counter()
            y_true = self.data[key]["labels"]
            y_pred = self.classifier_model.predict(self.data[key]["embeddings"])
            if key == "test":
                self.__time_mean_test += perf_counter() - start_time
            if key == "val":
                self.__time_mean_val += perf_counter() - start_time
            metrics.update({f"{key}_accuracy": accuracy_score(y_true, y_pred),
                            f"{key}_weighted_precision": precision_score(y_true, y_pred, average="weighted"),
                            f"{key}_seighted_recall": recall_score(y_true, y_pred, average="weighted"),
                            f"{key}_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
                            f"{key}_macro_precision": precision_score(y_true, y_pred, average="macro"),
                            f"{key}_macro_recall": recall_score(y_true, y_pred, average="macro"),
                            f"{key}_macro_f1": f1_score(y_true, y_pred, average="macro"),
                            f"{key}_micro_precision": precision_score(y_true, y_pred, average="micro"),
                            f"{key}_micro_recall": recall_score(y_true, y_pred, average="micro"),
                            f"{key}_micro_f1": f1_score(y_true, y_pred, average="micro"),
                            f"{key}_mcc": matthews_corrcoef(y_true, y_pred)})
        
        if not is_training:
            metrics["time_mean_test"] = self.__time_mean_test
        metrics["time_mean_val"] = self.__time_mean_val

        return metrics
    
    def plot_confusion_matrix(self,
                              split: str) -> plt.Figure: 
            
        from sklearn.metrics import confusion_matrix

        y_true = self.data[split]["labels"]
        y_pred = self.classifier_model.predict(self.data[split]["embeddings"])

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=False, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"Confusion Matrix - {split}")

        return fig



    def train(self) -> None:

        print("Training model")

        if "embeddings" not in self.data["train"].column_names:
            print('No embeddings found, computing them')
            self.embed()

        try:
            self.run_name = f"{self.model_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            with mlflow.start_run(run_name=self.run_name, experiment_id=self.experiment_id):
                self.classifier_model.fit(self.data["train"]["embeddings"], self.data["train"]["labels"])
                mlflow.log_params(self.training_args)
                mlflow.log_metrics(self.compute_metrics())
                mlflow.log_artifact(self.plot_confusion_matrix("train"))
                mlflow.log_artifact(self.plot_confusion_matrix("val"))
        except Exception as e:
            self.run_name = None
            raise e
    
    def evaluate(self) -> Dict[str, Any]:
        
        eval_run_name = f"{self.model_name}_evaluation_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

        with mlflow.start_run(run_name=eval_run_name, experiment_id=self.evaluation_experiment_id):
            metrics = self.compute_metrics(is_training=False)
            mlflow.log_metrics(metrics=metrics)
            mlflow.log_artifact(self.plot_confusion_matrix("test"))
        
            

            
