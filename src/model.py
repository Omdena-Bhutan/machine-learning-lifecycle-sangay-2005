import os
import json
import pickle
import numpy as np
import mlflow
import mlflow.transformers
import torch
from typing import Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from utils import load_params, ensure_dir


class HFDictDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, List], labels: np.ndarray):
        self.encodings = encodings
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class SentimentModel:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        # Freeze base encoder, train head only if available
        if hasattr(self.model, "distilbert"):
            for p in self.model.distilbert.parameters():
                p.requires_grad = False

    def train(
        self,
        train_data_path: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        max_length: int,
    ):
        # Load plain texts and labels produced by prepare stage
        with open(train_data_path, "rb") as f:
            train_texts, y_train = pickle.load(f)

        # Tokenize here to avoid torch import in prepare stage
        enc = self.tokenizer(
            list(train_texts),
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        train_ds = HFDictDataset(enc, np.array(y_train))

        training_args = TrainingArguments(
            output_dir="results",
            evaluation_strategy="no",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_dir="logs",
            save_strategy="epoch",
            report_to=[],
        )

        ensure_dir("models/trained")

        with mlflow.start_run():
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_length", max_length)

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_ds,
            )

            trainer.train()

            # Save HF model state dict
            model_save_path = "models/trained/model.pt"
            torch.save(self.model.state_dict(), model_save_path)

            # Log artifacts & model
            mlflow.log_artifact(model_save_path, artifact_path="model_artifacts")
            try:
                mlflow.transformers.log_model(self.model, artifact_path="hf_model")
            except Exception:
                # On Windows/limited envs, this may fail due to dependencies; ignore to proceed
                pass

            # Store basic training metric if available
            last = None
            if getattr(trainer.state, "log_history", None):
                for h in reversed(trainer.state.log_history):
                    if "loss" in h:
                        last = {"train_loss": float(h["loss"]) }
                        break
            metrics = last or {}
            with open("metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.log_artifact("metrics.json")

        print("Training complete. Model saved to models/trained/model.pt")


if __name__ == "__main__":
    params = load_params()
    sm = SentimentModel(model_name=params["train"]["model_name"])
    sm.train(
        train_data_path="data/processed/train.pkl",
        epochs=int(params["train"]["epochs"]),
        batch_size=int(params["train"]["batch_size"]),
        learning_rate=float(params["train"]["learning_rate"]),
        max_length=int(params["train"]["max_length"]),
    )
