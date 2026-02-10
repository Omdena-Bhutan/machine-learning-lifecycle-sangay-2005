import json
import pickle
import numpy as np
import mlflow
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_params


def evaluate_model(model, test_texts, y_true, tokenizer, max_length: int = 128):
    # Tokenize on the fly
    enc = tokenizer(list(test_texts), truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        input_tensors = {k: torch.tensor(v) for k, v in enc.items()}
        outputs = model(**input_tensors)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    acc = float(accuracy_score(y_true, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary")
    cm = confusion_matrix(y_true, preds).tolist()

    metrics = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }

    with open("eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    try:
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact("eval_metrics.json")
    except Exception:
        pass

    print("Evaluation complete. Metrics written to eval_metrics.json")
    return metrics


if __name__ == "__main__":
    params = load_params()

    # Load model
    model_name = params["train"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    state_dict = torch.load("models/trained/model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Load test data (plain texts and labels)
    with open("data/processed/test.pkl", "rb") as f:
        test_texts, y_test = pickle.load(f)

    evaluate_model(model, test_texts, np.array(y_test), tokenizer, max_length=int(params["train"]["max_length"]))
