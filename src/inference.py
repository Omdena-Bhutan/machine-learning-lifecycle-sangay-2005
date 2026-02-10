import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_model = None
_tokenizer = None
_model_name = "distilbert-base-uncased"
_model_path = "models/trained/model.pt"


def _load():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(_model_name, num_labels=2)
        state = torch.load(_model_path, map_location="cpu")
        _model.load_state_dict(state)
        _model.eval()


def predict(text: str):
    _load()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        conf, label = torch.max(probs, dim=1)
    return {
        "sentiment": "positive" if label.item() == 1 else "negative",
        "confidence": float(conf.item()),
    }
