import os
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_params, ensure_dir


class DataPreprocessor:
    def __init__(self, data_path: str, max_length: int = 128):
        self.data_path = data_path
        self.max_length = max_length

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(self, test_size: float):
        df = pd.read_csv(self.data_path)
        # Expect columns: review, sentiment (positive/negative or 1/0)
        if "review" not in df.columns:
            # Fallback common names
            for cand in ["text", "review_text", "content"]:
                if cand in df.columns:
                    df.rename(columns={cand: "review"}, inplace=True)
                    break
        if "sentiment" not in df.columns:
            for cand in ["label", "target", "sent"]:
                if cand in df.columns:
                    df.rename(columns={cand: "sentiment"}, inplace=True)
                    break
        assert "review" in df.columns and "sentiment" in df.columns, "Dataset must have 'review' and 'sentiment' columns"

        df["review"] = df["review"].apply(self.clean_text)
        # Map labels
        if df["sentiment"].dtype == object:
            label_map = {"negative": 0, "positive": 1, "neg": 0, "pos": 1}
            y = df["sentiment"].str.lower().map(label_map)
        else:
            y = df["sentiment"].astype(int)
        y = y.fillna(0).astype(int).values

        texts = df["review"].tolist()

        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, y, test_size=test_size, random_state=42, stratify=y
        )

        ensure_dir("data/processed")
        # Save plain texts and labels; tokenization will be done in training to avoid torch import here
        with open("data/processed/train.pkl", "wb") as f:
            pickle.dump((X_train_texts, y_train), f)
        with open("data/processed/test.pkl", "wb") as f:
            pickle.dump((X_test_texts, y_test), f)
        print("Data preprocessing complete (plain text splits): data/processed/train.pkl, data/processed/test.pkl")


if __name__ == "__main__":
    params = load_params()
    dp = DataPreprocessor(
        data_path=params["data"]["dataset_path"],
        max_length=params["train"]["max_length"],
    )
    dp.preprocess(test_size=params["data"]["test_size"])
