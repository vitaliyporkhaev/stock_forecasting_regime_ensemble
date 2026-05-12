import json
import os


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def load_metrics(path: str):
    with open(path, "r") as f:
        return json.load(f)
