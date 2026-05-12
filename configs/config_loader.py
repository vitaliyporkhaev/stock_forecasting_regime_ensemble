import yaml
from pathlib import Path

def load_config(name: str):

    base_path = Path(__file__).parent

    file_path = base_path / f"{name}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(f"Config not found: {file_path}")

    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    return config
