import pickle
from pathlib import Path
from .config import BASE_DIR

BASE_DIR = Path(__file__).resolve().parent.parent 

model_path = BASE_DIR / "models" / "linear_regression.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)


def predict(data_dict):
    features = [[
        float(data_dict["hubraum_l"]),
        int(data_dict["verkaufszahl"]),
        float(data_dict["kundenzufriedenheit"])
    ]]

    return model.predict(features)[0]