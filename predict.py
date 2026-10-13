import os
import numpy as np
from pathlib import Path
from keras.models import load_model
from keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report

data_dir = Path("../datasets/LCC_FASD/test")
dataset = image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="binary",
    class_names=["real", "spoof"],
    batch_size=32,
    image_size=(224, 224),
    shuffle=False
)

y_true = np.concatenate([y.numpy().flatten() for _, y in dataset]).astype(int)

models_config = [
    ("Custom Model", Path("./outputs/custom-model/best_model.keras"), Path("./outputs/custom-model/best_threshold.txt")),
    ("Backbone Model", Path("./outputs/backbone-model/best_model.keras"), Path("./outputs/backbone-model/best_threshold.txt")),
    ("MultiBranch Model", Path("./outputs/multibranch-model/best_model.keras"), Path("./outputs/multibranch-model/best_threshold.txt"))
]

for name, model_path, thresh_path in models_config:
    model = load_model(model_path)
    threshold = float(open(thresh_path).read().strip())

    y_prob = model.predict(dataset, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    print(name)
    print(classification_report(y_true, y_pred, target_names=["real", "spoof"], digits=4))

