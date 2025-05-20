from sklearn.model_selection import train_test_split
from extensions import db
from models.training_data import TrainingData
import numpy as np


def data_split():
    data = db.session.execute(db.select(TrainingData)).scalars().all()
    X = [(e.Filename, e.Roi_X1, e.Roi_Y1, e.Roi_X2, e.Roi_Y2) for e in data]
    y_original = [e.ClassId for e in data]
    unique_labels = sorted(list(np.unique(y_original)))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    index_to_label = {i: label for i, label in enumerate(unique_labels)}
    y_mapped = [label_to_index[label] for label in y_original]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_mapped, test_size=0.2, stratify=y_mapped, random_state=42
    )
    return X_train, X_val, y_train, y_val, label_to_index, index_to_label
