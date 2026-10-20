from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)

from src.data.constants import CLASS_NAMES
from src.utils.serialization import to_python


def score_predictions(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return to_python({
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0),
        "per_class": {
            CLASS_NAMES[label]: {
                "precision": precision[index], "recall": recall[index],
                "f1": f1[index], "support": support[index],
            }
            for index, label in enumerate([0, 1])
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "confusion_matrix_labels": ["outside(0)", "inside(1)"],
    })
