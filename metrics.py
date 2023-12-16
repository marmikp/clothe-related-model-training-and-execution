import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from predict_pipeline import predict


def calculate_precision_recall_f1(true_labels, predicted_labels, average='weighted'):
    label_encoder = {label: i for i, label in enumerate(set(true_labels))}
    true_labels_numeric = np.array([label_encoder[label] for label in true_labels])
    predicted_labels_numeric = np.array([label_encoder[label] for label in predicted_labels])
    precision = precision_score(true_labels_numeric, predicted_labels_numeric, average=average, zero_division=0)
    recall = recall_score(true_labels_numeric, predicted_labels_numeric, average=average, zero_division=0)
    f1 = f1_score(true_labels_numeric, predicted_labels_numeric, average=average, zero_division=0)
    return precision, recall, f1


def calculate_accuracy(true_labels, predicted_labels):
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Lists must have the same length.")

    correct_predictions = sum(1 for true, predicted in zip(true_labels, predicted_labels) if true == predicted)

    accuracy = correct_predictions / len(true_labels)

    return accuracy


def get_preds(files, labels, pipeline):
    preds = []
    true_labels = []
    for file, label in tqdm(list(zip(files, labels)), desc='Predicting Labels'):
        try:
            preds.append(predict(file, pipeline))
            true_labels.append(label)
        except:
            continue
    return preds, true_labels


def get_metrics(preds, true_labels):
    accuracy = calculate_accuracy(preds, true_labels)
    precision, recall, f1 = calculate_precision_recall_f1(preds, true_labels)
    return preds, true_labels, accuracy, precision, recall, f1


def get_predictions_and_metrics(files, labels, pipeline):
    preds, true_labels = get_preds(files, labels, pipeline)
    accuracy = calculate_accuracy(preds, true_labels)
    precision, recall, f1 = calculate_precision_recall_f1(preds, true_labels)
    return preds, true_labels, accuracy, precision, recall, f1
