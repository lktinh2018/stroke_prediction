import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load true labels and predicted labels from .npy files
true_labels = np.load("Gts.npy")
# predicted_labels = np.load("Outputs_XGBoost.npy")
predicted_labels = np.load("Outputs_RandomForest.npy")


# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predicted_labels, average="binary")
recall = recall_score(true_labels, predicted_labels, average="binary")
f1 = f1_score(true_labels, predicted_labels, average="binary")
accuracy = accuracy_score(true_labels, predicted_labels)


print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
