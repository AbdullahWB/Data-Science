import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score, accuracy_score, classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

# Create a directory to save plots if it doesn't exist
PLOT_DIR = "classification_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- 1. Fetching and preparing the MNIST dataset ---
print("Fetching MNIST dataset...")
mist = fetch_openml('mnist_784', as_frame=True, parser='auto')

X, y = mist.data, mist.target

# Convert pixel data to integers (they are typically float64 by default from fetch_openml)
X = X.astype(np.uint8)

# Convert target labels to integer type for consistent classification
y = y.astype(np.uint8)

# --- 2. Visualizing a digit (and saving it) ---
def plot_digit(image_data, filename="digit.png"):
    """
    Plots a single MNIST digit from a flattened array and saves it to a file.

    Args:
        image_data (numpy.ndarray or pandas.Series): A 1D array/Series of 784 pixel values.
        filename (str): The name of the file to save the plot.
    """
    if isinstance(image_data, pd.Series):
        image_data = image_data.to_numpy()

    image = image_data.reshape(28, 28)
    plt.figure(figsize=(4, 4)) # Create a new figure for each plot
    plt.imshow(image, cmap='binary')
    plt.axis('off')
    plt.title(f"Digit: {y.iloc[0]}" if filename == "digit.png" else "Noisy Digit")
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close() # Close the figure to free up memory

some_digit = X.iloc[0]
print(f"Label for the first digit: {y.iloc[0]}")
plot_digit(some_digit, filename="first_digit_example.png")
print(f"Saved example digit plot to {os.path.join(PLOT_DIR, 'first_digit_example.png')}")

# --- 3. Splitting the dataset into training and testing sets ---
x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[60000:]
print(f"Training set size: {len(x_train)} samples")
print(f"Test set size: {len(x_test)} samples")

# --- 4. Training a Binary Classifier (e.g., for digit '5') ---
print("\n--- Training Binary Classifier (SGD for digit '5') ---")
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)

prediction_first_digit = sgd_clf.predict([some_digit.to_numpy()])
print(f"Prediction for the first digit (is it a '5'?): {prediction_first_digit[0]}")

# --- 5. Evaluating the Binary Classifier ---
y_train_pred_5 = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)

conf_matrix_5 = confusion_matrix(y_train_5, y_train_pred_5)
print(f"Confusion Matrix for '5' classifier:\n{conf_matrix_5}")

precision_5 = precision_score(y_train_5, y_train_pred_5)
recall_5 = recall_score(y_train_5, y_train_pred_5)
f1_5 = f1_score(y_train_5, y_train_pred_5)
print(f"Precision for '5' classifier: {precision_5:.4f}")
print(f"Recall for '5' classifier: {recall_5:.4f}")
print(f"F1-Score for '5' classifier: {f1_5:.4f}")

y_scores_5 = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores_5)
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.legend(loc="center left")
plt.ylim([0, 1])
plt.title("Precision vs. Recall for '5' classifier")
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "precision_recall_curve_5.png"))
plt.close()
print(f"Saved Precision-Recall curve to {os.path.join(PLOT_DIR, 'precision_recall_curve_5.png')}")

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores_5)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label="SGD Classifier (ROC Curve)")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (ROC Curve)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Receiver Operating Characteristic (ROC) Curve for '5' classifier")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(PLOT_DIR, "roc_curve_5.png"))
plt.close()
print(f"Saved ROC curve to {os.path.join(PLOT_DIR, 'roc_curve_5.png')}")

roc_auc_5 = roc_auc_score(y_train_5, y_scores_5)
print(f"ROC AUC Score for '5' classifier: {roc_auc_5:.4f}")

# --- 6. Multiclass Classification (One-vs-One Strategy) ---
print("\n--- Training Multiclass Classifier (One-vs-One SGD) ---")
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(x_train, y_train)

multiclass_prediction_first_digit = ovo_clf.predict([some_digit.to_numpy()])
print(f"Multiclass prediction for the first digit (OVO-SGD): {multiclass_prediction_first_digit[0]}")

# --- 7. Evaluating the Multiclass Classifier (OVO-SGD) ---
y_train_pred_multiclass = cross_val_predict(ovo_clf, x_train, y_train, cv=3)

conf_matrix_multiclass = confusion_matrix(y_train, y_train_pred_multiclass)
print(f"Confusion Matrix for Multiclass classifier (OVO-SGD):\n{conf_matrix_multiclass}")

plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix_multiclass, cmap=plt.cm.Blues)
plt.title("Multiclass Confusion Matrix (OVO-SGD)")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.grid(False)
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix_ovo_sgd.png"))
plt.close()
print(f"Saved OVO-SGD Confusion Matrix to {os.path.join(PLOT_DIR, 'confusion_matrix_ovo_sgd.png')}")

precision_multiclass = precision_score(y_train, y_train_pred_multiclass, average='macro')
recall_multiclass = recall_score(y_train, y_train_pred_multiclass, average='macro')
f1_multiclass = f1_score(y_train, y_train_pred_multiclass, average='macro')

print(f"Precision for Multiclass classifier (OVO-SGD): {precision_multiclass:.4f}")
print(f"Recall for Multiclass classifier (OVO-SGD): {recall_multiclass:.4f}")
print(f"F1-Score for Multiclass classifier (OVO-SGD): {f1_multiclass:.4f}")

# --- 8. Training a K-Nearest Neighbors (KNN) Classifier ---
print("\n--- Training K-Nearest Neighbors (KNN) Classifier ---")
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(x_train, y_train)

knn_prediction_first_digit = knn_clf.predict([some_digit.to_numpy()])
print(f"KNN prediction for the first digit: {knn_prediction_first_digit[0]}")

# --- 9. Evaluating the KNN Classifier on Test Set ---
print("\n--- Evaluating KNN Classifier on Test Set ---")
y_test_pred_knn = knn_clf.predict(x_test)

overall_accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
print(f"Overall Accuracy for KNN on Test Set: {overall_accuracy_knn:.4f}")

print("\nPer-number Accuracy for KNN on Test Set:")
report_knn = classification_report(y_test, y_test_pred_knn, target_names=[str(i) for i in range(10)], output_dict=True)
for digit in range(10):
    if str(digit) in report_knn:
        print(f"  Accuracy for digit '{digit}': {report_knn[str(digit)]['recall']:.4f}") # Recall is accuracy for a single class when all other classes are negative
    else:
        print(f"  Digit '{digit}' not found in test set or predictions.")

conf_matrix_knn = confusion_matrix(y_test, y_test_pred_knn)
print(f"\nConfusion Matrix for KNN classifier on Test Set:\n{conf_matrix_knn}")

plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix_knn, cmap=plt.cm.Greens)
plt.title("KNN Confusion Matrix on Test Set")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.grid(False)
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix_knn_test.png"))
plt.close()
print(f"Saved KNN Test Set Confusion Matrix to {os.path.join(PLOT_DIR, 'confusion_matrix_knn_test.png')}")

# --- 10. Adding Random Noise and Denoising with KNN ---
print("\n--- Denoising with KNN ---")
x_train_mod = x_train.copy()
x_test_mod = x_test.copy()

noise = np.random.randint(0, 100, (len(x_train_mod), 784))
x_train_mod = (x_train_mod.to_numpy().astype(np.float32) + noise).astype(np.uint8)

noise = np.random.randint(0, 100, (len(x_test_mod), 784))
x_test_mod = (x_test_mod.to_numpy().astype(np.float32) + noise).astype(np.uint8)

y_train_mod = y_train.copy()
y_test_mod = y_test.copy()

# Plot a noisy digit from the modified test set.
print("Noisy digit (x_test_mod.iloc[0]):")
plot_digit(x_test_mod.iloc[0], filename="noisy_digit_example.png")
print(f"Saved noisy digit plot to {os.path.join(PLOT_DIR, 'noisy_digit_example.png')}")

knn_clf_denoise = KNeighborsClassifier(n_neighbors=5)
knn_clf_denoise.fit(x_train_mod, y_train_mod)

clean_digit_prediction = knn_clf_denoise.predict([x_test_mod.iloc[0].to_numpy()])[0]
print(f"Predicted clean digit label from noisy image: {clean_digit_prediction}")

# --- 11. Random Forest Classifier for Multiclass Classification ---
print("\n--- Training Random Forest Classifier ---")
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(x_train, y_train)

# --- 12. Evaluating the Random Forest Classifier on Test Set ---
print("\n--- Evaluating Random Forest Classifier on Test Set ---")
y_test_pred_forest = forest_clf.predict(x_test)

overall_accuracy_forest = accuracy_score(y_test, y_test_pred_forest)
print(f"Overall Accuracy for Random Forest on Test Set: {overall_accuracy_forest:.4f}")

print("\nPer-number Accuracy for Random Forest on Test Set:")
report_forest = classification_report(y_test, y_test_pred_forest, target_names=[str(i) for i in range(10)], output_dict=True)
for digit in range(10):
    if str(digit) in report_forest:
        print(f"  Accuracy for digit '{digit}': {report_forest[str(digit)]]['recall']:.4f}")
    else:
        print(f"  Digit '{digit}' not found in test set or predictions.")

conf_matrix_forest = confusion_matrix(y_test, y_test_pred_forest)
print(f"\nConfusion Matrix for Random Forest classifier on Test Set:\n{conf_matrix_forest}")

plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix_forest, cmap=plt.cm.Purples)
plt.title("Random Forest Confusion Matrix on Test Set")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.grid(False)
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix_random_forest_test.png"))
plt.close()
print(f"Saved Random Forest Test Set Confusion Matrix to {os.path.join(PLOT_DIR, 'confusion_matrix_random_forest_test.png')}")
