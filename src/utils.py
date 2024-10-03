import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss during model training.
    
    Parameters:
        history: The history object returned by the Keras model's fit method, 
                 containing the training and validation metrics.
    
    This function produces two plots:
    1. A plot showing the training and validation accuracy over epochs.
    2. A plot showing the training and validation loss over epochs.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(14, 5))
    
    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots a confusion matrix using true and predicted labels.
    
    Parameters:
        y_true: Array-like of true labels.
        y_pred: Array-like of predicted labels.
    
    This function generates a heatmap for the confusion matrix to visualize
    the model's performance in terms of true positives, false positives, 
    true negatives, and false negatives.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_pred_probs):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC).
    
    Parameters:
        y_true: Array-like of true labels.
        y_pred_probs: Array-like of predicted probabilities for the positive class.
    
    The ROC curve illustrates the trade-off between the true positive rate (TPR) 
    and the false positive rate (FPR) at various threshold settings. 
    The AUC value quantifies the overall performance of the model.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
