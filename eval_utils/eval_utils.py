"""
Evaluation utilities for model assessment.
Includes metrics calculation, confusion matrix visualization, and model evaluation.
"""
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score


def get_f1_score(y_true, y_pred, average='weighted', report=False):
    """
    Calculate F1 score for predictions.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging strategy ('weighted', 'macro', 'micro')
        report: Whether to print classification report
    
    Returns:
        F1 score (float)
    """
    if report:
        print("Classification Report:\n")
        print(classification_report(y_true, y_pred, zero_division=1))
    else:
        f1 = f1_score(y_true, y_pred, average=average)
        print(f"F1 Score: {f1:.4f}")
        return f1


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure (optional)
    
    Returns:
        Matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close(fig)
    return fig


def model_eval(model, data_loader, criterion=None, path="", device=None, 
               prefix="Model Evaluation", class_names=None):
    """
    Comprehensive model evaluation function.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for the dataset
        criterion: Loss function (optional)
        path: Path to save confusion matrix
        device: Device to use for computation ('cpu' or 'cuda')
        prefix: Title/prefix for printed metrics
        class_names: List of class names for classification
    
    Returns:
        Dictionary containing metrics: accuracy, loss, f1_score, classification_report
    """
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            _, target_class = targets.max(1)
            
            y_true.extend(target_class.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    report_dict = classification_report(y_true, y_pred, target_names=class_names, 
                                       output_dict=True, zero_division=0)
    if isinstance(report_dict, dict):
        accuracy = report_dict["accuracy"] * 100
    
    avg_loss = total_loss / len(data_loader) if criterion else None
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Print results
    print("\n" + "=" * 50)
    print(f"{prefix}")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.2f}%")
    if criterion:
        print(f"Average Loss: {avg_loss:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Save confusion matrix
    if class_names and path:
        save_path = f"{path}/{prefix.replace(' ', '_')}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=save_path)
    
    metrics = {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "f1_score": f1,
        "classification_report": report_dict,
    }
    return metrics


def hierarchical_model_eval(model, data_loader, criterion=None, path="", device=None,
                           prefix="Hierarchical Model Evaluation", class_names=None):
    """
    Evaluation function for hierarchical models (like Baseline 9).
    These models output both person and group predictions.
    
    Args:
        model: Hierarchical PyTorch model
        data_loader: DataLoader (expects person_labels and group_labels)
        criterion: Loss function (optional)
        path: Path to save confusion matrix
        device: Device to use
        prefix: Title for metrics
        class_names: Dict with 'person' and 'group' class names
    
    Returns:
        Dictionary containing group activity metrics
    """
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, person_labels, group_labels in data_loader:
            inputs = inputs.to(device)
            person_labels = person_labels.to(device)
            group_labels = group_labels.to(device)
            
            outputs = model(inputs)
            
            if criterion:
                # Calculate combined loss (adjust weights as needed)
                loss_person = criterion(outputs['person_output'], person_labels)
                loss_group = criterion(outputs['group_output'], group_labels)
                loss = 0.7 * loss_group + 0.3 * loss_person
                total_loss += loss.item()
            
            # Evaluate on group activity (main task)
            _, predicted = outputs['group_output'].max(1)
            _, target_class = group_labels.max(1)
            
            y_true.extend(target_class.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    group_class_names = class_names['group'] if isinstance(class_names, dict) else class_names
    report_dict = classification_report(y_true, y_pred, target_names=group_class_names,
                                       output_dict=True, zero_division=0)
    if isinstance(report_dict, dict):
        accuracy = report_dict["accuracy"] * 100
    
    avg_loss = total_loss / len(data_loader) if criterion else None
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Print results
    print("\n" + "=" * 50)
    print(f"{prefix}")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.2f}%")
    if criterion:
        print(f"Average Loss: {avg_loss:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=group_class_names, zero_division=0))
    
    # Save confusion matrix
    if group_class_names and path:
        save_path = f"{path}/{prefix.replace(' ', '_')}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, class_names=group_class_names, save_path=save_path)
    
    metrics = {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "f1_score": f1,
        "classification_report": report_dict,
    }
    return metrics
