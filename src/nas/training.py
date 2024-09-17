
import os
import json
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import classification_report, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import nni
import sys
import os

from  processing import get_data_loaders,get_weights

id2label = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"
}
label2id = {v: k for k, v in id2label.items()}

def compute_metrics(predictions, labels):
    try:
        # Ensure predictions and labels are list of lists
        if isinstance(predictions[0], torch.Tensor):
            predictions = [pred.cpu().tolist() for pred in predictions]
        if isinstance(labels[0], torch.Tensor):
            labels = [label.cpu().tolist() for label in labels]

        # Flatten the predictions and labels
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_labels = [item for sublist in labels for item in sublist]

        # Filter out padding (-100) from both predictions and labels
        filtered_predictions = []
        filtered_labels = []
        for p, l in zip(flat_predictions, flat_labels):
            if l != -100:
                filtered_predictions.append(p)
                filtered_labels.append(l)

        # # Debug prints
        # print(f"Filtered Predictions (sample): {filtered_predictions[:10]}")
        # print(f"Filtered Labels (sample): {filtered_labels[:10]}")

        # Convert to string labels
        true_predictions = [id2label[p] for p in filtered_predictions]
        true_labels = [id2label[l] for l in filtered_labels]

        # # More debug prints
        # print(f"True Predictions (sample): {true_predictions[:10]}")
        # print(f"True Labels (sample): {true_labels[:10]}")

        # Compute metrics using sklearn (for flat lists)
        sk_accuracy = sk_accuracy_score(true_labels, true_predictions)
        sk_f1 = sk_f1_score(true_labels, true_predictions, average='weighted')
        sk_report = sk_classification_report(true_labels, true_predictions)

        # Compute metrics using seqeval (for list of lists)
        seq_accuracy = accuracy_score([true_labels], [true_predictions])
        seq_f1 = f1_score([true_labels], [true_predictions])
        seq_report = classification_report([true_labels], [true_predictions])

        return {
            "accuracy": sk_accuracy,
            "f1": sk_f1,
            "classification_report": sk_report,
            "seqeval_accuracy": seq_accuracy,
            "seqeval_f1": seq_f1,
            "seqeval_report": seq_report
        }
    except Exception as e:
        print(f"Error in compute_metrics: {str(e)}")
        # print(f"Sample predictions: {predictions[:2]}")
        # print(f"Sample labels: {labels[:2]}")
        raise


def train_epoch(model, device, train_loader, optimizer, epoch, _class_weights):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        word_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)
        
        emissions = model(word_ids, attention_mask)
        loss = model.loss(emissions, labels, attention_mask,_class_weights)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(word_ids)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return total_loss / len(train_loader)

def test_epoch(model, device, test_loader,class_weights):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            word_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            try:
                emissions = model(word_ids, attention_mask)
                loss = model.loss(emissions, labels, attention_mask,class_weights)
                total_loss += loss.item()

                predictions = model.decode(emissions, attention_mask)
                # Convert to list and append
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().tolist())

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                print(f"Input shape: {word_ids.shape}")
                print(f"Attention mask shape: {attention_mask.shape}")
                print(f"Labels shape: {labels.shape}")
                raise

    avg_loss = total_loss / len(test_loader)

    try:
        metrics = compute_metrics(all_predictions, all_labels)

        print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {metrics["accuracy"]:.4f}, F1 Score: {metrics["f1"]:.4f}\n')
        print(f'Classification Report:\n{metrics["classification_report"]}')

        return avg_loss, metrics["accuracy"], metrics["f1"]

    except Exception as e:
        print(f"Error in compute_metrics: {str(e)}")
        print(f"Sample predictions: {all_predictions[:2]}")
        print(f"Sample labels: {all_labels[:2]}")
        print(f"Unique predictions: {set(sum(all_predictions, []))}")
        print(f"Unique labels: {set(sum(all_labels, []))}")
        raise


def evaluate_model(model):
    # Extract chosen values from the model's input choices
    model_config = {
        "embedding_dim": model.embedding_dim.chosen[0] if isinstance(model.embedding_dim.chosen, list) else model.embedding_dim.chosen,
        "cnn_filters": model.cnn_filters.chosen[0] if isinstance(model.cnn_filters.chosen, list) else model.cnn_filters.chosen,
        "cnn_kernel_size": model.cnn_kernel_size.chosen[0] if isinstance(model.cnn_kernel_size.chosen, list) else model.cnn_kernel_size.chosen,
        "lstm_hidden_size": model.lstm_hidden_size.chosen[0] if isinstance(model.lstm_hidden_size.chosen, list) else model.lstm_hidden_size.chosen,
        "lstm_num_layers": model.lstm_num_layers.chosen[0] if isinstance(model.lstm_num_layers.chosen, list) else model.lstm_num_layers.chosen,
        "lstm_dropout": model.lstm_dropout.chosen[0] if isinstance(model.lstm_dropout.chosen, list) else model.lstm_dropout.chosen,
        "use_highway": model.use_highway.chosen[0] if isinstance(model.use_highway.chosen, list) else model.use_highway.chosen,
        "use_crf": model.use_crf.chosen[0] if isinstance(model.use_crf.chosen, list) else model.use_crf.chosen
    }

    # Create a unique directory for this model evaluation
    timestamp = int(time.time())
    model_dir = f"model_results/BLSTM_CNN_CRF_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model configuration
    with open(f"{model_dir}/model_config.json", "w") as f:
        json.dump(model_config, f, indent=4)

    # Get data loaders (assuming batch size is fixed or defined elsewhere)
    train_loader, val_loader, test_loader = get_data_loaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    _class_weights = get_weights().to(device)
    model.to(device)
    
    # Use a fixed optimizer (you might want to make this configurable in the model as well)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    accuracies = []
    f1_scores = []

    for epoch in range(10):  # Increased to 10 epochs
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch, _class_weights)
        val_loss, accuracy, f1 = test_epoch(model, device, val_loader,_class_weights)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        f1_scores.append(f1)

        nni.report_intermediate_result({
            'default': f1,
            'accuracy': accuracy,
            'f1': f1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

    # Final test
    test_loss, test_accuracy, test_f1 = test_epoch(model, device, test_loader,class_weights=_class_weights)

    # Plot and save metrics
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{model_dir}/loss_plot.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, accuracies, 'g', label='Accuracy')
    plt.plot(epochs, f1_scores, 'p', label='F1 Score')
    plt.title('Accuracy and F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f"{model_dir}/metrics_plot.png")
    plt.close()

    # Generate and save confusion matrix
    y_true, y_pred = get_predictions(model, device, test_loader)
    y_true_flat = [item for sublist in y_true for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(id2label.values()),
                yticklabels=list(id2label.values()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{model_dir}/confusion_matrix.png")
    plt.close()

    # Save final results
    final_results = {
        'default': test_f1,
        'accuracy': test_accuracy,
        'f1': test_f1,
        'test_loss': test_loss
    }
    with open(f"{model_dir}/final_results.json", "w") as f:
        json.dump(final_results, f, indent=4)

    nni.report_final_result(final_results)
# Helper function to get predictions (implement as needed)
def get_predictions(model, device, loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch in loader:
            input,input2 = batch['input_ids'].to(device),batch['attention_mask'].to(device)
            outputs = model(input,input2)
            predictions = outputs.argmax(dim=-1).cpu().tolist()
            labels = batch['labels'].cpu().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    return all_labels, all_predictions
