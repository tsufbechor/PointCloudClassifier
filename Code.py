import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import NormalizeScale, SamplePoints, Compose
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import TransformerConv, knn_graph
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn.pool import voxel_grid, radius, fps

# --------------------
# 1. Set Seed for Reproducibility
# --------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# --------------------
# 2. Define Transformations
# --------------------
transform = Compose([
    NormalizeScale(),
    SamplePoints(1024)
])

# --------------------
# 3. Load the Dataset
# --------------------
train_dataset = ModelNet(root='data/ModelNet10', name='10', train=True, transform=transform)
test_dataset = ModelNet(root='data/ModelNet10', name='10', train=False, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# --------------------
# 4. Define Models
# KPCONV
class KPConv(nn.Module):
    def __init__(self, num_classes=10):
        super(KPConv, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        B, N = data.num_graphs, data.num_nodes // data.num_graphs
        x = data.pos.view(B, N, 3).transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
# --------------------


# --------------------
# PointNet
class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, data):
        B = data.num_graphs  # Batch size
        N = data.num_nodes // B  # Number of points per graph
        x = data.pos.view(B, N, 3).transpose(1, 2)  # Reshape and transpose to [batch_size, 3, num_points]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# PointTransformer
class PointTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super(PointTransformer, self).__init__()
        self.conv1 = TransformerConv(3, 64)
        self.conv2 = TransformerConv(64, 128)
        self.conv3 = TransformerConv(128, 256)
        self.global_pool = global_max_pool
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, data):
        edge_index = knn_graph(data.pos, k=16, batch=data.batch)
        x = F.relu(self.conv1(data.pos, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.global_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# CurveNet
class CurveNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CurveNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, data):
        B, N = data.num_graphs, data.num_nodes // data.num_graphs
        x = data.pos.view(B, N, 3).transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x).squeeze(2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# 5. Helper Functions for Visualization
# --------------------
def plot_label_distribution(dataset, dataset_name):
    labels = [data.y.item() for data in dataset]
    plt.figure()
    plt.hist(labels, bins=len(set(labels)), edgecolor='black')
    plt.title(f'{dataset_name} Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.savefig(f'{dataset_name}_label_distribution.png')


def plot_confusion_matrix(true_labels, predictions, epoch, model_name):
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name} (Epoch {epoch})')
    plt.savefig(f'{model_name}_confusion_matrix_epoch_{epoch}.png')
    plt.close()
from sklearn.metrics import classification_report

def plot_class_wise_accuracy(true_labels, predictions, model_name):
    report = classification_report(true_labels, predictions, output_dict=True)
    class_accuracy = {cls: metrics["precision"] for cls, metrics in report.items() if cls.isdigit()}
    
    plt.figure()
    plt.bar(class_accuracy.keys(), class_accuracy.values())
    plt.title(f'{model_name} Class-Wise Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(list(class_accuracy.keys()))
    plt.savefig(f'{model_name}_class_wise_accuracy.png')
    plt.close()
def plot_precision_recall_f1(true_labels, predictions, model_name):
    report = classification_report(true_labels, predictions, output_dict=True)
    precision = [metrics["precision"] for cls, metrics in report.items() if cls.isdigit()]
    recall = [metrics["recall"] for cls, metrics in report.items() if cls.isdigit()]
    f1 = [metrics["f1-score"] for cls, metrics in report.items() if cls.isdigit()]
    classes = list(range(len(precision)))

    x = np.arange(len(classes))
    width = 0.2

    plt.figure()
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')

    plt.title(f'{model_name} Precision, Recall, and F1-Score')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(x, classes)
    plt.legend()
    plt.savefig(f'{model_name}_precision_recall_f1.png')
    plt.close()
from sklearn.manifold import TSNE

def visualize_latent_space(model, loader, device, model_name):
    model.eval()
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            latent = model(data)  # Replace this with the penultimate layer if required
            latent_vectors.append(latent.cpu().numpy())
            labels.extend(data.y.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title(f'{model_name} Latent Space Visualization')
    plt.savefig(f'{model_name}_latent_space.png')
    plt.close()

# --------------------
# 6. Enhanced Training and Testing Functions
# --------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct += (out.argmax(1) == data.y).sum().item()
        total += data.num_graphs
        all_preds.extend(out.argmax(1).tolist())
        all_labels.extend(data.y.tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def test(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            correct += (out.argmax(1) == data.y).sum().item()
            total += data.num_graphs
            all_preds.extend(out.argmax(1).tolist())
            all_labels.extend(data.y.tolist())
    return total_loss / total, correct / total, all_preds, all_labels

# --------------------
# 7. Run Experiments and Enhanced Visualizations
# --------------------
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

os.chdir('visualizations')

# Plot EDA visualizations
plot_label_distribution(train_dataset, "Train")
plot_label_distribution(test_dataset, "Test")

# Train models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {
    "KPConv": KPConv(10),
    "PointTransformer": PointTransformer(10),
    "PointNet": PointNet(10),
    "CurveNet": CurveNet(10),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.NLLLoss()
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    all_preds, all_labels = [], []

    for epoch in range(1, 51):  # Train for 100 epochs
        train_loss, train_acc, train_preds, train_labels = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = test(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            plot_confusion_matrix(val_labels, val_preds, epoch, name)

    results[name] = {"train_losses": train_losses, "val_losses": val_losses,
                     "train_accuracies": train_accuracies, "val_accuracies": val_accuracies}

    # Save the model
    torch.save(model.state_dict(), f"best_{name}.pth")

    # Plot per-model results
    epochs = range(1, 51)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title(f"{name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{name}_loss.png")

    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{name}_accuracy.png")
    plot_class_wise_accuracy(val_labels, val_preds, name)
    plot_precision_recall_f1(val_labels, val_preds, name)
    visualize_latent_space(model, test_loader, device, name)
    


# Save aggregated plots
plt.figure()
for name, res in results.items():
    plt.plot(range(1, 51), res["train_losses"], label=f"{name} Train")
    plt.plot(range(1, 51), res["val_losses"], label=f"{name} Validation")
plt.title("Comparison of Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("all_models_loss_comparison.png")

plt.figure()
for name, res in results.items():
    plt.plot(range(1, 51), res["train_accuracies"], label=f"{name} Train")
    plt.plot(range(1, 51), res["val_accuracies"], label=f"{name} Validation")
plt.title("Comparison of Accuracies")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("all_models_accuracy_comparison.png")

print("Training complete. Results and visualizations saved.")