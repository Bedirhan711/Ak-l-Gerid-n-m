import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

# ----------------------------
# Hiperparametreler
# ----------------------------
batch_size = 8
num_epochs = 15
learning_rate = 0.001
num_classes = 3
data_dir = 'images/train'
class_names = ['cam', 'metal', 'plastic']
val_split = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Data Transforms
# ----------------------------
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ----------------------------
# Dataset
# ----------------------------
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(
    indices,
    test_size=val_split,
    stratify=[full_dataset.targets[i] for i in indices],
    random_state=42
)
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# Model
# ----------------------------
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Class-weight'leri hesapla
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(full_dataset.targets),
    y=full_dataset.targets
)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# Eğitim
# ----------------------------
best_val_loss = float('inf')
patience = 3
trigger_times = 0
dtrain, dval = [], []

print("Eğitim başlıyor...")
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_train_loss = running_loss / len(train_loader)
    dtrain.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    dval.append(avg_val_loss)
    val_acc = 100 * correct / total
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Accuracy = {val_acc:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_resnet50_model.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Erken durdurma tetiklendi.")
            break

print("Eğitim tamamlandı.")

# ----------------------------
# Kayıp Grafiği
# ----------------------------
plt.plot(dtrain, label='Train Loss')
plt.plot(dval, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train vs Val Loss')
plt.savefig('loss_plot.png')
plt.show()

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# ----------------------------
# Classification Report
# ----------------------------
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
print(report)
with open("classification_report.txt", "w") as f:
    f.write(report)

# ----------------------------
# Precision / Recall / F1-score
# ----------------------------
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
x = np.arange(len(class_names))

plt.figure(figsize=(10, 6))
bar_width = 0.25
plt.bar(x - bar_width, precision, width=bar_width, label='Precision')
plt.bar(x, recall, width=bar_width, label='Recall')
plt.bar(x + bar_width, f1, width=bar_width, label='F1-score')
plt.xticks(x, class_names)
plt.ylim(0, 1)
plt.ylabel("Skor")
plt.title("Precision / Recall / F1-Score")
plt.legend()
plt.tight_layout()
plt.savefig("prf_scores.png")
plt.show()

# ----------------------------
# GRAD-CAM Fonksiyonu
# ----------------------------
def visualize_gradcam(model, target_layer, image_path, class_names):
    model.eval()

    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    handle_forward.remove()
    handle_backward.remove()

    grads_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads_val, axis=(1, 2))

    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations_val[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig_img = np.array(img.resize((224, 224)))
    overlayed_img = heatmap * 0.4 + orig_img

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Prediction: {class_names[pred_class]}")
    plt.imshow(orig_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(overlayed_img.astype(np.uint8))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_output.png')
    plt.show()

# ----------------------------
# GRAD-CAM'i Kullan
# ----------------------------
target_layer = model.layer4[2]
image_path = 'images/test/plastic_01.jpg'  # Kendi test görsel yolunu buraya yaz
visualize_gradcam(model, target_layer, image_path, class_names)
