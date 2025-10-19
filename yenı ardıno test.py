import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
from PIL import Image
import cv2

# ----------------------------
# 1. Hiperparametreler & Yol Tanımları
# ----------------------------
batch_size   = 8
num_epochs   = 15
learning_rate= 1e-3
data_dir     = 'images/train'     
class_names  = ['cam','metal','plastic']
val_split    = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Ürün Maskesi Fonksiyonu
# ----------------------------
def mask_product_hsv(pil_img):
    # PIL Image'ı OpenCV formatına çevir
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # HSV uzayına çevir
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Burada uygun HSV aralığını ayarla (örnek):
    lower = np.array([0, 30, 30])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Maskeyi uygula (ürün dışı alanları siyah yap)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Tekrar RGB'ye çevir ve PIL formatına döndür
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

# ----------------------------
# 3. Özel Augmentasyon Dataseti
# ----------------------------
from torchvision.datasets import ImageFolder
from torchvision import transforms

class CustomAugmentedDataset(ImageFolder):
    def __init__(self, root, class_names):
        super().__init__(root=root, transform=None)
        self.class_names = class_names
        self.transforms_by_class = {
            'cam': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3,[0.5]*3)
            ]),
            'plastic': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3,[0.5]*3)
            ]),
            'metal':  transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3,[0.5]*3)
            ])
        }

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')

        # Ürün maskesi uygula
        img = mask_product_hsv(img)

        class_name = self.class_names[label]
        img = self.transforms_by_class[class_name](img)
        return img, label

# ----------------------------
# 4. Dataset & DataLoader
# ----------------------------
full_dataset = CustomAugmentedDataset(data_dir, class_names)
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices,
    test_size=val_split,
    stratify=[full_dataset.targets[i] for i in indices],
    random_state=42
)
train_ds = Subset(full_dataset, train_idx)
val_ds   = Subset(full_dataset, val_idx)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

# ----------------------------
# 5. Model, Kayıp, Optimizasyon
# ----------------------------
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# Dengesiz veri için class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_names)),
    y=full_dataset.targets
)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float).to(device))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# 6. Eğitim Döngüsü
# ----------------------------
best_val_acc = 0.0
patience = 3  # Erken durdurma için sabır sayısı
patience_counter = 0

train_losses = []
val_accuracies = []

all_labels = []
all_preds = []

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch}: Train Loss={running_loss / len(train_loader):.4f}, Val Acc={val_acc * 100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_resnet34(2)(3)_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Erken Durdurma: Model gelişmedi, eğitim durduruluyor.")
            break

# ----------------------------
# 7. Grafikler ve Analizler
# ----------------------------
plt.plot(train_losses, label='Train Loss')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.title('Train Loss vs Val Accuracy')
plt.savefig('loss_accuracy_plot.png')
plt.show()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Classification Report
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
print(report)
with open("classification_report.txt", "w") as f:
    f.write(report)

# Precision, Recall, F1-Score Bar Plot
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
x = np.arange(len(class_names))
plt.figure(figsize=(10, 6))
bar_width = 0.25
plt.bar(x - bar_width, precision, width=bar_width, label='Precision')
plt.bar(x, recall, width=bar_width, label='Recall')
plt.bar(x + bar_width, f1, width=bar_width, label='F1-score')
plt.xticks(x, class_names)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Precision / Recall / F1-Score")
plt.legend()
plt.tight_layout()
plt.savefig("prf_scores.png")
plt.show()

print("Eğitim tamamlandı. En iyi val accuracy:", best_val_acc * 100)
