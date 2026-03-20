import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# =============================
# Device Setup
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================
# Paths
# =============================

train_path = "./imagenet-10"
val_path = "./imagenet-10"

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# =============================
# Transforms
# =============================

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =============================
# Dataset
# =============================

train_dataset = datasets.ImageFolder(train_path, transform=transform_train)
test_dataset = datasets.ImageFolder(val_path, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =============================
# Models
# =============================

models_dict = {}

alexnet = torchvision.models.alexnet(weights=None)
alexnet.classifier[6] = nn.Linear(4096, 10)
models_dict["alexnet"] = alexnet.to(device)

vgg16 = torchvision.models.vgg16(weights=None)
vgg16.classifier[6] = nn.Linear(4096, 10)
models_dict["vgg16"] = vgg16.to(device)

resnet = torchvision.models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
models_dict["resnet18"] = resnet.to(device)

criterion = nn.CrossEntropyLoss()

# =============================
# Training Function
# =============================

def train_model(model, name, epochs=5):

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    start_epoch = 0
    path = os.path.join(checkpoint_dir, f"{name}.pth")

    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming {name} from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader)

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_description(f"{name} Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        acc = 100 * correct / total
        print(f"{name} Epoch {epoch+1} Accuracy: {acc:.2f}%")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, path)

# =============================
# Test Function
# =============================

def test_model(model):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100 * correct / total

# =============================
# SimBA Attack
# =============================

def simba_attack(model, images, labels, epsilon=0.2, iters=10):

    model.eval()

    adv_images = images.clone().to(device)

    batch_size, c, h, w = adv_images.shape
    flat_dim = c * h * w

    for i in range(batch_size):

        x = adv_images[i:i+1]
        y = labels[i:i+1]

        perm = torch.randperm(flat_dim)

        for k in range(iters):

            idx = perm[k]

            diff = torch.zeros(flat_dim).to(device)
            diff[idx] = epsilon

            diff = diff.view(1, c, h, w)

            x_pos = torch.clamp(x + diff, 0, 1)
            x_neg = torch.clamp(x - diff, 0, 1)

            loss_pos = criterion(model(x_pos), y)
            loss_neg = criterion(model(x_neg), y)

            if loss_pos > loss_neg:
                x = x_pos
            else:
                x = x_neg

        adv_images[i] = x

    return adv_images

# =============================
# Evaluate Under Attack
# =============================

def evaluate_under_attack(model):

    model.eval()

    correct = 0
    total = 0

    for images, labels in tqdm(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        adv_images = simba_attack(model, images, labels)

        outputs = model(adv_images)

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return 100 * correct / total

# =============================
# Run Training + Evaluation
# =============================

results = {}

for name, model in models_dict.items():

    print(f"\nTraining {name}")

    train_model(model, name, epochs=5)

    clean_acc = test_model(model)

    adv_acc = evaluate_under_attack(model)

    results[name] = (clean_acc, adv_acc)

    print(f"{name} Clean Accuracy: {clean_acc:.2f}%")
    print(f"{name} Adversarial Accuracy: {adv_acc:.2f}%")

# =============================
# Final Results
# =============================

print("\nFinal Results")

for k, v in results.items():
    print(k, "Clean:", v[0], "Adversarial:", v[1])
