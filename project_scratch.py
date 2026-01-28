import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


# Define transformations
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = datasets.ImageFolder('dataset/train', transform=transforms)
val_dataset = datasets.ImageFolder('dataset/val', transform=transforms)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=25, shuffle=True)


# Making the Neural Net
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # The following convolutional layers are made to semi resemble YOLO's own since this will be primarily tested against YOLO

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)  # 56->28
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)  # 28->14
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)  # 14->7
        )

        # I cannot replicate YOLO's 53 convo backbone so I have opted for 5 followed by the classifier head itself

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Reshapes [batch, 512, 7, 7] → [batch, 25088]
            nn.Linear(512 * 7 * 7, 256),  # Learns 25088→256 mapping
            nn.LeakyReLU(0.1),  # Non-linearity
            nn.Dropout(0.5),  # Randomly zeros 50% of neurons to prevent overfitting
            nn.Linear(256, 1),  # Takes the 256 features and turns them into one binary output
            nn.Sigmoid()  # Forces the output to [0,1] probability
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x


# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet().to(device)
loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 50
metrics_history = []

# Training
for epoch in range(num_epochs):

    model.train()
    running_loss = 0
    train_preds, train_labels = [], []

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images, labels = images.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_preds.extend((outputs > 0.5).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    # Validation
    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            loss = loss_func(outputs, labels)

            val_loss += loss.item()
            val_preds.extend((outputs > 0.5).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # Metrics
    train_acc = accuracy_score(train_preds, train_labels)
    val_acc = accuracy_score(val_preds, val_labels)
    val_precision = precision_score(val_preds, val_labels)
    val_recall = recall_score(val_preds, val_labels)
    val_f1 = f1_score(val_preds, val_labels)

    # Logging metrics
    metrics_history.append({
        'epoch': epoch + 1,
        'train_loss': running_loss / len(train_loader),
        'val_loss': val_loss / len(val_loader),
        'train_acc': train_acc,
        'metrics/accuracy_top1': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1_score': val_f1
    })

    print(f'Epoch {epoch + 1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, F1={val_f1:.4f}')

    # Saving to YOLO style CSV
    df = pd.DataFrame(metrics_history)
    df.to_csv('custom_model_metrics_history.csv', index=False)
    print("Metrics saved")

    torch.save(model.state_dict(), 'from_Scratch/custom_classifier_state_dict.pth')
