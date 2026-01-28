import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

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
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x


def evaluate_yolo_model(model_path):
    model = YOLO(model_path)
    results = model.val(data='dataset.yaml', split='test')

    accuracy = results.top1

    # 1. Get the raw matrix and number of classes
    matrix = results.confusion_matrix.matrix
    nc = results.confusion_matrix.nc  # Number of classes

    # 2. Extract TP, FP, FN excluding the background row/col
    tp = np.diag(matrix)[:nc]
    fp = matrix[:nc, nc:].sum(0) + matrix[:nc, :nc].sum(0) - tp
    fn = matrix[nc:, :nc].sum(1) + matrix[:nc, :nc].sum(1) - tp

    # 3. Calculate F1
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-9)
    f1_eval = np.mean(f1_per_class)

    return accuracy, f1_eval


def evaluate_pytorch_model(model_path, test_loader_param, device='cuda'):
    model = NeuralNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader_param:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1_torch = f1_score(y_true, y_pred, average='macro')

    return accuracy, f1_torch


def main():
    # Trained models
    models = {
        'YOLO_v26_1': 'runs/classify/pretrained_classifier_v26_/weights/best.pt',
        'YOLO_26_2': 'runs/classify/pretrained_classifier_v26_2/weights/best.pt',
        'YOLO_v8_1': 'runs/classify/pretrained_classifier_v8_/weights/best.pt',
        'YOLO_v8_2': 'runs/classify/pretrained_classifier_v8_2/weights/best.pt',
        'YOLO_26_non_pretrained_1': 'runs/classify/non_pretrained_classifier_v26_1st/weights/best.pt',
        'YOLO_26_non_pretrained_2': 'runs/classify/non_pretrained_classifier_v26_2nd/weights/best.pt',
        'From_Scratch': 'From_Scratch/custom_classifier_state_dict.pth',
    }

    results_list = []

    # Model eval
    for name, path in models.items():
        print(f"Evaluating {name}...")

        if path.endswith('.pt'):  # Only YOLO models end with pt
            acc, f1 = evaluate_yolo_model(path)

        else:  # If it does not it means it's from scratch
            test_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            test_dataset = datasets.ImageFolder('dataset/test', transform=test_transforms)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=25, shuffle=False)
            acc, f1 = evaluate_pytorch_model(path, test_loader)

        results_list.append({
            'Model': name,
            'Accuracy': acc,
            'F1 Score': f1
        })

    # Creating comparison dataframe
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.sort_values('F1 Score', ascending=False)

    print("\n" + "=" * 50)
    print("MODEL COMPARISON (Ranked by F1 Score)")
    print("=" * 50)
    print(comparison_df.to_string(index=False))
    print("=" * 50)

    # Save results
    comparison_df.to_csv('model_comparison.csv', index=False)


if __name__ == '__main__':
    main()
