import pandas as pd
import matplotlib.pyplot as plt

results_v1 = {
    "YOLO_v26_1 pretrained": 'runs/classify/pretrained_classifier_v26_/results.csv',
    "YOLO_v8_1 pretrained": 'runs/classify/pretrained_classifier_v8_/results.csv',
    "YOLO_v26_1 non-pretrained ": 'runs/classify/non_pretrained_classifier_v26_1st/results.csv',
    "Custom made model": 'From_Scratch/custom_model_metrics_history.csv'
}

results_v2 = {
    "YOLO_v8_2 pretrained": 'runs/classify/pretrained_classifier_v8_2/results.csv',
    "YOLO_v26_2 pretrained": 'runs/classify/pretrained_classifier_v26_2/results.csv',
    "YOLO_v26_2 non-pretrained": 'runs/classify/non_pretrained_classifier_v26_2nd/results.csv'
}

# I shall only be plotting accuracy over epochs rather than F-1 score due to it being too complicated and time-consuming
# This is due to F-1 score (on YOLO) being possible to acquire only through validating again
# i shall still be using F-1 score and accuracy in testing results visualization

# Plotting V1 of models before hyper-param adjustments
for label, csv_path in results_v1.items():
    results = pd.read_csv(csv_path)
    results.columns = results.columns.str.strip()
    plt.plot(results['epoch'], results['metrics/accuracy_top1'], label=label, marker='o', markersize=6, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('V1 Models comparison - Validation Accuracy per Epoch')
plt.show()

# With the knowledge and graphs of V1, V2 models were made and are now plotted as well
for label, csv_path in results_v2.items():
    results = pd.read_csv(csv_path)
    results.columns = results.columns.str.strip()
    plt.plot(results['epoch'], results['metrics/accuracy_top1'], label=label, marker='o', markersize=6, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('V2 Models comparison - Validation Accuracy per Epoch')
plt.show()

# Finally all the models are plotted together to be compared against each other
all_results = {**results_v1, **results_v2}
for label, csv_path in all_results.items():
    results = pd.read_csv(csv_path)
    results.columns = results.columns.str.strip()
    plt.plot(results['epoch'], results['metrics/accuracy_top1'], label=label, marker='o', markersize=6, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('All models comparison - Validation Accuracy per Epoch')
plt.show()

# After making the previous Graphs I saw that my model tanks the graphic and have opted to make ones that only show YOLO
results_v1_YOLO = {
    "YOLO_v26_1 pretrained": 'runs/classify/pretrained_classifier_v26_/results.csv',
    "YOLO_v8_1 pretrained": 'runs/classify/pretrained_classifier_v8_/results.csv',
    "YOLO_v26_1 non-pretrained ": 'runs/classify/non_pretrained_classifier_v26_1st/results.csv',
}

all_YOLO = {**results_v1_YOLO, **results_v2}

for label, csv_path in results_v1_YOLO.items():
    results = pd.read_csv(csv_path)
    results.columns = results.columns.str.strip()
    plt.plot(results['epoch'], results['metrics/accuracy_top1'], label=label, marker='o', markersize=6, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('YOLO v1 models comparison - Validation Accuracy per Epoch')
plt.show()

for label, csv_path in all_YOLO.items():
    results = pd.read_csv(csv_path)
    results.columns = results.columns.str.strip()
    plt.plot(results['epoch'], results['metrics/accuracy_top1'], label=label, marker='o', markersize=6, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('All YOLO models comparison - Validation Accuracy per Epoch')
plt.show()





