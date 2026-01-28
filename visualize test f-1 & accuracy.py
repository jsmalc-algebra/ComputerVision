import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('model_comparison.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy chart
df_sorted_acc = df.sort_values('Accuracy')
ax1.barh(df_sorted_acc['Model'], df_sorted_acc['Accuracy'], color='steelblue')
ax1.set_xlabel('Accuracy')
ax1.set_title('Model Accuracy Ranking')
ax1.grid(True, alpha=0.3, axis='x')

# F1 Score chart
df_sorted_f1 = df.sort_values('F1 Score')
ax2.barh(df_sorted_f1['Model'], df_sorted_f1['F1 Score'], color='coral')
ax2.set_xlabel('F1 Score')
ax2.set_title('Model F1 Score Ranking')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
