import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


csv_path = './content/data/daic_rmse/Confusion matrix/conf_matrix_4.csv'
data = pd.read_csv(csv_path, delimiter='\t', index_col=0)

fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(data.values, cmap='Blues')

# set label
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')

# Setting labels for axis scales
ax.set_xticks(np.arange(len(data.columns)))
ax.set_yticks(np.arange(len(data.index)))
ax.set_xticklabels(data.columns, fontsize=10)
ax.set_yticklabels(data.index, fontsize=10)

# Displaying values in confusion matrices
for i in range(len(data)):
    for j in range(len(data)):
        ax.text(j, i, data.values[i, j], ha='center', va='center', color='black', fontsize=8)

# Adjustment of subgraph layout
plt.tight_layout()

# Create splits directory
if not os.path.isdir('./data/fig_Confusion_matrix/'):
    os.makedirs('./data/fig_Confusion_matrix/')

plt.savefig('./data/fig_Confusion_matrix/d-c2-Confusion_matrix.svg')
plt.show()
