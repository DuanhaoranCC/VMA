import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

categories = ['1', '5', '10', '20', '30', '40']
values = [
    [83, 83.8, 82.7, 81.4, 80, 77],  # 数据集1
    [80.4, 82, 81.6, 82.2, 82.4, 80],  # 数据集2
    [82.2, 83, 82.9, 83, 83.3, 81],  # 数据集3
    # [79.5, 79.9, 80.0, 79, 90]  # 数据集4
]
dataset = ['Cora', 'PubMed', 'Computer']
Cora = np.array([[83, 83.8, 82.7, 81.4, 80, 77]])
Wiki = np.array([[80.4, 82, 81.6, 82.2, 82.4, 80]])
PubMed = np.array([[82.2, 83, 82.9, 83, 83.3, 81]])
z = pd.DataFrame(np.concatenate([Cora, Wiki, PubMed], axis=0),
                 columns=['1', '5', '10', '20', '30', '40'],
                 index=["Cora", "PubMed", "Computer"])
sns.set_palette("pastel")
# sns.set(style='darkgrid')
sns.lineplot(data=z.transpose(), markers=True, linewidth=3,
             markersize=12, palette=['#f1c81f', '#0a263d', '#884c30'])
plt.xlabel("t")
plt.ylabel("Accuracy")
plt.grid()
# sns.despine()
# plt.title("Graph Classification")
# plt.grid()
plt.savefig("./t1.svg", dpi=800)
plt.show()
