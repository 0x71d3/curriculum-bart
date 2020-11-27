import csv
import os
import sys

from sklearn import metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

data_dir, output_dir = sys.argv[1:]

targets = []
with open(os.path.join(data_dir, 'test.tsv')) as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        targets.append(int(row[-1]))

outputs = []
with open(os.path.join(output_dir, 'pred.txt')) as f:
    for line in f:
        outputs.append(int(line))

# print(metrics.accuracy_score(targets, outputs))
print(metrics.classification_report(targets, outputs, digits=4))

cm = metrics.confusion_matrix(targets, outputs)

labels = (
    ['positive', 'negative'] if data_dir == 'sst-2'
    else ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize = (10, 7))
sn.heatmap(df_cm, annot=True, cmap='Purples', fmt='g')

plt.savefig(os.path.join(output_dir, 'heatmap.png'))
