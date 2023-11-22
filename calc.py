from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import pandas as pd

test_data = "datasets/laprest14/output_know_insert/test.tsv"
file_name = f"results/test_logits_test_deberta_Base.txt"

df = pd.read_csv(test_data, sep="\t")

true_data = df.sentiment.values.astype(int)
pred_data = np.empty(true_data.shape).astype(int)
reviews = df.review.values
aspects = df.aspect.values

all_true_data = df.sentiment.values.astype(int)
all_pred_data = np.empty(all_true_data.shape).astype(int)

with open(file_name, "r") as f_:
    i_line = 0
    data = f_.read()
    print("Total test samples: ", len(data.split('\n')) - 1)
    for i, line in enumerate(data.split("\n")):
        if len(line) < 1:
            continue

        pred = [float(x) for x in line.split('\t')]

        all_pred_data[i_line] = np.argmax(pred) - 1
        i_line += 1

print("Confusion Matrix: ")

cf_matrix = confusion_matrix(all_true_data, all_pred_data)
acc_score = accuracy_score(all_true_data, all_pred_data)
f_score = f1_score(all_true_data, all_pred_data, average="macro")

print(cf_matrix)
print(f"Accuracy:\t\t{acc_score}")
print(f"F_Score:\t{f_score}")

incorrect = 0
for i in range(len(data.split('\n')) - 1):
    if all_true_data[i] != all_pred_data[i]:
        incorrect += 1
        print('True: ', all_true_data[i], 'Predicted: ', all_pred_data[i], '#', reviews[i], '#', aspects[i])
    i += 1

print('Total test samples: ', i, 'Incorrect: ', incorrect)
print("Confusion Matrix: ")
print(cf_matrix)
print(f"Accuracy:\t\t{acc_score}")
print(f"F_Score:\t{f_score}")