import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from os import listdir

dev_file_path = "./free-spoken-digit/dev"
eval_file_path = "./free-spoken-digit/eval"
file_dev = listdir(dev_file_path)
file_eval = listdir(eval_file_path)

y_train = [file[-5] for file in file_dev]  # I get the number I have to predict

fourier_transformation_train_df = pd.read_csv("train_df.csv")
fourier_transformation_test_df = pd.read_csv("test_df.csv")

fourier_transformation_train_df = fourier_transformation_train_df.fillna(0)
fourier_transformation_test_df = fourier_transformation_test_df.fillna(0)

fourier_transformation_train_df.drop(fourier_transformation_train_df.columns[[0]], axis=1, inplace=True)
fourier_transformation_test_df.drop(fourier_transformation_test_df.columns[[0]], axis=1, inplace=True)

fourier_transformation_train_df = fourier_transformation_train_df.astype("float32")
fourier_transformation_test_df = fourier_transformation_test_df.astype("float32")

fourier_transformation_train_df.reset_index()
fourier_transformation_test_df.reset_index()

col_mask = fourier_transformation_test_df.isnull().any(axis=0)
row_mask = fourier_transformation_test_df.isnull().any(axis=1)
print(fourier_transformation_test_df.loc[row_mask, col_mask])

clf = RandomForestClassifier(random_state=0)
clf.fit(fourier_transformation_train_df, y_train)

y_pred = clf.predict(fourier_transformation_test_df)

print(file_eval)
print(y_pred)

df = pd.DataFrame(data=np.zeros(500), columns=["Predicted"])

with open("myfile.csv", "w") as f:
    f.write("Id,Predicted\n")
    for file_id, label in zip(file_eval, y_pred):
        riga = int(file_id.split('.')[0])
        df.iloc[riga, [0]] = label

    print(df)

    for i in range(df.shape[0]):
        stringa = str(i) + "," + str(df.iloc[i, 0] + "\n")
        f.write(stringa)
