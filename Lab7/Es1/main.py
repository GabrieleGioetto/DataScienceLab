from os import listdir
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys


def show_audio_plot(samplerate, data):
    length = data.shape[0] / samplerate

    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


dev_file_path = "./free-spoken-digit/dev"
eval_file_path = "./free-spoken-digit/eval"
file_dev = listdir(dev_file_path)
file_eval = listdir(eval_file_path)

fourier_transformation_train = np.empty((1500, 4001))
fourier_transformation_test = np.empty((500, 4001))

fourier_transformation_train.fill(np.nan)
fourier_transformation_test.fill(np.nan)

max_samples = 0
min_samples = sys.maxsize
min_samples_file = ""

for i, file_name in enumerate(file_dev):
    f, data = wavfile.read(dev_file_path + "/" + file_name)
    f_b = f / 2

    # frequencies, times, spectrogram = signal.spectrogram(x=data, fs=f)

    data_avg = np.mean(np.abs(data))
    data_without_silence = []
    for j in range(1, len(data) - 1):
        if np.abs(np.mean(data[[j - 1, j, j + 1]])) > 0.2 * data_avg:
            data_without_silence.append(data[j])

    data_without_silence = np.array(data_without_silence)

    N = data_without_silence.shape[0]  # numero di campioni
    length = N / f  # length: durata in secondi

    if N > max_samples:
        max_samples = N
    if N < min_samples:
        min_samples = N
    min_samples_file = file_name

    ft = fft(data_without_silence)

    amplitudes_f = 2 / N * np.abs(ft[0:N // 2])
    # phase_f = np.arctan2(np.imag(ft[0:N // 2]), np.real(ft[0:N // 2]))

    frequences = np.linspace(0, f_b, N // 2)  # frequenze in Hz in cui ho calcolato i moduli
    length_frequences = len(frequences) - 1
    for j in range(length_frequences):
        if amplitudes_f[j] > 0:
            fourier_transformation_train[i, round(frequences[j])] = amplitudes_f[j]

    print(f"i:{i}, file_name:{file_name}")

    # if (int(file_name[-5]) == 9):
    #     plt.plot(frequences, 2.0 / N * np.abs(ft[0:N // 2]), label="Numero: " + file_name[-5])
    #     plt.legend(loc="upper right")
    #
    #     plt.show()

print(f"MAX SAMPLES: {max_samples}")
print(f"MIN SAMPLES: {min_samples}")
print(f"MIN SAMPLES file: {min_samples_file}")

for i, file_name in enumerate(file_eval):
    f, data = wavfile.read(eval_file_path + "/" + file_name)
    f_b = f / 2

    data_avg = np.mean(np.abs(data))
    data_without_silence = []
    for j in range(1, len(data) - 1):
        if np.abs(np.mean(data[[j - 1, j, j + 1]])) > 0.2 * data_avg:
            data_without_silence.append(data[j])

    data_without_silence = np.array(data_without_silence)

    N = data_without_silence.shape[0]  # numero di campioni
    length = N / f  # length: durata in secondi

    ft = fft(data_without_silence)
    amplitudes_f = 2.0 / N * np.abs(ft[0:N // 2])
    # phase_f = np.arctan2(np.imag(ft[0:N // 2]), np.real(ft[0:N // 2]))

    frequences = np.linspace(0, f_b, N // 2)  # frequenze in Hz in cui ho calcolato i moduli
    length_frequences = len(frequences)
    for j in range(length_frequences):
        if amplitudes_f[j] > 0:
            fourier_transformation_test[i, round(frequences[j])] = amplitudes_f[j]

    print(f"i:{i}, file_name:{file_name}")

y_train = [file[-5] for file in file_dev]  # I get the number I have to predict

fourier_transformation_train_df = pd.DataFrame(fourier_transformation_train).ffill(axis=1)
fourier_transformation_test_df = pd.DataFrame(fourier_transformation_test).ffill(axis=1)

fourier_transformation_train_df = fourier_transformation_train_df.fillna(0)
fourier_transformation_test_df = fourier_transformation_test_df.fillna(0)

fourier_transformation_train_df = fourier_transformation_train_df.astype("float32")
fourier_transformation_test_df = fourier_transformation_test_df.astype("float32")

fourier_transformation_train_df.to_csv("train_df.csv")
fourier_transformation_test_df.to_csv("test_df.csv")

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
