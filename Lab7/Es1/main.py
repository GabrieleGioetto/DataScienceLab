from os import listdir
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


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

# fr = [[[[[[0, 0]]] * 4001]] * 1500]
# print(fr)

# fourier_transformation_train = pd.DataFrame(data=np.nan(shape=(1500, 4001)), columns=list(range(4001)))
fourier_transformation_train = np.array([[[0.0, 0.0]] * 4001] * 1500)
# fourier_transformation_test = pd.DataFrame(data=np.nan(shape=(500, 4001)), columns=list(range(4001)))
fourier_transformation_test = np.array([[[0.0, 0.0]] * 4001] * 500)

max_samples = 0

for i, file_name in enumerate(file_dev):
    f, data = wavfile.read(dev_file_path + "/" + file_name)
    f_b = f / 2

    N = data.shape[0]  # numero di campioni
    length = N / f  # length: durata in secondi

    # print(f"samplerate: {f}")
    # print(f"length: {length}")
    # print(f"# of data: {len(data)}")
    # print(f"data: {data}")

    if N > max_samples:
        max_samples = N

    ft = fft(data)
    amplitudes_f = 2 / N * np.abs(ft[0:N // 2])
    phase_f = np.arctan2(np.imag(ft[0:N // 2]), np.real(ft[0:N // 2]))

    frequences = np.linspace(0, f_b, N // 2)  # frequenze in Hz in cui ho calcolato i moduli
    length_frequences = len(frequences) - 1
    for j in range(length_frequences):
        fourier_transformation_train[i, round(frequences[j]), :] += [amplitudes_f[j], phase_f[j]]
    print(i)

print(f"MAX SAMPLES: {max_samples}")

for i, file_name in enumerate(file_eval):
    f, data = wavfile.read(eval_file_path + "/" + file_name)
    f_b = f / 2

    N = data.shape[0]  # numero di campioni
    length = N / f  # length: durata in secondi

    ft = fft(data)
    amplitudes_f = 2 / N * np.abs(ft[0:N // 2])
    phase_f = np.arctan2(np.imag(ft[0:N // 2]), np.real(ft[0:N // 2]))

    frequences = np.linspace(0, f_b, N // 2)  # frequenze in Hz in cui ho calcolato i moduli
    length_frequences = len(frequences)
    for j in range(length_frequences):
        fourier_transformation_train[i, round(frequences[j]), :] += [amplitudes_f[j], phase_f[j]]
    print(i)

print(len(fourier_transformation_train))

y_train = [file[-5] for file in file_dev]

clf = KNeighborsClassifier(n_neighbors=8)
clf.fit(fourier_transformation_train, y_train)

y_pred = clf.predict(fourier_transformation_test)

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
        print(df.iloc[i, 0])
        stringa = str(i) + "," + str(df.iloc[i, 0] + "\n")
        f.write(stringa)

# # phase_f = np.arctan2(np.imag(fourier_transformation[0:N // 2]), np.real(fourier_transformation[0:N // 2]))
