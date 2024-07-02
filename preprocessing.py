import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def load_data(file_path):
    data = pd.read_csv(file_path)
    data.replace({'GALAXY': 0, 'STAR': 1, 'QUASAR': 2}, inplace=True)
    return data

def remove_outliers(data):
    data_normed = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    mean, stdev = np.mean(data_normed, axis=0), np.std(data_normed, axis=0)
    outliers = ~((np.abs(data_normed - mean) > stdev).any(axis=1))
    return data[outliers]

def create_datasets(data, train_size, validation_size, test_size):
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    train_set = data.iloc[:train_size]
    validation_set = data.iloc[train_size:train_size+validation_size]
    test_set = data.iloc[-test_size:]
    return train_set, validation_set, test_set

def convert_one_hot(label_array):
    label_ohot = np.zeros((label_array.shape[0], 3))
    for i in range(label_array.shape[0]):
        label_ohot[i][int(label_array[i])] = 1
    return label_ohot

start_time = time.time()

# Carregar os dados
file_path = 'star_classification.csv'
data = load_data(file_path)

# Remover outliers
no_outliers = remove_outliers(data)

# Mostrar a frequência de cada classe
class_counts = no_outliers['class'].value_counts().sort_index()
classnames = ['Galaxy', 'Star', 'Quasar']
counts = class_counts.values
fig = px.bar(x=classnames, y=counts, text=counts, labels={'x': 'Classes', 'y': 'Total'},
             title='Frequência com que cada classe aparece nos dados', opacity=0.4)
fig.update_traces(marker_color='blue')
fig.update_layout(xaxis_title='Classes', yaxis_title='Total')
fig.update_yaxes(tickformat="000")
fig.show()

# Criar os conjuntos de treino, validação e teste
train_set, validation_set, test_set = create_datasets(no_outliers, train_size=6000, validation_size=3000, test_size=3000)

# Salvar os conjuntos de dados
np.save("train_set.npy", train_set.to_numpy())
np.save("test_set.npy", test_set.to_numpy())
np.save("validation_set.npy", validation_set.to_numpy())

# Calcular a matriz de correlação
dataset_label = convert_one_hot(no_outliers['class'].to_numpy() * 2)
dataset_feat = no_outliers.drop('class', axis=1).to_numpy()
all_data = np.hstack((dataset_feat, dataset_label))
df = pd.DataFrame(all_data, columns=["alpha", "delta", "u", "g", "r", "i", "z", "field_ID", "redshift", "Galaxy", "Star", "Quasar"])
plt.figure()
sns.heatmap(df.corr().round(3), annot=True, cmap='coolwarm', annot_kws={"fontsize":8})
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
