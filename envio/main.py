"""
main.py
Desenvolvido pela equipe Pandara:
 - Aline Guimarães de Oliveira
 - Adriano Domingos Goulart
 - Ítalo Della Garza Silva
Para o Hackathon JPA Agro 2021

23/02/2021

Execução:
Para executar a solução, crie um virtualenv na pasta do código e
execute o seguinte comando:

python3 -m venv env

Ative então sua virtualenv

source env/bin/activate

Instale as bibliotecas necessárias com o comando:

pip install notebook==6.1.6 numpy==1.19.4 pandas==1.1.4 scikit-learn==0.23.2 matplotlib==3.3.3 tensorflow==2.4.0 seaborn==0.11.1

Execute o código (é necessário ter o arquivo de dados de
treino "dataset_train.csv" no mesmo diretório do projeto.):

python main.py

"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.rnn = tf.keras.layers.LSTM(
            units = 350,
            activation = 'relu',
            return_sequences = True,
            kernel_initializer=tf.keras.initializers.GlorotNormal()
        )
        self.dense = tf.keras.layers.Dense(units = 1, kernel_initializer=tf.keras.initializers.GlorotNormal())
    
    def call(self, inputs):
        x = self.rnn(inputs)
        x = self.dense(x)
        return x


def rmse(y_predict, y_true):
    return np.sqrt(
        np.square(
            np.subtract(y_predict, y_true)
        ).mean()
    )

def mae(y_predict, y_true):
    return mean_absolute_error(y_predict, y_true)


def next_batch(data):
    # Seleciona um número aleatório dentro dos índices da base
    begin = np.random.randint(0,len(data)-31)
    # Seleciona dentro do conjunto teste o X_test
    X_train = data[begin: begin + 30].reshape(1,30,1)
    # Seleciona dentro do conjunto teste o y_test
    y_train = data[begin+1: begin + 31].reshape(1,30,1)
    return X_train, y_train


def main():

    tf.random.set_seed(42)

    # Leitura e manipulação dos dados.
    base = pd.read_csv('dataset/dataset_train.csv')
    base = base.dropna()
    scaler_price = MinMaxScaler()
    base[['sold_price']] = scaler_price.fit_transform(base[['sold_price']])
    base = base.iloc[:, 1].values

    # Divisão em treino e teste
    treino = base[:-31]
    X_teste = base[-31:-1].reshape(1,30,1)
    y_teste = base[-30:].reshape(1,30,1)

    # Treinamento e avaliação do modelo e obtenção do RMSE e MAE médios.
    
    rmses = []
    maes = []

    for i in range(10):
        print("AVALIACAO: ", i)
        model = Model()
        otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001)
        for epoca in range(1000):
            X_batches, y_batches = next_batch(treino)
            with tf.GradientTape() as tape:
                out = model(X_batches)
                erro = tf.keras.losses.mean_squared_error(out, y_batches)
            
            gradientes = tape.gradient(erro, model.trainable_variables)
            otimizador.apply_gradients(zip(gradientes, model.trainable_variables))

            if epoca % 100 == 0:
                print(epoca + 1, ' erro: ', tf.reduce_mean(erro).numpy())
        previsoes = model(X_teste)
        previsoes = np.ravel(scaler_price.inverse_transform(previsoes[0]))
        real = np.ravel(scaler_price.inverse_transform(y_teste[0]))
        print('MAE = ', mae(previsoes, real))
        print('RMSE = ', rmse(previsoes, real))
        print()
        rmses.append(rmse(previsoes, real))
        maes.append(mae(previsoes, real))

    print('MAE medio = ', np.array(maes).mean())
    print('RMSE medio = ', np.array(rmses).mean())
    print()

    # Treinamento do modelo para a predição final
    model = Model()
    otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001)
    for epoca in range(1000):
        X_batches, y_batches = next_batch(base)
        with tf.GradientTape() as tape:
            out = model(X_batches)
            erro = tf.keras.losses.mean_squared_error(out, y_batches)
        
        gradientes = tape.gradient(erro, model.trainable_variables)
        otimizador.apply_gradients(zip(gradientes, model.trainable_variables))

        if epoca % 100 == 0:
            print(epoca + 1, ' erro: ', tf.reduce_mean(erro).numpy())

    X_final = base[-30:]
    for i in range(30):
        y = np.ravel(model(X_final[i:].reshape(1,30,1)))[-1]
        X_final = np.append(X_final, y)

    y_gen = np.ravel(scaler_price.inverse_transform(X_final[-30:].reshape(1,-1))).tolist()

    string_final = ''
    for number in y_gen:
        string_final += str('{:.2f}'.format(round(number, 2))) + ';'
    string_final = string_final[:-1]
    open('predicted_values.txt', 'w').write(string_final)


if __name__ == '__main__':
    main()