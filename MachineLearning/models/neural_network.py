# Importando bibliotecas
from re import S
import pandas as pd
import numpy as np
import tensorflow as tf

class DenseNeuralNetwork():

    # Inicia a Rede Neural a partir de um shape de input e da quantidade de classes do problema
    def __init__(self, input_shape, n_classes):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=input_shape, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])

    # Compila a rede neural
    def compile(self, optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy']):

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    # Fita o modelo conforme X_train e y_train
    def fit(self, X_train, y_train, batch_size=8, epochs=10):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    
    # Retorna as predições da rede
    def predict(self, X_test, style='sparse'):
        pred = self.model.predict(X_test)
        return [np.argmax(v) for v in pred]