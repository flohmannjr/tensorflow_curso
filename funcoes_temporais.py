import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from tensorflow.keras.callbacks import ModelCheckpoint

COR_TREINO   = '#663399'
COR_PREVISAO = '#f22424'
COR_TESTE    = '#345cd3'

LINHA_ESPESSURA = 1

def grafico_series(X_treino=[], y_treino=[],
                   X_teste=[], y_teste=[],
                   X_previsao=[], y_previsao=[],
                   inicio=0, fim=None):

    if len(X_treino) > 0:
        sns.lineplot(x=X_treino[inicio:fim], y=y_treino[inicio:fim], color=COR_TREINO, linewidth=LINHA_ESPESSURA, label='Treino')

    if len(X_teste) > 0:
        sns.lineplot(x=X_teste[inicio:fim], y=y_teste[inicio:fim], color=COR_TESTE, linewidth=LINHA_ESPESSURA, label='Teste')

    if len(X_previsao) > 0:
        sns.lineplot(x=X_previsao[inicio:fim], y=y_previsao[inicio:fim], color=COR_PREVISAO, linewidth=LINHA_ESPESSURA, label='Previsão')

    plt.title('Fechamentos')
    plt.xlabel('')
    plt.ylabel('Valor em dólares')

    plt.xticks(rotation=-45)

    plt.legend(loc=(1.03, 0.88), frameon=True, facecolor='white')

    plt.show()

def grafico_metrica(metrica, titulo=None):

    metrica = pd.DataFrame(metrica).T

    sns.barplot(data=metrica, color=COR_TREINO)

    plt.title(titulo)
    plt.xlabel('Modelo')
    plt.ylabel('')

    plt.show()

def mean_absolute_scaled_error(y_teste, y_previsao):

    # Mean absolute error
    mae = np.mean(np.absolute(np.subtract(y_teste, y_previsao)))

    # Mean absolute error de previsão ingênua (Sem período.)
    mae_ingenuo = np.mean(np.absolute(np.subtract(y_teste[1:], y_teste[:-1])))

    return np.divide(mae, mae_ingenuo)

def metricas_modelo(y_teste, y_previsao):

    mae = mean_absolute_error(y_teste, y_previsao)
    rmse = np.sqrt(mean_squared_error(y_teste, y_previsao))
    mape = mean_absolute_percentage_error(y_teste, y_previsao)
    mase = mean_absolute_scaled_error(y_teste, y_previsao)

    return {'Mean Absolute Error': mae,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Percentage Error': mape,
            'Mean Absolute Scaled Error': mase}

def criar_janelas(dados, janela_tamanho, horizonte_tamanho, premios=[]):

    # Array 2D de 0 a janela_tamanho + horizonte_tamanho.
    janela_primaria = np.expand_dims(np.arange(janela_tamanho + horizonte_tamanho), axis=0)

    # Array 2D com todas as janelas completas com os índices dos dados.
    indices = janela_primaria + np.expand_dims(np.arange(len(dados) - (janela_tamanho + horizonte_tamanho - 1)), axis=0).T

    # Dados em formato de janelas com horizontes.
    janelas_horizontes = dados[indices]

    # Separa os dados em janelas, horizonte.
    if len(premios) == 0:
        janelas = janelas_horizontes[:, :-horizonte_tamanho]
    else:
        janelas = np.column_stack((janelas_horizontes[:, :-horizonte_tamanho], premios[indices[:, -(horizonte_tamanho + 1)]]))

    horizontes = janelas_horizontes[:, -horizonte_tamanho:]

    return janelas, horizontes

def separar_janelas_treino_teste(janelas, horizontes, tamanho_teste=0.2):

    quantidade_teste = int(len(janelas) * (1 - tamanho_teste))

    janelas_treino    = janelas[:quantidade_teste]
    janelas_teste     = janelas[quantidade_teste:]
    horizontes_treino = horizontes[:quantidade_teste]
    horizontes_teste  = horizontes[quantidade_teste:]

    return janelas_treino, janelas_teste, horizontes_treino, horizontes_teste

def criar_marco_modelo(modelo_nome, caminho='marcos'):

    return ModelCheckpoint(filepath=os.path.join(caminho, modelo_nome),
                           monitor='val_loss',
                           save_best_only=True,
                           verbose=0)
