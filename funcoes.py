import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import zipfile
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Função para criar callback TensorBoard
def criar_callback_tensorboard(diretorio, experimento):
    # Diretório do TensorBoard
    dir_log = diretorio + '/' + experimento + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Criando o callback
    cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=dir_log)

    print(f"Salvando log TensorBoard em: '{dir_log}'.")

    return cb_tensorboard

def criar_modelo(modelo_url, quantidade_classes,
                 formato_entrada=(224, 224),
                 ativacao='softmax',
                 treinavel=False):
    """
    Cria um modelo sequencial Keras.

    Args:
        modelo_url (str): URL do modelo no TensorFlow Hub.
        quantidade_classes (int): Quantidade de neurônios na camada de saída (classes).
        formato_entrada (int, int): Formato da imagem.
        ativacao (str): Função de ativação da camada de saída.
        treinavel (bool): Informação se o modelo será treinável.
    
    Returns:
        Um modelo sequencial Keras não-compilado, criado com os argumentos informados.
    """

    modelo = Sequential()

    modelo.add(hub.KerasLayer(handle=modelo_url,
                              trainable=treinavel,
                              input_shape=formato_entrada))

    modelo.add(Dense(quantidade_classes, activation=ativacao))

    return modelo

def grafico_historico_por_iteracao(historico):

    for chave in historico.history.keys():
        sns.lineplot(data=historico.history[chave], label=chave)

    plt.title('Histórico por iteração')
    plt.xlabel('Iteração')
    plt.ylabel('')

    plt.legend(frameon=True, facecolor='white')
    plt.show()

def grafico_historicos_complementares(original, complementar):
    """
    Cria e apresenta um gráfico de linhas (Seaborn) contendo a evolução dos dados contidos nos históricos.
    Os históricos devem ser complementares. (Ter a mesma estrutura.)

    Args:
        original (obj History): Histórico com iterações originais.
        complementar (obj History): Histórico com iterações complementares.
    """

    if original.history.keys() != complementar.history.keys():
        print('Os históricos não são complementares.')
        return

    for chave in original.history.keys():
        sns.lineplot(data=(original.history[chave] + complementar.history[chave]), label=chave)
    
    plt.axvline(x=original.epoch[-1], color='black')

    plt.title('Históricos complementares por iteração')
    plt.xlabel('Iteração')
    plt.ylabel('')

    plt.legend(frameon=True, facecolor='white')
    plt.show()

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img/255.
    else:
        return img

def pred_and_plot(model, filename, class_names):
    """
    Imports an image located at filename, makes a prediction with model
    and plots the image with the predicted class as the title.
    """

    # Import the target image and preprocess it
    img = load_and_prep_image(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1: # Multi-class
        pred_class = class_names[pred.argmax()]
        pred_perc = pred.max()
    else: # Binary
        pred_class = class_names[int(tf.round(pred))]
        pred_perc = tf.squeeze(pred if pred >= 0.5 else 1 - pred)

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class} ({(pred_perc * 100):0.2f}%)")
    plt.axis(False)
    plt.show()

def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def avaliar_modelo(validacao, previsao):
    print(classification_report(validacao, previsao))

    ConfusionMatrixDisplay.from_predictions(validacao, previsao, cmap='summer_r')
    plt.grid(False)

    relatorio = classification_report(validacao, previsao, output_dict=True)

    return {'acuracia': relatorio['accuracy'],
            'precisao': relatorio['weighted avg']['precision'],
            'revocacao': relatorio['weighted avg']['recall'],
            'pontuacao-f1': relatorio['weighted avg']['f1-score']}
