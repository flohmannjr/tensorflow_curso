import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import seaborn as sns

import datetime

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

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
                 ativacao='softmax'
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

def grafico_perda_precisao_por_iteracao(historico):

    for chave in historico.keys():
        sns.lineplot(data=historico.history[chave], label=chave)

    plt.title('Histórico por iteração')
    plt.xlabel('Iteração')
    plt.ylabel('')

    plt.legend(frameon=True, facecolor='white')
    plt.show()

# Create a function to import and image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224):
    """
    Reads an image from filename, turns it into a tensor and reshapes it 
    to (img_shape, img_shape, colour_channels).
    """

    # Read in the image
    img = tf.io.read_file(filename)

    # Decode the read file into a tensor
    img = tf.image.decode_image(img)

    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])

    # Rescale the image (get all values between 0 and 1)
    img = img/255.

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

