# Dê upload do dataset no formato .zip e utilize a trecho de código comentado abaixo para descompacta-lo

# import zipfile
# import os

# # Caminho do arquivo zip
# caminho_arquivo_zip = '/content/dataset.zip'

# # Pasta de destino para extração
# pasta_destino = '/content'  # Aqui é onde a pasta será extraída, atualizada com o caminho correto

# # Extrair o arquivo zip
# with zipfile.ZipFile(caminho_arquivo_zip, 'r') as zip_ref:
#     zip_ref.extractall(pasta_destino)

# # Listar os arquivos extraídos
# arquivos_extraidos = os.listdir(pasta_destino)
# print("Arquivos extraídos:")
# print(arquivos_extraidos)

import os

# from google.colab import drive

import random
import numpy as np
import keras

import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping

train_dir = '/content/dataset/'
validation_dir = '/content/dataset/'
test_dir = '/content/dataset/'

root = '/content/dataset/'
exclude = ['.ipynb_checkpoints']
train_split, val_split = 0.7, 0.15

categories = [x[0] for x in os.walk(root) if x[0]][1:]
categories = [c for c in categories if c not in [os.path.join(root, e) for e in exclude]]

print(categories)

def get_image(path):
  img = image.load_img(path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  return img, x

data = []
for c, category in enumerate(categories):
  images = [os.path.join(dp, f) for dp, dn, filenames
            in os.walk(category) for f in filenames
            if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
  for img_path in images:
    img, x = get_image(img_path)
    data.append({'x':np.array(x[0]), 'y':c})

num_classes = len(categories)

random.shuffle(data)

idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))
train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

x_train, y_train = np.array([t['x'] for t in train]), [t['y'] for t in train]
x_val, y_val = np.array([t['x'] for t in val]), [t['y'] for t in val]
x_test, y_test = np.array([t['x'] for t in test]), [t['y'] for t in test]
print(y_test)

# normalizar os dados
x_train = x_train.astype('float32') / 255
x_val = x_val.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# corverter labels para one-hot vertores
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test.shape)

datagen = ImageDataGenerator(
    rotation_range=20,  # Variação de até 20 graus na rotação
    width_shift_range=0.1,  # Variação de até 10% na largura
    height_shift_range=0.1,  # Variação de até 10% na altura
    zoom_range=0.2,  # Zoom de até 20%
    horizontal_flip=True,  # Inversão horizontal
    brightness_range=[0.5, 1.5]  # Variação de brilho entre 0.5 e 1.5
)

# Aplique o aumento de dados às suas imagens
# Supondo que 'X_train' seja um array numpy contendo suas imagens
datagen.fit(x_train)

train_generator = datagen.flow(x_train, y_train, batch_size=128)

val_datagen = ImageDataGenerator()

val_generator = val_datagen.flow(x_val, y_val, batch_size=128)

# sumário
print(f'finalizado o carregamento de {len(data)} imagens de {num_classes} categorias')
print(f'treino: {len(x_train)}, validação: {len(x_val)}, teste: {len(x_test)}')
print(f'data shape de treinamento: {x_train.shape}')
print(f'labels shape de treinamento: {y_train.shape}')

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) for f
          in filenames if os.path.splitext(f)[1].lower() in ['.jpg', '.png', 'jpeg']]
idx = [int(len(images) * random.random()) for i in range(8)]
imgs = [image.load_img(images[i], target_size=(224, 224)) for i in idx]
concat_image = np.concatenate([np.asarray(img) for img in imgs], axis = 1)
plt.figure(figsize = (16,4))
plt.imshow(concat_image)

# construindo a network
model = Sequential()
print(f'Dimensões de entrada: {x_train.shape[1:]}')

model.add(Conv2D(32, (3, 3), input_shape = x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# compile the model to use cateforical cross-entropy loss function and adadelta optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=20,
                    validation_data=val_generator,
                    callbacks=[early_stopping])

fig = plt.figure(figsize = (16, 4))
ax = fig.add_subplot(121)
ax.plot(history.history['val_loss'])
ax.set_title('perca na validação')
ax.set_xlabel('épocas')

ax2 = fig.add_subplot(122)
ax2.plot(history.history['val_accuracy'])
ax2.set_title('acurácia da validação')
ax2.set_xlabel('épocas')
ax2.set_ylim(0, 1)

plt.show()

loss, accuracy = model.evaluate(x_test, y_test, verbose = 0)
print('Perca do teste: ', loss)
print('Acurácia do teste: ', accuracy)

vgg = keras.applications.VGG16(weights='imagenet', include_top = True)
vgg.summary()

# faz uma referência a camada de input do model VGG
inp = vgg.input

# faz uma nova camada softmax com os neurônios de num_classes
new_classification_layer = Dense(num_classes, activation='softmax')

# conecta a nova camada as camadas da segunda em diante do model VGG, fazendo referência a ela
out = new_classification_layer(vgg.layers[-2].output)

# cria uma nova network entre inp e out
model_new = Model(inp, out)

# tornar todas as camadas não treináveis e congelar seus pesos (exceto pela última)
for l, layer in enumerate(model_new.layers[:-1]):
  layer.trainable = False

# garantir que a última camada é treinável (não está congelada)
for l, layer in enumerate(model_new.layers[-1:]):
  layer.trainable = True

model_new.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model_new.summary()

history_transfer = model_new.fit(train_generator,
                    epochs=20,
                    validation_data=val_generator,
                    callbacks=[early_stopping])

fig = plt.figure(figsize = (16, 4))
ax = fig.add_subplot(121)
ax.plot(history.history['val_loss'])
ax.plot(history_transfer.history['val_loss'])
ax.set_title('perca na validação')
ax.set_xlabel('épocas')

ax2 = fig.add_subplot(122)
ax2.plot(history.history['val_accuracy'])
ax2.plot(history_transfer.history['val_accuracy'])
ax2.set_title('acurácia da validação')
ax2.set_xlabel('épocas')
ax2.set_ylim(0, 1)

plt.show()

loss, accuracy = model_new.evaluate(x_test, y_test, verbose = 0)

print('Perca do teste: ', loss)
print('Acurácia do teste: ', accuracy)

print(categories)

img, x = get_image('/content/dataset/cards-pokemon/71B6ErTUBzL._AC_UF894,1000_QL80_.jpg')
probabilities = model_new.predict([x])

for nome_classe, probabilidade in zip(categories, probabilities[0]):
    print(f'Probabilidade de pertencer à classe {nome_classe}: {probabilidade * 100:.2f}%')