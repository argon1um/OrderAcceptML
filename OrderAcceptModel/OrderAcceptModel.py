from keras.src.utils import pad_sequences
import numpy as np
# import tensorflow_datasets as tfds
# import tensorflow as tf
# import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import pymorphy3
import inspect
import pandas as pd
import re
import nltk
import argparse
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords


nltk.download('stopwords')

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")
print(stopwords_ru)
ma = MorphAnalyzer()

def clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols  
    text = " ".join(ma.parse((word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>3)

    return text

df = pd.read_csv('Orders.csv', delimiter=';')
print(df.head())
df['Description'] = df.apply(lambda x: clean_text(x[u'OrderDescription']), axis=1)

categories = {}
for key,value in enumerate(df[u'OrderTheme'].unique()):
    categories[value] = key + 1

df['theme'] = df[u'OrderTheme'].map(categories)

total_categories = len(df[u'OrderTheme'].unique())
print('Всего тем: {}'.format(total_categories))

descriptions = df['Description']
categories = df[u'theme']

max_words = 0
for desc in descriptions:
    words = len(desc.split())
    if words > max_words:
        max_words = words
print('Максимальная длина описания: {} слов'.format(max_words))

maxSequenceLength = 55

tokenizer = Tokenizer()
tokenizer.fit_on_texts([str(value) for value in descriptions])

textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

def load_data_from_arrays(strings, labels, train_test_split=0.9):
    data_size = len(strings)
    test_size = int(data_size - round(data_size * train_test_split))
    print("Test size: {}".format(test_size))
    
    print("\nTraining set:")
    x_train = strings[test_size:]
    print("\t - x_train: {}".format(len(x_train)))
    y_train = labels[test_size:]
    print("\t - y_train: {}".format(len(y_train)))
    
    print("\nTesting set:")
    x_test = strings[:test_size]
    print("\t - x_test: {}".format(len(x_test)))
    y_test = labels[:test_size]
    print("\t - y_test: {}".format(len(y_test)))

    return x_train, y_train, x_test, y_test

X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, categories, train_test_split=0.8)
total_words = len(tokenizer.word_index)
print('В словаре {} слов'.format(total_words))


num_words = total_words

print(u'Преобразуем описания заявок в векторы чисел...')
tokenizer = Tokenizer(num_words=num_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('Размерность X_train:', X_train.shape)
print('Размерность X_test:', X_test.shape)

print(u'Преобразуем категории в матрицу двоичных чисел '
      u'(для использования categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

epochs = 100000

print(u'Собираем модель...')
model = Sequential()
model.add(Dense(512, input_shape=(num_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(total_categories))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

score = model.evaluate(X_test, y_test,
                       batch_size=32, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))

while(True):
    text = input("Введите текст для обработки: ")
    cleaned_text = clean_text(text)
    cleaned_list = cleaned_text.split()
    tokenized_text = tokenizer.texts_to_sequences(cleaned_list)
    padded_text = pad_sequences(tokenized_text, maxlen=36, padding='post')
    prediction = model.predict(padded_text)
    print("Прогнозируемая категория:", np.argmax(prediction))


# def plot_graphs(history, metric):
#   plt.plot(history.history[metric])
#   plt.plot(history.history['val_'+metric], '')
#   plt.xlabel("Epochs")
#   plt.ylabel(metric)
#   plt.legend([metric, 'val_'+metric])
  
# dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
# train_dataset, test_dataset = dataset['train'], dataset['test']

# train_dataset.element_spec

# for example, label in train_dataset.take(1):
#   print('text: ', example.numpy())
#   print('label: ', label.numpy())
  
# BUFFER_SIZE = 10000
# BATCH_SIZE = 64

# train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# for example, label in train_dataset.take(1):
#   print('texts: ', example.numpy()[:3])
#   print()
#   print('labels: ', label.numpy()[:3])
  
# VOCAB_SIZE = 1000
# encoder = tf.keras.layers.TextVectorization(
#     max_tokens=VOCAB_SIZE)
# encoder.adapt(train_dataset.map(lambda text, label: text))

# vocab = np.array(encoder.get_vocabulary())
# vocab[:20]

# encoded_example = encoder(example)[:3].numpy()
# encoded_example

# for n in range(3):
#   print("Original: ", example[n].numpy())
#   print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
#   print()
  
# model = tf.keras.Sequential([
#     encoder,
#     tf.keras.layers.Embedding(
#         input_dim=len(encoder.get_vocabulary()),
#         output_dim=64,
#         mask_zero=True),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# print([layer.supports_masking for layer in model.layers])

# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(1e-4),
#               metrics=['accuracy'])

# history = model.fit(train_dataset, epochs=3,
#                     validation_data=test_dataset,
#                     validation_steps=30)

# test_loss, test_acc = model.evaluate(test_dataset)

# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_acc)

# plt.figure(figsize=(6, 6))
# plt.subplot(1, 2, 1)
# plot_graphs(history, 'accuracy')
# plt.ylim(None, 1)
# plt.subplot(1, 2, 2)
# plot_graphs(history, 'loss')
# plt.ylim(0, None)

# sample_text = ('The movie was cool. The animation and the graphics '
#                'were out of this world. I would recommend this movie.')
# predictions = model.predict(np.array([sample_text]))
# print(predictions[0])