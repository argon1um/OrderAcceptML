import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# �������� ������ �� CSV-�����
data = pd.read_csv('Orders.csv', delimiter=';')
labels = [0, 1]
# �������������� ������ � ������, ��������� ��� ������ �� ���� ������
# ��������, ���������� ������ �� ������� �������� � ������� ����������
X = data.drop(columns=['OrderTheme','OrderService', 'OrderSender', 'OrderDescription'])
y = data['OrderAnswer']

print("File Opened")

max_theme_lenght = data['OrderTheme'].apply(lambda x: len(x.split())).max()
max_service_lenght = data['OrderService'].apply(lambda x: len(x.split())).max()
max_sender_lenght = data['OrderSender'].apply(lambda x: len(x.split())).max()
max_description_lenght = data['OrderDescription'].apply(lambda x: len(x.split())).max()

input_theme = Input(shape=(max_theme_lenght,))
input_service = Input(shape=(max_service_lenght,))
input_description = Input(shape=(max_description_lenght,))
input_sender = Input(shape=(max_sender_lenght,))

input_data = {
    "input_theme": input_theme,
    "input_service": input_service,
    "input_description": input_description,
    "input_sender": input_sender
}

#������� ���������� ������� � �������
num_theme_vocab = data['OrderTheme'].nunique()
num_service_vocab = data['OrderService'].nunique()
num_sender_vocab = data['OrderSender'].nunique()
num_description_vocab = data['OrderDescription'].nunique()

embedding_dim = 100

# ��������� ������������� ��� ������� �����
theme_embedding = Embedding(input_dim=num_theme_vocab, output_dim=embedding_dim)(input_theme)
service_embedding = Embedding(input_dim=num_service_vocab, output_dim=embedding_dim)(input_service)
description_embedding = Embedding(input_dim=num_description_vocab, output_dim=embedding_dim)(input_description)
sender_embedding = Embedding(input_dim=num_sender_vocab, output_dim=embedding_dim)(input_sender)

#���� ��� ��������� ��������� ������(LSTM)
lstm_theme = LSTM(128)(theme_embedding)
lstm_service = LSTM(128)(service_embedding)
lstm_description = LSTM(128)(description_embedding)
lstm_sender = LSTM(128)(sender_embedding)

# ���������� ���������� LSTM �����
concatenated = Concatenate()([lstm_theme, lstm_service, lstm_description, lstm_sender])

# �������������� ���� ��� �������������
output = Dense(1, activation='sigmoid')(concatenated)

# ������� ������
model = Model(inputs=[input_theme, input_service, input_description, input_sender], outputs=output)

# ����������� ������
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#
labels_np = np.array(y)
# ������� ������
model.fit(
    {
        "input_theme": X_train['OrderTheme'].to_numpy(),
        "input_service": X_train['OrderService'].to_numpy(),
        "input_description": X_train['OrderDescription'].to_numpy(),
        "input_sender": X_train['OrderSender'].to_numpy()
    },
    y_train.to_numpy(),
    epochs=10,
    batch_size=32,
    validation_data=(
        {
            "input_theme": X_val['OrderTheme'].to_numpy(),
            "input_service": X_val['OrderService'].to_numpy(),
            "input_description": X_val['OrderDescription'].to_numpy(),
            "input_sender": X_val['OrderSender'].to_numpy()
        },
        y_val.to_numpy()
    )
)