from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, LeakyReLU, Dropout, TimeDistributed
from keras.layers import SpatialDropout1D
from config import LSTM_units


def get_model_binary(vocab_size, sequence_length):
    embedding_size = 64
    model=Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_length))
    model.add(SpatialDropout1D(0.15))
    model.add(LSTM(LSTM_units, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

def get_model_5stars(vocab_size, sequence_length, embedding_size):
    model=Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_length))
    model.add(SpatialDropout1D(0.15))
    model.add(LSTM(LSTM_units, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="linear"))
    model.summary()
    return model