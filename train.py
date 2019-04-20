# to use CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
                    #    )

import os
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence

from preprocess import load_review_data
from model import get_model_5stars
from config import sequence_length, embedding_size, batch_size, epochs

X_train, X_test, y_train, y_test, vocab = load_review_data()

vocab_size = len(vocab)

print("Vocab size:", vocab_size)

X_train = sequence.pad_sequences(X_train, maxlen=sequence_length)
X_test = sequence.pad_sequences(X_test, maxlen=sequence_length)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

model = get_model_5stars(vocab_size, sequence_length=sequence_length, embedding_size=embedding_size)
model.load_weights("results/model_V20_0.38_0.80.h5")
model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])

if not os.path.isdir("results"):
    os.mkdir("results")

checkpointer = ModelCheckpoint("results/model_V20_{val_loss:.2f}_{val_acc:.2f}.h5", save_best_only=True, verbose=1)

model.fit(X_train, y_train, epochs=epochs,
          validation_data=(X_test, y_test),
          batch_size=batch_size,
          callbacks=[checkpointer])

