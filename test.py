import os
# disable keras loggings
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
# to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.ConfigProto(intra_op_parallelism_threads=5,

                        inter_op_parallelism_threads=5, 

                        allow_soft_placement=True,

                        device_count = {'CPU' : 1,

                                        'GPU' : 0}

                       )

from model import get_model_5stars
from utils import clean_text, tokenize_words
from config import embedding_size, sequence_length
from keras.preprocessing.sequence import pad_sequences

import pickle

vocab2int = pickle.load(open("data/vocab2int.pickle", "rb"))
model = get_model_5stars(len(vocab2int), sequence_length=sequence_length, embedding_size=embedding_size)

model.load_weights("results/model_V20_0.38_0.80.h5")


if __name__ == "__main__":
    import argparse
    try:
        parser = argparse.ArgumentParser(description="Food Review evaluator")
        parser.add_argument("review", type=str, help="The review of the product in text")
        args = parser.parse_args()

        review = tokenize_words(clean_text(args.review), vocab2int)
        x = pad_sequences([review], maxlen=sequence_length)

        print(f"{model.predict(x)[0][0]:.2f}/5")
    except Exception as e:
        print(e)
        print("this script was run without feeing any arguements")

def spitter(inputstring):
    review = tokenize_words(clean_text(inputstring), vocab2int)
    x = pad_sequences([review], maxlen=sequence_length)
    return model.predict(x)[0][0]