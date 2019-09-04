import numpy as np
from keras.layers import Input,LSTM,Dense,Embedding,Reshape,Lambda
from keras import Model
from keras.optimizers import Adam,SGD,RMSprop
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import json,time
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,TensorBoard,ModelCheckpoint

train_data=open('../Data/train_3.txt','r',encoding='utf-8').readlines()
from keras_text.processing import WordTokenizer
from keras_text.data import Dataset
tokenizer = WordTokenizer()
tokenizer.build_vocab(train_data)
x=[]
y=[]
from keras_text.models.sentence_model import SentenceModelFactory
from keras_text.models.sequence_encoders import AttentionRNN
for i,sample in enumerate(train_data):
    x.append(sample.split('\t')[0])
    y.append(sample.split('\t')[1])
ds = Dataset(x, y, tokenizer=tokenizer)
ds.update_test_indices(test_size=0.1)
ds.save('dataset')

