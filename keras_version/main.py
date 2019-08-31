import numpy as np
from collections import defaultdict
from keras.layers import Input,LSTM,GRU,Bidirectional,Lambda,Embedding,Permute,Reshape
from keras.preprocessing.text import Tokenizer
from keras import Model
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

import json
class Data(object):
    def __init__(self,train_data_path='../Data/train_3.txt',
                 valid_data_path='../Data/valid_3.txt',
                 batch_size=64,concept_data_path=None,
                 dict_path='../Data/all_dict.json',
                 max_len=250,max_sen=11):
        """
        输出 batch_size*
        :param train_data_path:
        :param valid_data_path:
        :param batch_size:
        """
        self.train_data_path=train_data_path
        self.valid_data_path=valid_data_path
        self.batch_size=batch_size
        self.dict_path=dict_path
        self.cocept_data_path=concept_data_path
        self.train_data=open(self.train_data_path,encoding='utf-8').readlines()
        self.valid_data=open(self.valid_data_path,encoding='utf-8').readlines()
        self.concept_data=json.load(open(self.cocept_data_path,'r')) if concept_data_path!=None else None
        self.max_len=max_len
        self.max_sen=max_sen
        self.dict=json.load(open(self.dict_path,'r',encoding='utf-8'))
        replace_key=list(self.dict.keys())[0]
        self.dict['<pad>']=0
        self.dict['<eou>'] = len(self.dict.keys())
        self.dict[replace_key] = len(self.dict.keys())
        self.train_index=np.array(range(len(self.train_data)))
        self.valid_index=np.array(range(len(self.valid_data)))
        np.random.shuffle(self.train_index)
        np.random.shuffle(self.valid_index)
        self.steps_per_epoch=len(self.train_data)//self.batch_size
        self.valid_steps_per_epoch=len(self.valid_data)//self.batch_size
    def generator(self,is_valid=False,use_concept=False):
        print('start generating...')
        data=self.train_data if is_valid==False else self.valid_data
        data=np.array(data)
        index=self.valid_index if is_valid else self.train_index
        if use_concept==False:
            encoder_input=[]
            decoder_input=[]
            id=0
            while True:
                if id+self.batch_size<data.shape[0]:
                    samples=data[index[id:id+self.batch_size]]
                else:
                    temp_index=index[id:]+index[:(id+self.batch_size)%len(data)]
                    samples=data[temp_index]
                utt_all = []
                utt_len=[]
                for sample in samples:
                    temp=sample.split('\t')
                    r=temp[1].split(' ')
                    response=[self.dict[i] for i in r]
                    decoder_input.append(response)

                    utt_lines=temp[0].split('<eou>')
                    utt_id=[]
                    for i in range(self.max_sen):
                        if i <len(utt_lines):
                            id_line=[self.dict[j] for j in utt_lines[i].split(' ')]
                            utt_id.append(id_line)
                            utt_len.append(len(id_line))
                        else:
                            utt_id.append([0])
                            utt_len.append(1)

                    utt_all.append(utt_id)

                # for i in range(self.max_sen):
                #     line_sample=[utt_all[j][i] for j in range(self.batch_size)]
                #     line_sample=pad_sequences(line_sample,maxlen=max(utt_len))
                #     encoder_input.append(line_sample)
                for sample in utt_all:
                    sample=pad_sequences(sample,maxlen=max(utt_len))
                    encoder_input.append(sample)
                decoder_input=pad_sequences(decoder_input)
                """
                encoder是固定max_sen行，不定长句子的list,hhhh
                """
                encoder_input=np.array(encoder_input)
                decoder_input=np.array(decoder_input)
                print(encoder_input.shape,decoder_input.shape)

                yield {'encoder_input':encoder_input,'decoder_input':decoder_input}
                id = (id + self.batch_size) % (len(data))
                encoder_input = []
                decoder_input = []

        else:
            pass

class seq2seq(object):
    def __init__(self,hidden,max_vocab_len=600000,drop_out=0.1,max_sentence_len=250,max_sen=11,batch_size=64):
        self.max_vocab_len=max_vocab_len
        self.max_sentence_len=max_sentence_len
        self.drop_out=drop_out
        self.max_sen=max_sen
        self.hidden=hidden
        self.batch_size=batch_size
    def reshape(self,x):
        return K.reshape(x,shape=(self.batch_size,self.max_sen,self.hidden))

    def build_network(self):
        input_encoder=Input(name='encoder_input',batch_shape=(self.batch_size,self.max_sen,None))
        embedding_layer=Embedding(self.max_vocab_len, self.hidden, input_length=None, batch_size=self.batch_size)
        lstm_layer = LSTM(self.hidden, return_sequences=True,return_state=True,dropout=self.drop_out)

        input_shape=input_encoder.get_shape()
        print('input shape:{}'.format(input_shape))
        every_line_last_hidden_state=[]
        every_line_hidden_state=[]
        for line in range(int(input_shape[1])):
            oneline=embedding_layer(input_encoder[:,line,:])
            oneline,state_h, state_c=lstm_layer(oneline)
            every_line_last_hidden_state.append(state_h)
            every_line_hidden_state.append(K.concatenate(oneline,axis=-1))
        x=K.concatenate(every_line_last_hidden_state,axis=-1)
        print(x.get_shape())



        # x=K.concatenate(x,axis=-1)
        # x=Lambda(function=self.reshape)(x)
        # print(x.get_shape())
        # dialog=LSTM(self.hidden,dropout=self.drop_out)(x)
        # print('dialog shape:',dialog.get_shape())
        #
        # response_input=Input(name='decoder_input',batch_shape=(self.batch_size,None))




        # print(x.get_shape())


if __name__=='__main__':
    computer=1
    if computer==1:
        d = Data()
        t=d.generator()
        t.__next__()
        t.__next__()
        model=seq2seq(hidden=128)
        model.build_network()
    else:
        d = Data(
            train_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/train_2.tsv',
            valid_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/valid_2.tsv',
            batch_size=64,
            concept_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/concept_dict.json'
        )
        t=d.generator()



