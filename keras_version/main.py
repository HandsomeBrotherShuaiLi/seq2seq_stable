import numpy as np
from collections import defaultdict
from keras.layers import Input,LSTM,GRU,Bidirectional,Lambda,Embedding
from keras.preprocessing.text import Tokenizer
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
        self.dict['<pad>']=len(self.dict.keys())
        self.dict['<eou>'] = len(self.dict.keys())
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
                        try:
                            utt_id.append([self.dict[j] for j in utt_lines[i]])
                        except:
                            utt_id.append([self.dict['<pad>']])
                        utt_len.append(len(utt_id[-1]))
                    utt_all.append(utt_id)

                for i in range(self.max_sen):
                    line_sample=[utt_all[j][i] for j in range(self.batch_size)]
                    line_sample=pad_sequences(line_sample,maxlen=max(utt_len))
                    encoder_input.append(line_sample)
                id=(id+self.batch_size)%(len(data))
                """
                encoder是固定max_sen行，不定长句子的list
                """
                print(len(encoder_input),len(encoder_input[0]),print(len(encoder_input[0][0]),len(encoder_input[0][1])))
                yield encoder_input,decoder_input

        else:
            pass

class seq2seq(object):
    def __init__(self,hidden,max_vocab_len,drop_out,max_sentence_len=250,max_sen=11,batch_size=12):
        self.max_vocab_len=max_vocab_len
        self.max_sentence_len=max_sentence_len
        self.drop_out=drop_out
        self.max_sen=max_sen
        self.hidden=hidden
        self.batch_size=batch_size
    def encoder_input(self,x):
        embedding_layer=Embedding(self.max_vocab_len,self.hidden,input_length=None,batch_size=self.batch_size)
        lstm_layer=Bidirectional(LSTM(self.hidden,return_sequences=True,return_state=True,dropout=),merge_mode='sum')
        shape=x.get_shape()
        for line in shape[0]:
            line_tensor=x[line,:,:]
            emb=embedding_layer(line_tensor)
            lstm,sequences,states=lstm_layer(emb)





    def build_network(self):
        encoder_input=Lambda()
        encoder_input=Input(shape=(None,self.max_vocab_len),name='encoder_input')
        encoder_output,state_h,state_c=LSTM(self.hidden,return_state=True)(encoder_input)
        decoder_input=Input(shape=(None,self.max_vocab_len),name='decoder_input')
        decoder_output,_,_=LSTM(self.hidden,return_sequences=True,return_state=True)

if __name__=='__main__':
    computer=1
    if computer==1:
        d = Data(
            train_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/train_2.tsv',
            valid_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/valid_2.tsv',
            batch_size=64,
            concept_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/concept_dict.json'
        )
        d.generator().__next__()
    else:
        d = Data(
            train_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/train_2.tsv',
            valid_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/valid_2.tsv',
            batch_size=64,
            concept_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/concept_dict.json'
        )
        d.generator().__next__()
