import numpy as np
from collections import defaultdict
from keras.layers import Input,LSTM,GRU,Bidirectional
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
        self.train_index=np.array(range(len(self.train_data)))
        self.valid_index=np.array(range(len(self.valid_data)))
        np.random.shuffle(self.train_index)
        np.random.shuffle(self.valid_index)
        self.steps_per_epoch=len(self.train_data)//self.batch_size
        self.valid_steps_per_epoch=len(self.valid_data)//self.batch_size
    def generator(self,is_valid=False,use_concept=False):
        data=self.train_data if is_valid==False else self.valid_data
        index=self.valid_index if is_valid else self.train_index
        if use_concept==False:
            encoder_input=[]
            decoder_input=[]
            count=0
            while True:
                for id in index:

        else:
            pass

class seq2seq(object):
    def __init__(self,max_vocab_len,hidden):
        self.max_vocab_len=max_vocab_len
        self.hidden=hidden
    def build_network(self):
        encoder_input=Input(shape=(None,self.max_vocab_len),name='encoder_input')
        encoder_output,state_h,state_c=LSTM(self.hidden,return_state=True)(encoder_input)
        decoder_input=Input(shape=(None,self.max_vocab_len),name='decoder_input')
        decoder_output,_,_=LSTM(self.hidden,return_sequences=True,return_state=True)




if __name__=='__main__':

    d=Data(
        train_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/train_2.tsv',
        valid_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/valid_2.tsv',
        batch_size=8,
        concept_data_path='/home/next/PycharmProjects/zhoujianyun/ConceptNet/concept_dict.json'
    )