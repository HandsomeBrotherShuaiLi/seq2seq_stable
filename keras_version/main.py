import numpy as np
from collections import defaultdict
from keras.layers import Input,LSTM,GRU,Bidirectional
import json
class Data(object):
    def __init__(self,train_data_path,valid_data_path,batch_size,concept_data_path=None,
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
        self.cocept_data_path=concept_data_path
        train_data=open(self.train_data_path,encoding='utf-8').readlines()
        valid_data=open(self.valid_data_path,encoding='utf-8').readlines()
        self.concept_data=json.load(open(self.cocept_data_path,'r')) if concept_data_path!=None else None
        self.max_len=max_len
        self.max_sen=max_sen
        self.dict=set()
        train_data_=[]
        valid_data_=[]
        for i in train_data:
            temp=i.split('\t')
            word_len=[len(s.split(' ')) for s in temp[0].split('<eou>')]
            word_len.append(len(temp[1].split(' ')))
            if len(temp)==3 and len(temp[0].split('<eou>'))<=self.max_sen and max(word_len)<=self.max_len:
                train_data_.append(temp)
                words=temp[0].split(' ')+temp[1].split(' ')
                for j in words:
                    self.dict.add(j)
                    try:
                        expand_list=self.concept_data[j].keys()
                        for t in expand_list:
                            self.dict.add(t)
                    except:
                        pass
        for i in valid_data:
            temp=i.split('\t')
            word_len = [len(s.split(' ')) for s in temp[0].split('<eou>')]
            word_len.append(len(temp[1].split(' ')))
            if len(temp)==3 and len(temp[0].split('<eou>'))<=self.max_sen and max(word_len)<=self.max_len:
                valid_data_.append(temp)
                words=temp[0].split(' ')+temp[1].split(' ')
                for j in words:
                    self.dict.add(j)
                    try:
                        expand_list = self.concept_data[j].keys()
                        for t in expand_list:
                            self.dict.add(t)
                    except:
                        pass

        print(len(train_data),len(train_data_))
        with open('../Data/train_3.csv','w',encoding='utf-8') as f:
            f.writelines(train_data_)
        with open('../Data/valid_3.csv','w',encoding='utf-8') as f:
            f.writelines(valid_data_)
        dict_index={w:i for i,w in enumerate(self.dict)}
        with open('../Data/all_dict.json','w',encoding='utf-8') as f:
            json.dump(f,dict_index)
    def generator(self,is_valid=False):
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