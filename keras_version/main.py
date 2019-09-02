import numpy as np
from keras.layers import Input,LSTM,Dense,Embedding,Reshape,Lambda
from keras import Model
from keras.optimizers import Adam,SGD,RMSprop
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
import json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,TensorBoard,ModelCheckpoint
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T
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
class Data_2(object):
    def __init__(self,train_data_path,batch_size,split_ratio):
        self.train_data=open(train_data_path,'r',encoding='utf-8').readlines()
        train_index=np.array(range(len(self.train_data)))
        valid_num=int(split_ratio*len(self.train_data))
        self.valid_index=np.random.choice(train_index,size=valid_num,replace=False)
        self.train_index=np.array([i for i in train_index if i not in self.valid_index])
        self.steps_per_epoch=len(self.train_index)//batch_size
        self.valid_steps_per_epoch=len(self.valid_index)//batch_size
        self.batch_size=batch_size
        input_len=[]
        target_len=[]
        dict=set()
        dict.add('<pad>')
        dict.add('<eou>')
        each_sample_context_maxlen_list=[]#每个样本上下文任意一句最大的单词个数
        each_sample_context_sentence_num=[]#每个样本多少行上下文
        for sample in self.train_data:
            temp=sample.split('\t')
            maxlen=0
            each_sample_context_sentence_num.append(len(temp[0].split('<eou>')))
            for sentence in temp[0].split('<eou>'):
                templen=len(sentence.split(' '))
                if templen>maxlen:
                    maxlen=templen
            each_sample_context_maxlen_list.append(maxlen)

            input_len.append(len(temp[0].split(' ')))
            target_len.append(len(temp[1].split(' ')))
            for j in temp[0].split(' '):
                dict.add(j)
            for j in temp[1].split(' '):
                dict.add(j)

        #此三个变量用来做batch padding用的
        self.samples_context_maxlen_list=np.array(each_sample_context_maxlen_list)
        self.samples_context_sentence_num=np.array(each_sample_context_sentence_num)
        self.samples_response_len=np.array(target_len)
        #

        self.dict={word:i for i,word in enumerate(list(dict))}
        self.max_input_len=max(input_len)
        self.max_target_len=max(target_len)
        self.max_vocab_len=len(self.dict)+1
        print('单词个数为:{}'.format(self.max_vocab_len))
        print('总数是{},训练steps是{}'.format(len(self.train_index),self.steps_per_epoch))
    def generator(self,is_valid=False,use_concept=False,union=True):
        index=self.train_index if is_valid==False else self.valid_index
        data=np.array(self.train_data)
        # print('start generating...')
        if use_concept==False and union:
            encoder_input_data = np.zeros(
                (self.batch_size, self.max_input_len),
                dtype='float32')
            decoder_input_data = np.zeros(
                (self.batch_size, self.max_target_len),
                dtype='float32')
            decoder_target_data = np.zeros(
                (self.batch_size, self.max_target_len, self.max_vocab_len),
                dtype='float32')
            id = 0
            while True:
                if id + self.batch_size < len(index):
                    samples = data[index[id:id + self.batch_size]]
                else:
                    temp_index = np.hstack((index[id:],index[:(id + self.batch_size) % (len(index))]))
                    samples = data[temp_index]
                for i, sample in enumerate(samples):
                    temp = sample.split('\t')
                    input_text = temp[0].split(' ')
                    target_text = temp[1].split(' ')
                    for j, word in enumerate(input_text):
                        encoder_input_data[i, j] = self.dict[word]
                    for j, word in enumerate(target_text):
                        decoder_input_data[i, j] = self.dict[word]
                        if j > 0:
                            decoder_target_data[i, j - 1, self.dict[word]] = 1.0
                inputs = {'encoder_input': encoder_input_data, 'decoder_input': decoder_input_data}
                outputs = {'decoder_target': decoder_target_data}
                yield (inputs, outputs)
                id = (id + self.batch_size) % (len(index))

        elif use_concept==False and union==False:
            id=0
            while True:
                if id + self.batch_size < len(index):
                    samples = data[index[id:id + self.batch_size]]
                    samples_context_maxlens=self.samples_context_maxlen_list[index[id:id + self.batch_size]]
                    samples_context_sentences_nums=self.samples_context_sentence_num[index[id:id + self.batch_size]]
                    samples_response_lens=self.samples_response_len[index[id:id + self.batch_size]]
                else:
                    temp_index =np.hstack((index[id:] , index[:(id + self.batch_size) % len(index)]))
                    samples = data[temp_index]
                    samples_context_maxlens = self.samples_context_maxlen_list[temp_index]
                    samples_context_sentences_nums = self.samples_context_sentence_num[temp_index]
                    samples_response_lens = self.samples_response_len[temp_index]
                encoder_input_data=np.zeros(shape=(self.batch_size,max(samples_context_sentences_nums),max(samples_context_maxlens)),
                                            dtype='float32')
                decoder_input_data=np.zeros(shape=(self.batch_size,max(samples_response_lens)),dtype='float32')
                decoder_target_data=np.zeros(shape=(self.batch_size,max(samples_response_lens),self.max_vocab_len),dtype='float32')
                for i,sample in enumerate(samples):
                    temp=sample.split('\t')
                    context=temp[0].split('<eou>')
                    response=temp[1].split(' ')
                    for j,sentence in enumerate(context):
                        words=sentence.split(' ')
                        for z,word in enumerate(words):
                            encoder_input_data[i,j,z]=self.dict[word]
                    for j,word in enumerate(response):
                        decoder_input_data[i,j]=self.dict[word]
                        if j>0:
                            decoder_target_data[i,j-1,self.dict[word]]=1.0
                inputs = {'encoder_input': encoder_input_data, 'decoder_input': decoder_input_data}
                outputs = {'decoder_target': decoder_target_data}
                # print('encoder_input shape {}, decoder_input_shape {}, decoder_target shape {}'.format(
                #     encoder_input_data.shape,decoder_input_data.shape,decoder_target_data.shape
                # ))
                yield (inputs,outputs)
                id = (id + self.batch_size) % (len(index))

class seq2seq(object):
    def __init__(self,hidden):
        self.hidden=hidden
    def predict(self,x):
        return K.argmax(x,axis=-1)

    def build_network(self,max_vocab_len,baseline=True,is_training=True,union=False,hierarchical=False):
        """
        :param baseline:
        :param is_training:
        :return:
        """
        if baseline and union and hierarchical==False:
            """上下文整合成一句话来训练"""
            encoder_inputs = Input(shape=(None,),name='encoder_input')
            shared_embedding_layer=Embedding(max_vocab_len,self.hidden,mask_zero=True)
            encoder = LSTM(self.hidden, return_state=True)
            encoder_embed=shared_embedding_layer(encoder_inputs)
            encoder_outputs, state_h, state_c = encoder(encoder_embed)
            encoder_states = [state_h, state_c]

            decoder_inputs = Input(shape=(None,),name='decoder_input')
            decoder_lstm = LSTM(self.hidden, return_sequences=True, return_state=True)
            decoder_embed=shared_embedding_layer(decoder_inputs)
            decoder_outputs, _, _ = decoder_lstm(decoder_embed,
                                                 initial_state=encoder_states)
            # decoder_outputs = Dense(max_vocab_len // 2, activation='relu')(decoder_outputs)
            decoder_softmax = Dense(max_vocab_len, activation='softmax',name='decoder_target')(decoder_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_softmax) if is_training else Model(encoder_inputs, decoder_softmax)
            model.summary()
            return model
        elif baseline and union==False and hierarchical==False:
            """上下文分割成多行话来训练，但是非层次训练"""
            encoder_inputs = Input(shape=(None,None,), name='encoder_input')
            shared_embedding_layer = Embedding(max_vocab_len, self.hidden)
            encoder = LSTM(self.hidden, return_state=True)
            encoder_embed = shared_embedding_layer(encoder_inputs)
            encoder_embed=Reshape(target_shape=(-1,self.hidden))(encoder_embed)
            encoder_outputs, state_h, state_c = encoder(encoder_embed)
            encoder_states = [state_h, state_c]

            decoder_inputs = Input(shape=(None,), name='decoder_input')
            decoder_lstm = LSTM(self.hidden, return_sequences=True, return_state=True)
            decoder_embed = shared_embedding_layer(decoder_inputs)
            decoder_outputs, _, _ = decoder_lstm(decoder_embed,
                                                 initial_state=encoder_states)
            # decoder_outputs=Dense(max_vocab_len//2,activation='relu')(decoder_outputs)
            decoder_softmax = Dense(max_vocab_len, activation='softmax',name='decoder_target')(decoder_outputs)
            # decoder_argmax=Lambda(function=self.predict,name='decoder_target')(decoder_softmax)
            model = Model([encoder_inputs, decoder_inputs], decoder_softmax) if is_training else Model(encoder_inputs,
                                                                                                       decoder_softmax)
            model.summary()
            return model

    def train(self,batch_size,baseline=True,train_data_path='../Data/train_3.txt',split_ratio=0.1,
              union=False,hierarchical=False):
        data=Data_2(train_data_path=train_data_path,batch_size=batch_size,
                    split_ratio=split_ratio)
        model=self.build_network(max_vocab_len=data.max_vocab_len,is_training=True,baseline=baseline,hierarchical=hierarchical,
                                 union=union)
        model.compile(
            optimizer=RMSprop(lr=0.001),
            loss=['categorical_crossentropy']
        )
        model_att='union' if union else 'multi-lines'+'_hier_' if hierarchical else '_unhier_'
        model.fit_generator(
            generator=data.generator(is_valid=False,union=union),
            steps_per_epoch=data.steps_per_epoch,
            validation_data=data.generator(is_valid=True,union=union),
            validation_steps=data.valid_steps_per_epoch,
            epochs=100,
            initial_epoch=0,
            callbacks=[
                TensorBoard('logs'),
                ReduceLROnPlateau(patience=8,verbose=1,monitor='val_loss'),
                EarlyStopping(monitor='val_loss',min_delta=1e-4,patience=28,verbose=1),
                ModelCheckpoint(filepath='models/'+model_att+'seq2seq-{epoch:03d}--{val_loss:.5f}--{loss:.5f}.hdf5',
                                save_best_only=True,save_weights_only=False,period=4)
            ]
        )
if __name__=='__main__':
    #'/diskA/wenqiang/lishuai/seq2seq_stable/data/train_3.txt'
    app=seq2seq(hidden=256)
    app.train(batch_size=32,baseline=True,union=False,hierarchical=False,split_ratio=0.2,
              train_data_path='/diskA/wenqiang/lishuai/seq2seq_stable/Data/valid_3.txt')