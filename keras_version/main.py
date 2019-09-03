import numpy as np
from keras.layers import Input,LSTM,Dense,Embedding,Reshape,Lambda
from keras import Model
from keras.optimizers import Adam,SGD,RMSprop
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import json,time
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,TensorBoard,ModelCheckpoint
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T
class Data(object):
    def __init__(self,train_data_path,batch_size,split_ratio,union=False):
        # print('{}-开始初始化'.format(time.ctime()))
        self.train_data=open(train_data_path,'r',encoding='utf-8').readlines()
        train_index=np.array(range(len(self.train_data)))
        valid_num=int(split_ratio*len(self.train_data))
        self.valid_index=np.random.choice(train_index,size=valid_num,replace=False)
        self.train_index=np.array([i for i in train_index if i not in self.valid_index])
        self.steps_per_epoch=len(self.train_index)//batch_size
        self.valid_steps_per_epoch=len(self.valid_index)//batch_size
        self.batch_size=batch_size
        self.union=union
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
        self.max_vocab_len = len(self.dict) + 1
        if self.union:
            self.all_encoder_input = np.zeros(shape=(len(self.train_data), max(self.samples_context_sentence_num),
                                                     max(self.samples_context_maxlen_list)))
            self.all_decoder_input = np.zeros(shape=(len(self.train_data), max(self.samples_response_len)))
            self.all_decoder_target = np.zeros(shape=(len(self.train_data), max(self.samples_response_len)),dtype='int')
            for i, sample in enumerate(self.train_data):
                temp = sample.split('\t')
                context = temp[0].split('<eou>')
                response = temp[1].split(' ')
                for j, word in enumerate(response):
                    self.all_decoder_input[i, j] = self.dict[word]
                    if j > 0:
                        self.all_decoder_target[i, j - 1] = self.dict[word]
                for j, sentence in enumerate(context):
                    for z, word in enumerate(sentence.split(' ')):
                        self.all_encoder_input[i, j, z] = self.dict[word]
        self.max_input_len=max(input_len)
        self.max_target_len=max(target_len)

    def generator(self,is_valid=False,use_concept=False):
        index=self.train_index if is_valid==False else self.valid_index
        data=np.array(self.train_data)
        # print('{}-开始生成新的批次数据'.format(time.ctime()))
        if use_concept==False and self.union:
            id = 0
            while True:
                if id + self.batch_size < len(index):
                    batch_encoder_input_data=self.all_encoder_input[index[id:id + self.batch_size],:,:]
                    batch_decoder_input_data=self.all_decoder_input[index[id:id + self.batch_size],:]
                    batch_decoder_target_data=self.all_decoder_target[index[id:id + self.batch_size],:]
                    batch_decoder_target_data=to_categorical(batch_decoder_target_data,self.max_vocab_len)
                else:
                    temp_index = np.hstack((index[id:],index[:(id + self.batch_size) % (len(index))]))
                    batch_encoder_input_data = self.all_encoder_input[temp_index, :, :]
                    batch_decoder_input_data = self.all_decoder_input[temp_index, :]
                    batch_decoder_target_data = self.all_decoder_target[temp_index, :]
                    batch_decoder_target_data = to_categorical(batch_decoder_target_data, self.max_vocab_len)

                inputs = {'encoder_input': batch_encoder_input_data, 'decoder_input': batch_decoder_input_data}
                outputs = {'decoder_target': batch_decoder_target_data}
                yield (inputs, outputs)
                id = (id + self.batch_size) % (len(index))

        elif use_concept==False and self.union==False:
            id=0
            #每一个批次的size不一样,不是固定的size
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
                decoder_target_data=[]
                # decoder_target_data=np.zeros(shape=(self.batch_size,max(samples_response_lens),self.max_vocab_len),dtype='float32')
                # 32 × 11 × 20
                # print('{}-开始循环'.format(time.ctime()))
                for i,sample in enumerate(samples):
                    temp=sample.split('\t')
                    context=temp[0].split('<eou>')
                    response=temp[1].split(' ')
                    sample_target_data = np.zeros(shape=(max(samples_response_lens),), dtype='int')
                    for j,sentence in enumerate(context):
                        words=sentence.split(' ')
                        for z,word in enumerate(words):
                            encoder_input_data[i,j,z]=self.dict[word]
                    for j,word in enumerate(response):
                        decoder_input_data[i,j]=self.dict[word]
                        if j>0:
                            sample_target_data[j-1]=int(self.dict[word])
                    # sample_target_onehot=np.zeros((max(samples_response_lens),self.max_vocab_len))
                    # sample_target_onehot[np.arange(max(samples_response_lens)),sample_target_data]=1
                    sample_target_onehot=to_categorical(sample_target_data,self.max_vocab_len)
                    decoder_target_data.append(sample_target_onehot)
                decoder_target_data=np.array(decoder_target_data)
                inputs = {'encoder_input': encoder_input_data, 'decoder_input': decoder_input_data}
                outputs = {'decoder_target': decoder_target_data}
                # print('encoder_input shape {}, decoder_input_shape {}, decoder_target shape {}'.format(
                #     encoder_input_data.shape,decoder_input_data.shape,decoder_target_data.shape
                # ))
                # print('{}-送入网络'.format(time.ctime()))
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
        if baseline and hierarchical==False:
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
        data=Data(train_data_path=train_data_path,batch_size=batch_size,
                    split_ratio=split_ratio,union=union)
        model=self.build_network(max_vocab_len=data.max_vocab_len,is_training=True,baseline=baseline,hierarchical=hierarchical,
                                 union=union)
        model.compile(
            optimizer=Adam(lr=0.001),
            loss=['categorical_crossentropy']
        )
        model_att='union' if union else 'multi-lines'+'_hier_' if hierarchical else '_unhier_'
        model.fit_generator(
            generator=data.generator(is_valid=False),
            steps_per_epoch=data.steps_per_epoch,
            validation_data=data.generator(is_valid=True),
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
    #'/diskA/wenqiang/lishuai/seq2seq_stable/Data/train_3.txt'
    app=seq2seq(hidden=256)
    app.train(batch_size=32,baseline=True,union=True,hierarchical=False,split_ratio=0.2,
              train_data_path='../Data/train_3.txt')