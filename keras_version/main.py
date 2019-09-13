import numpy as np
from keras.layers import Input,LSTM,Dense,Embedding,Reshape,Lambda,Dropout,RepeatVector
from keras import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,TensorBoard,ModelCheckpoint
from keras.utils import plot_model
"""
Word level seq2seq models
"""

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T
class Data(object):
    def __init__(self,train_data_path,batch_size,split_ratio,union=False):
        """

        :param train_data_path:
        :param batch_size:
        :param split_ratio:
        :param union:
        """
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
        dict=[]
        dict.append('<pad>')
        dict.append('<eou>')
        dict.append('<unk>')
        # \t response的开始， \n response的结束
        dict.append('\t')
        dict.append('\n')
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
            temp[1]='\t '+temp[1]+' \n'
            target_len.append(len(temp[1].split(' ')))
            for j in temp[0].split(' '):
                if j not in dict:
                    dict.append(j)
            for j in temp[1].split(' '):
                if j not in dict:
                    dict.append(j)
        #此三个变量用来做batch padding用的
        self.samples_context_maxlen_list=np.array(each_sample_context_maxlen_list)
        self.samples_context_sentence_num=np.array(each_sample_context_sentence_num)
        self.samples_response_len=np.array(target_len)
        print('最大的行数是{}，最大的句子单词个数是{},最大的反馈的长度是{}'.format(max(self.samples_context_sentence_num),
                                             max(self.samples_context_maxlen_list),max(self.samples_response_len)))
        self.dict={word:i for i,word in enumerate(dict)}
        self.max_vocab_len = len(self.dict) + 1
        self.max_input_len=max(input_len)
        self.max_target_len=max(target_len)
        del each_sample_context_maxlen_list,each_sample_context_sentence_num,dict,input_len,target_len
        del train_index,valid_num

    def generator(self,is_valid=False,use_concept=False):
        """

        :param is_valid:
        :param use_concept:
        :return:
        """
        index=self.train_index if is_valid==False else self.valid_index
        data=np.array(self.train_data)
        if use_concept==False:
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
                if self.union==False:
                    encoder_input_data = np.zeros(
                        shape=(self.batch_size, max(samples_context_sentences_nums), max(samples_context_maxlens)))
                else:
                    encoder_input_data = np.zeros(
                        shape=(self.batch_size, max(self.samples_context_sentence_num), max(samples_context_maxlens)))
                decoder_input_data=np.zeros(shape=(self.batch_size,max(samples_response_lens)))
                decoder_target_data=np.zeros(shape=(self.batch_size,max(samples_response_lens)))
                for i,sample in enumerate(samples):
                    temp=sample.split('\t')
                    context=temp[0].split('<eou>')
                    temp[1]='\t '+temp[1]+' \n'
                    response=temp[1].split(' ')
                    for j,sentence in enumerate(context):
                        words=sentence.split(' ')
                        for z,word in enumerate(words):
                            encoder_input_data[i,j,z]=self.dict[word]
                    for j,word in enumerate(response):
                        decoder_input_data[i,j]=self.dict[word]
                        if j>0:
                            decoder_target_data[i,j-1]=self.dict[word]
                inputs = {'encoder_input': encoder_input_data, 'decoder_input': decoder_input_data,'decoder_target':decoder_target_data}
                outputs = {'nllloss': np.zeros([self.batch_size])}
                yield (inputs,outputs)
                id = (id + self.batch_size) % (len(index))

class seq2seq(object):
    def __init__(self,hidden):
        """

        :param hidden:
        """
        self.hidden=hidden
    def nllloss(self,x,max_vocab_len=None):
        """
        custom negative log likelihood loss for Keras

        :param x:
        :param max_vocab_len:
        :return:
        """
        softmax_logs,targets=x
        targets=K.cast(targets,dtype='int32')
        targets = K.one_hot(targets, max_vocab_len) if max_vocab_len!=None else K.one_hot(targets,int(softmax_logs.get_shape()[-1]))
        res=K.categorical_crossentropy(targets,softmax_logs)
        return K.mean(res)
    def concat_function(self,x):
        t=K.concatenate(x,axis=1)
        return t
    def repeat(self,x):
        return RepeatVector(1)(x)

    def slice_repeat(self,x,context_line_num,depth,dropout):
        state = None
        encoder_hidden_states = []
        encoder_outputs=[]
        for level in range(depth):
            level_lstm=LSTM(self.hidden,return_sequences=True,return_state=True,name='encoder_lstm_{}'.format(level))
            level_dropout=Dropout(dropout,name='encoder_dropout_{}'.format(level))
            if level==0:
                for i in range(context_line_num):
                    try:
                        utt_encoder_output, state_h, state_c = level_lstm(x[:, i, :, :], initial_state=state)
                        utt_encoder_output = level_dropout(utt_encoder_output)
                        encoder_outputs.append(utt_encoder_output)
                        state = [state_h, state_c]
                        state_h = RepeatVector(1)(state_h)
                        encoder_hidden_states.append(state_h)
                    except Exception as e:
                        print(e)
                        break
            else:
                for i in range(context_line_num):
                    try:
                        utt_encoder_output, state_h, state_c = level_lstm(encoder_outputs[i], initial_state=state)
                        utt_encoder_output = level_dropout(utt_encoder_output)
                        encoder_outputs[i] = utt_encoder_output
                        state = [state_h, state_c]
                        state_h = RepeatVector(1)(state_h)
                        encoder_hidden_states.append(state_h)
                    except Exception as e:
                        print(e)
                        break
        encoder_hidden_states = K.concatenate(encoder_hidden_states,axis=1)
        print(encoder_hidden_states.get_shape())
        return encoder_hidden_states

    def build_network(self,max_vocab_len,is_training=True,hierarchical=False,
                      context_line_num=None,depth=1,dropout=0.0,attention=False,vis=False):
        """

        :param max_vocab_len:
        :param baseline:
        :param is_training:
        :param union:
        :param hierarchical:
        :param context_maxlen:
        :param context_maxlines:
        :param response_max_num:
        :param batch_size:
        :return:
        """
        if isinstance(depth,list) or isinstance(depth,tuple):
            pass
        else:
            depth = (depth, depth)
        if hierarchical==False:

            encoder_inputs = Input(shape=(None, None,), name='encoder_input')
            shared_embedding_layer = Embedding(max_vocab_len, self.hidden,name='shared_embedding')
            encoder_embed = shared_embedding_layer(encoder_inputs)
            encoder_embed = Reshape(target_shape=(-1, self.hidden),name='reshape_layer')(encoder_embed)
            encoder_outputs, state_h, state_c = LSTM(self.hidden, return_state=True,return_sequences=True,
                                                     name='encoder_lstm_0')(encoder_embed)
            encoder_states = [state_h, state_c]

            for _ in range(1, depth[0]):
                encoder_outputs = Dropout(dropout,name='encoder_dropout_{}'.format(_-1))(encoder_outputs)
                encoder_outputs, state_h, state_c = LSTM(self.hidden, return_state=True,return_sequences=True,
                                                         name='encoder_lstm_{}'.format(_))(encoder_outputs,initial_state=encoder_states)
                encoder_states = [state_h, state_c]

            decoder_inputs = Input(shape=(None,), name='decoder_input')
            decoder_state_input_h = Input(shape=(self.hidden,),name='decoder_state_input_h')
            decoder_state_input_c = Input(shape=(self.hidden,),name='decoder_state_input_c')
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_embed = shared_embedding_layer(decoder_inputs)

            def decoder_lstm_function(is_training):
                if is_training:
                    decoder_emb = Dropout(dropout,name='decoder_dropout_0')(decoder_embed)
                    decoder_tensor, s_h, s_c = LSTM(self.hidden, return_sequences=True, return_state=True,name='decoder_lstm_0')(decoder_emb,
                                                                                                           initial_state=encoder_states)
                    states = [s_h, s_c]
                    for _ in range(1, depth[1]):
                        decoder_tensor=Dropout(dropout,name='decoder_dropout_{}'.format(_))(decoder_tensor)
                        decoder_tensor, s_h, s_c = LSTM(self.hidden, return_state=True, return_sequences=True,name='decoder_lstm_{}'.format(_))(
                            decoder_tensor,
                            initial_state=states)
                        states = [s_h, s_c]
                    return decoder_tensor
                else:
                    decoder_emb = Dropout(dropout,name='decoder_dropout_0')(decoder_embed)
                    #注意此时输入的initial state是 decoder_input_states
                    decoder_tensor, s_h, s_c = LSTM(self.hidden, return_sequences=True, return_state=True,name='decoder_lstm_0')(decoder_emb,
                                                                                                           initial_state=decoder_states_inputs)
                    states = [s_h, s_c]
                    for _ in range(1, depth[1]):
                        decoder_tensor = Dropout(dropout, name='decoder_dropout_{}'.format(_))(decoder_tensor)
                        decoder_tensor, s_h, s_c = LSTM(self.hidden, return_state=True, return_sequences=True,name='decoder_lstm_{}'.format(_))(
                            decoder_tensor,
                            initial_state=states)
                        states = [s_h, s_c]
                    return decoder_tensor, states

            decoder_outputs = decoder_lstm_function(is_training=True)
            decoder_dense = Dense(max_vocab_len, activation='softmax',name='softmax_vocab_len')
            decoder_softmax = decoder_dense(decoder_outputs)
            decoder_target = Input(shape=(None,), name='decoder_target')
            nllloss = Lambda(function=self.nllloss, name='nllloss', arguments={'max_vocab_len': max_vocab_len})(
                [decoder_softmax, decoder_target])
            train_model = Model([encoder_inputs, decoder_inputs, decoder_target], nllloss)

            # for inference models
            encoder_model = Model(encoder_inputs, encoder_states)

            decoder_outputs, decoder_states = decoder_lstm_function(is_training=False)
            decoder_softmax = decoder_dense(decoder_outputs)
            decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_softmax] + decoder_states)
            if is_training:
                train_model.summary()
                if vis:
                    plot_model(train_model,'pngs/train_model.png')
                return train_model
            else:
                encoder_model.summary()
                decoder_model.summary()
                if vis:
                    plot_model(encoder_model,'pngs/encoder_model.png')
                    plot_model(decoder_model,'pngs/decoder_model.png')
                return encoder_model, decoder_model
        else:
            encoder_inputs = Input(shape=(context_line_num,None,), name='encoder_input')
            shared_embedding_layer = Embedding(max_vocab_len, self.hidden, name='shared_embedding')
            encoder_embed = shared_embedding_layer(encoder_inputs)
            encoder_hidden_states=Lambda(self.slice_repeat,
                                         arguments={'context_line_num':context_line_num,'depth':depth[0],'dropout':dropout},
                                         name='encoder_lambda')(encoder_embed)
            decoder_inputs = Input(shape=(None,), name='decoder_input')
            decoder_state_input_h = Input(shape=(None,self.hidden,), name='decoder_state_input_h')

            decoder_embed = shared_embedding_layer(decoder_inputs)

            context_lstm=LSTM(self.hidden,return_state=True,name='context_lstm')
            if is_training == True:
                context_ouputs, context_h, context_c = context_lstm(encoder_hidden_states)
            else:
                context_outputs,context_h,context_c= context_lstm(decoder_state_input_h)
            context_states=[context_h,context_c]
            def decoder_lstm_function(is_training):
                decoder_emb = Dropout(dropout, name='decoder_dropout_0')(decoder_embed)
                decoder_tensor, s_h, s_c = LSTM(self.hidden, return_sequences=True, return_state=True,
                                                name='decoder_lstm_0')(decoder_emb,
                                                                       initial_state=context_states)
                states = [s_h, s_c]
                for _ in range(1, depth[1]):
                    decoder_tensor = Dropout(dropout, name='decoder_dropout_{}'.format(_))(decoder_tensor)
                    decoder_tensor, s_h, s_c = LSTM(self.hidden, return_state=True, return_sequences=True,
                                                    name='decoder_lstm_{}'.format(_))(
                        decoder_tensor,
                        initial_state=states)
                    states = [s_h, s_c]
                if is_training:
                    return decoder_tensor
                else:
                    return decoder_tensor,states

            decoder_outputs = decoder_lstm_function(is_training=True)
            decoder_dense = Dense(max_vocab_len, activation='softmax', name='softmax_vocab_len')
            decoder_softmax = decoder_dense(decoder_outputs)
            decoder_target = Input(shape=(None,), name='decoder_target')
            nllloss = Lambda(function=self.nllloss, name='nllloss', arguments={'max_vocab_len': max_vocab_len})(
                [decoder_softmax, decoder_target])

            if is_training:
                train_model = Model([encoder_inputs, decoder_inputs, decoder_target], nllloss)
                train_model.summary()
                if vis:
                    plot_model(train_model,'pngs/hierarchical_train_model.png')
                return train_model
            else:
                encoder_model = Model(encoder_inputs, encoder_hidden_states)
                decoder_outputs, decoder_states = decoder_lstm_function(is_training=False)
                decoder_softmax = decoder_dense(decoder_outputs)
                decoder_model = Model(
                    [decoder_inputs] + [decoder_state_input_h],
                    [decoder_softmax] + decoder_states)
                encoder_model.summary()
                decoder_model.summary()
                if vis:
                    plot_model(encoder_model,'pngs/hierarchical_encoder_model.png')
                    plot_model(decoder_model,'pngs/hierarchical_decoder_model.png')
                return encoder_model,decoder_model


    def train(self,batch_size,train_data_path='../Data/train_3.txt',
              split_ratio=0.1,valid_data_path='../Data/valid_3.txt',predict_model_path=None,
              union=False,hierarchical=False,depth=1,dropout=0.0,attention=False,mode=1):
        """

        :param batch_size:
        :param train_data_path:
        :param split_ratio:
        :param union:
        :param hierarchical:
        :param depth:
        :param dropout:
        :param attention:
        :param mode: 1 denotes train mode and others means prediction mode
        :return:
        """
        if hierarchical:
            union=True
        data=Data(train_data_path=train_data_path,batch_size=batch_size,
                    split_ratio=split_ratio,union=union)
        if hierarchical==False:
            model = self.build_network(max_vocab_len=data.max_vocab_len, is_training=True,
                                       hierarchical=hierarchical,
                                       depth=depth,dropout=dropout,attention=attention)
            encoder_model,decoder_model=self.build_network(max_vocab_len=data.max_vocab_len, is_training=False,
                                       hierarchical=hierarchical,
                                       depth=depth,dropout=dropout,attention=attention)
            model.compile(
                optimizer=Adam(lr=0.001),
                loss={'nllloss': lambda y_true, y_pred: y_pred}
            )
        else:
            model = self.build_network(max_vocab_len=data.max_vocab_len, is_training=True,
                                       hierarchical=hierarchical,
                                       context_line_num=max(data.samples_context_sentence_num),
                                       depth=depth, dropout=dropout, attention=attention)
            encoder_model, decoder_model = self.build_network(max_vocab_len=data.max_vocab_len, is_training=False,
                                                              hierarchical=hierarchical,
                                                              context_line_num=max(data.samples_context_sentence_num),
                                                              depth=depth, dropout=dropout, attention=attention)
            model.compile(
                optimizer=Adam(lr=0.001),
                loss={'nllloss': lambda y_true, y_pred: y_pred})
        model_att='new_union' if union else 'new_multi-lines'+'_hier_' if hierarchical else '_unhier_'
        if mode==1:
            model.fit_generator(
                generator=data.generator(is_valid=False),
                steps_per_epoch=data.steps_per_epoch,
                validation_data=data.generator(is_valid=True),
                validation_steps=data.valid_steps_per_epoch,
                epochs=100,
                initial_epoch=0,
                callbacks=[
                    TensorBoard('logs'),
                    ReduceLROnPlateau(patience=8, verbose=1, monitor='val_loss'),
                    EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=28, verbose=1),
                    ModelCheckpoint(
                        filepath='models/' + model_att + 'seq2seq-{epoch:03d}--{val_loss:.5f}--{loss:.5f}.hdf5',
                        save_best_only=False, save_weights_only=False, period=4, verbose=1)
                ]
            )
        else:
            if predict_model_path==None:
                raise EnvironmentError('please set the predict_model_path')
            elif valid_data_path==None:
                raise EnvironmentError('please set the valid_data_path')
            else:
                print('loading the models...')
                encoder_model.load_weights(predict_model_path,by_name=True)
                decoder_model.load_weights(predict_model_path,by_name=True)
                print('finished!')
                id2word={i:word for word,i in data.dict.items() }
                valid_data=open(valid_data_path,'r',encoding='utf-8').readlines()
                for index in range(len(valid_data)):
                    batch_valid_data=valid_data[index]
                    batch_encoder_input=np.zeros(shape=(1,max(data.samples_context_sentence_num),
                                                        max(data.samples_context_maxlen_list)))
                    #max(data.samples_response_len)
                    batch_decoder_input=np.zeros(shape=(1,max(data.samples_response_len)))
                    ground_truths=[]
                    contexts=[]
                    generated_responses=[]
                    for i,sample in enumerate(batch_valid_data):
                        temp=sample.split('\t')
                        context=temp[0].split('<eou>')
                        contexts.append('\n'.join(context))
                        # ground_truths.append(temp[1])
                        for line,sentence in enumerate(context):
                            if line<max(data.samples_context_sentence_num):
                                words=sentence.split(' ')
                                for z,word in enumerate(words):
                                    if z<max(data.samples_context_maxlen_list):
                                        try:
                                            batch_encoder_input[i, line, z] = data.dict[word]
                                        except:
                                            batch_encoder_input[i, line, z] = data.dict['<unk>']
                    encoder_states=encoder_model.predict(batch_encoder_input)
                    batch_decoder_input[0,0]=data.dict['\t']
                    stop_condition = False
                    decoded_sentence = ''
                    while not stop_condition:
                        output_tokens, h, c = decoder_model.predict(
                            [batch_decoder_input] + [encoder_states])
                        sampled_token_index = np.argmax(output_tokens[:, 0, :],axis=-1)
                        print(sampled_token_index)
                        break
                        # decoder_word=id2word[sampled_token_index]
                    break

if __name__=='__main__':
    app=seq2seq(hidden=256)
    app.train(batch_size=2,train_data_path='../Data/train_3.txt',union=False,hierarchical=True,
              depth=2,dropout=0.3,mode=1,predict_model_path='models/nonebatch_unionseq2seq-004--0.99970--1.00288.hdf5',
              valid_data_path='../Data/valid_3.txt')