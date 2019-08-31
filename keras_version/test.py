from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = '../fra-eng/fra.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
# encoder_lstm_layer = LSTM(self.hidden, return_state=True)
            # input_shape = input_encoder.get_shape()
            # print('input shape:{}'.format(input_shape))
            # encoder_embedding_tensor = embedding_layer(input_encoder)
            # encoder_embedding_tensor_shape = encoder_embedding_tensor.get_shape()
            # print('embedding_encoder_tensor_shape is {}'.format(encoder_embedding_tensor_shape))
            # hidden_h, hidden_c = None, None
            # # 这里baseline没有使用到hidden_h和hidden_c 形成新的序列来做 lstm形成dialog_hidden
            # for line in range(self.max_sen):
            #     single_line_tensor = encoder_embedding_tensor[:, line, :, :]
            #     if line == 0:
            #         encoder_outputs, hidden_h, hidden_c = encoder_lstm_layer(single_line_tensor)
            #     else:
            #         encoder_outputs, hidden_h, hidden_c = encoder_lstm_layer(single_line_tensor,
            #                                                                  initial_state=[hidden_h, hidden_c])
            #     print(encoder_outputs.get_shape(), hidden_h.get_shape(), hidden_c.get_shape())
            #
            # decoder_input = Input(shape=(None,), name='decoder_input')
            # decoder_embedding_tensor = embedding_layer(decoder_input)
            # decoder_lstm_layer = LSTM(self.hidden, return_sequences=True)
            # print('*' * 100)
            # print(decoder_embedding_tensor.get_shape())
            # decoder_outputs = decoder_lstm_layer(decoder_embedding_tensor, initial_state=[hidden_h, hidden_c])
            # print(decoder_outputs.get_shape())
            # decoder_predict = Dense(self.max_vocab_len, activation='softmax')(decoder_outputs)
            # print(decoder_predict.get_shape())
            # model = Model([input_encoder, decoder_input], decoder_outputs) if is_training else Model(input_encoder,
            #                                                                                          decoder_predict)
            # model.summary()
            # return model

# train_data=open(train_data_path,'r',encoding='utf-8').readlines()
#         dict_data=json.load(open(dict_path,'r',encoding='utf-8'))
#         replace_key = list(dict_data.keys())[0]
#         dict_data['<pad>'] = 0
#         dict_data['<eou>'] = len(dict_data.keys())
#         dict_data[replace_key] = len(dict_data.keys())
#         num_tokens=len(dict_data.keys())
#         input_texts=[]
#         input_len=[]
#         target_texts=[]
#         target_len=[]
#         for sample in train_data:
#             temp=sample.split('\t')
#             input=temp[0].split(' ')
#             input_len.append(len(input))
#             input_texts.append(input)
#             target=sample[1].split(' ')
#             target_len.append(len(target))
#             target_texts.append(target)
#         print(len(train_data),max(input_len),num_tokens)
#
#         encoder_input_data = np.zeros(
#             (len(train_data), max(input_len), num_tokens),
#             dtype='int32')
#         decoder_input_data = np.zeros(
#             (len(train_data), max(target_len), num_tokens),
#             dtype='int32')
#         decoder_target_data = np.zeros(
#             (len(train_data),max(target_len), num_tokens),
#             dtype='int32')
#         for i,(input_text,target_text) in enumerate(zip(input_texts,target_texts)):
#             for t, word in enumerate(input_text):
#                 encoder_input_data[i,t,dict_data[word]]=1.
#             for t,word in enumerate(target_text):
#                 decoder_input_data[i,t,dict_data[word]]=1.
#                 if t>0:
#                     decoder_target_data[i, t - 1, dict_data[word]] = 1.
#
