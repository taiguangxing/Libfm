from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import pickle
import pandas as pd

# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		# if yhat in tokenizer.index_word:
		# 	out_word = tokenizer.index_word[int(yhat)]
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text


data = pd.read_csv('D:/data_set/lestore_rec/user_download_sequence.csv')
data['split']=data['user_download_package_yesterday.items'].map(lambda x:x.split('|'))

black_list = set(['com.lenovo.calendar','com.lenovo.leos.appstore','com.lenovo.browser','com.lenovo.lps.cloud.sync'])
data= list(data['split'])
all_app_list =[]
for pkg_list in data:
	tmp=[]
	for pkg in pkg_list:
		if pkg not in black_list:
			tmp.append(pkg)
	all_app_list.append(tmp)


# # source text
# data = """ Jack and Jill went up the hill\n
# 		To fetch a pail of water\n
# 		Jack fell down and broke his crown\n
# 		And Jill came tumbling after\n """
#
# data2 = '''Mike and Jill are good friends'''
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_app_list)
print(len(tokenizer.word_index))



# 保存编码器模型
with open('./model/app_tokenizer.pickle', 'wb') as f:
	pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
# 下载编码器模型
# with open('./model/app_tokenizer.pickle', 'rb') as handle:
# 	tokenizer = pickle.load(handle)


encoded = tokenizer.texts_to_sequences(all_app_list)




# retrieve vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# encode 2 words -> 1 word
sequences = list()

for i in range(len(encoded)):
	if len(encoded[i])<=6:
		sequences.append(encoded[i])
	else:
		for j in range(5,len(encoded[i])):
			sequence = encoded[i][j-5:j+1]
			sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
# pad sequences


max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
print(y.shape)
# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=100, verbose=2)
model.save('./model/rnn.model')

print(all_app_list[0])
print(type(tokenizer.index_word))
# evaluate model
print(generate_seq(model, tokenizer, max_length-1, ' '.join(all_app_list[1][:6]), 6))

# print(generate_seq(model, tokenizer, max_length-1, 'And Jill', 3))
# print(generate_seq(model, tokenizer, max_length-1, 'Jill fell', 5))
# print(generate_seq(model, tokenizer, max_length-1, 'pail of jack', 3))
