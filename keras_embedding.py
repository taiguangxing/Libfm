from keras.layers.embeddings import Embedding
from keras import Sequential
from keras.layers import Dense,LSTM
import pandas as pd
import numpy as np
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import copy
from functools import wraps
data = pd.read_csv('D:/data_set/lestore_rec/user_download_sequence.csv')
from functools import update_wrapper,partial
def wrapped_partial(func,*args,**kwargs):
    top3_acc = partial(func, k=args)
    update_wrapper(top3_acc,func)
    return top3_acc
top3_acc = wrapped_partial(keras.metrics.top_k_categorical_accuracy,3)
print(top3_acc.__name__)


data['split']=data['user_download_package_yesterday.items'].map(lambda x:x.split('|'))
all_app_list = list(data['split'])
# print(all_app_list)

tokenizer = Tokenizer(num_words=3000)
tokenizer.fit_on_texts(all_app_list)
print(len(tokenizer.index_word))
print(tokenizer.index_word[3])

test =tokenizer.word_index['com.android.deskclock'] #if 'com.android.deskclock' in tokenizer.word_index() else 0
print(test)
def generate_train_data(all_app_list,train_seq_length):
    X=[]
    Y=[]
    for app_list in all_app_list:
        if len(app_list)<=train_seq_length:
            X.append(tokenizer.texts_to_sequences([app_list[:-1]])[0])
            Y.append(tokenizer.word_index[app_list[-1]] if app_list[-1] in tokenizer.word_index else 0)
        else:
            for i in range(len(app_list)-train_seq_length):
                X.append(tokenizer.texts_to_sequences([app_list[i:i+train_seq_length]])[0])
                Y.append(tokenizer.word_index[app_list[i+train_seq_length]] if app_list[i+train_seq_length] in tokenizer.word_index else 0)
    return X,to_categorical(Y)

X,y = generate_train_data(all_app_list,train_seq_length=12)
print(len(X))
X = keras.preprocessing.sequence.pad_sequences(X,maxlen=12,padding='pre')
train_x,test_x,train_y,test_y =train_test_split(X,y,test_size=0.2,random_state=42)


model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1,50,input_length=12,trainable=True))
model.add(LSTM(50))
model.add(Dense(1000))
model.add(Dense(train_y.shape[1],activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy','top_k_categorical_accuracy'])
model.fit(train_x,train_y,batch_size=100,epochs=10,verbose=1,validation_data=(test_x,test_y))



def getListMaxNumIndex(num_list,topk=3):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    tmp_list=copy.deepcopy(num_list)
    tmp_list.sort()
    max_num_index=[(num_list.index(one),one) for one in tmp_list[::-1][:topk]]
    # min_num_index=[(num_list.index(one),one) for one in tmp_list[:topk]]
    return max_num_index
count=0
for kk in test_y:
    if kk[1]==1:
        count+=1
print(count)
def test_human(index):
    print('历史下载序列')
    for id in test_x[index]:
        print(tokenizer.index_word[id])
    print('用户真实下一次点击',tokenizer.index_word[np.argmax(test_y[index])])
    print('预测top10可能下载的应用为')
    res=model.predict(test_x[index:index+1,:])[0].tolist()
    for ind,probility in getListMaxNumIndex(res,10):
        print(ind,tokenizer.index_word[ind],probility)
test_human(1289)

