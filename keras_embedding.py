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
data = pd.read_csv('D:/data_set/lestore_rec/user_download_sequence.csv')

import functools
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

print(data.head(5))
print(data.shape)

data['split']=data['user_download_package_yesterday.items'].map(lambda x:x.split('|'))

all_app_list = list(data['split'])
print(all_app_list)

tokenizer = Tokenizer.fit_on_texts(all_app_list)

tmp=[]
for app_list in all_app_list:
    tmp.extend(app_list)
all_app_set = set(tmp)
app_id_dict = {}
id_app_dict = {}
for ind,app in enumerate(all_app_set):
    app_id_dict[app]=ind+1
    id_app_dict[ind+1]=app

print(len(app_id_dict))

def get_train_data(all_app_list,train_seq_length):
    X=[]
    Y=[]
    for app_list in all_app_list:
        if len(app_list)<=train_seq_length:
            X.append([app_id_dict[app] for app in app_list[:-2] ])
            Y.append(app_id_dict[app_list[-1]])
        else:
            for i in range(len(app_list)-train_seq_length):
                X.append([app_id_dict[app] for app in app_list[i:i+train_seq_length]])
                Y.append(app_id_dict[app_list[i+train_seq_length]])
    return X,to_categorical(Y)

X,y = get_train_data(all_app_list,12)
print(len(X),len(y))

X = keras.preprocessing.sequence.pad_sequences(X,maxlen=12,padding='pre')
train_x,test_x,train_y,test_y =train_test_split(X,y,test_size=0.2,random_state=42)


model = Sequential()
model.add(Embedding(len(app_id_dict)+1,50,input_length=12,trainable=True))
model.add(LSTM(30))
model.add(Dense(1000))
model.add(Dense(train_y.shape[1],activation='softmax'))
model.compile('adam','categorical_crossentropy',metrics=['accuracy','top_k_categorical_accuracy',top3_acc])


model.fit(train_x,train_y,batch_size=100,epochs=5,verbose=1,validation_data=(test_x,test_y))


test_x.shape

test_x[:1,:].shape
res=model.predict(test_x[:1,:])
class_res=model.predict_classes(test_x[:1,:])
print(class_res)



def getListMaxNumIndex(num_list,topk=3):
    '''
    获取列表中最大的前n个数值的位置索引
    '''

    tmp_list=copy.deepcopy(num_list)
    tmp_list.sort()
    max_num_index=[num_list.index(one) for one in tmp_list[::-1][:topk]]
    min_num_index=[num_list.index(one) for one in tmp_list[:topk]]
    print ('max_num_index:',max_num_index)
    print ('min_num_index:',min_num_index)


getListMaxNumIndex(res[0].tolist())


print(res.shape)
print(res)









