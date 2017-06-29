import numpy
import os
import nltk
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import gensim


os.chdir("D:\Deep Learning\Word2Vec");
model = gensim.models.Word2Vec.load('word2vec.bin');
os.chdir("D:\Deep Learning\Alice_in_wonderland");
filename = "alice_in_wonderland.txt"
f = filename.encode('utf-8') 
raw_text = open(f).read()
raw_text = raw_text.lower()
x = [] 
y = []

words = raw_text.split()
for i in range(len(words)-1):
	x.append(words[i]);
	y.append(words[i+1]);

tok_x=[]
tok_y=[]

for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))

sentend=np.ones((300,),dtype=np.float32) 

vec_x=[]
for sent in tok_x:
    sentvec = [model[w] for w in sent if w in model.vocab]
    print(sentvec)
    vec_x.append(sentvec)
    
vec_y=[]
for sent in tok_y:
    sentvec = [model[w] for w in sent if w in model.vocab]
    print(sentvec)
    vec_y.append(sentvec) 

    
for tok_sent in vec_x:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_x:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)    
            
for tok_sent in vec_y:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_y:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)             
            
            
vec_x=np.array(vec_x,dtype=np.float64)
vec_y=np.array(vec_y,dtype=np.float64)  

x_train,x_test, y_train,y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)