#project - Training a MLPClassifier Model to predict the Emotion associated with the tweet


#importing libraries
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as mp
import math as ma
import pandas as pa
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#importing dataset
emotions=['neutral','worry','happiness','saddness','fun','surprise','anger']
ds=pa.read_csv('data.csv')
ds=ds[ds['sentiment'].isin(emotions)]
ds=ds.iloc[:,:].values

#Encoding the Emotion Labels
encoder=LabelEncoder()
ds[:,1]=encoder.fit_transform(ds[:,1])
x_train,x_test=train_test_split(ds,shuffle=True,train_size=0.8)
x=list(x_train[:,3])
y=list(x_train[:,1])

#Preprocessing

##removing links, hashtags, numbers, handles
def rempun(a):
    s=''
    for i in a:
        if i.isalpha():
            s+=i
    return s

tmp=[]
for i in x:
    li=[]
    i=str(i)
    s=i.split()
    for j in s:
        j=j.lower()
        if j.isnumeric() or j[0]=='@':
            a=1
        elif j[0]=='#':
            j=j.replace('#','')
            li.append(rempun(j))
        elif len(j)>=4 and j[0]!='h' and j[1]!='t' and j[2]!='t' and j[3]!='p':
            li.append(rempun(j))
        elif len(s)==3:
            li.append(rempun(j))
    tmp.append(li)
x=tmp
del(tmp)



#Preparing BOW(Bag of Words) model
count=0
map={}
for i in x:
    for j in i:
        tmp=map.get(j,'a')
        if tmp=='a':
            map[j]=1
        else:
            map[j]=map[j]+1

l=[]
for i in map.keys():
    if map[i]>=2:
        l.append([map[i],i])
l.sort(reverse=True)
l=l[:2000]

##preparing the container
con=[]
for i in l:
    con.append(i[1])


#Preparing the training matrix taking 2000 feautres
input=[]
for i in x:
    terms=[]
    for j in con:
        co=0
        for k in i:
            if k==j:
                co=1
                break
        terms.append(co)
    input.append(terms)

x_train=np.array(input)
y_train=np.array(y)

#Training Model (Artificial Neural Network)
net=MLPClassifier()
net=net.fit(x_train,y_train)

#Predictions
x=list(x_test[:,3])
y=list(x_test[:,1])

#Preprocessing

##removing links, hashtags, numbers, handles
def rempun(a):
    s=''
    for i in a:
        if i.isalpha():
            s+=i
    return s


tmp=[]
for i in x:
    li=[]
    i=str(i)
    s=i.split()
    for j in s:
        j=j.lower()
        if j.isnumeric() or j[0]=='@':
            a=1
        elif j[0]=='#':
            j=j.replace('#','')
            li.append(rempun(j))
        elif len(j)>=4 and j[0]!='h' and j[1]!='t' and j[2]!='t' and j[3]!='p':
            li.append(rempun(j))
        elif len(s)==3:
            li.append(rempun(j))
    tmp.append(li)
x=tmp
del(tmp)


#Preparing the training matrix taking 2000 feautres
input=[]
for i in x:
    terms=[]
    for j in con:
        co=0
        for k in i:
            if k==j:
                co=1
                break
        terms.append(co)
    input.append(terms)

x_train=np.array(input)
y_train=np.array(y)

#Accuracy
from sklearn.metrics import accuracy_score

y_pred=net.predict(x_train)
confusion_matrix(y_train,y_pred)
accuracy_score(y_train,y_pred)
























