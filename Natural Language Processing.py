#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing

# # Install libraries

# In[1]:


get_ipython().system('pip install keras')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install pandas')
get_ipython().system('pip install nltk')
get_ipython().system('pip install sklearn')


# # Machine learning

# ## Importing the libraries

# In[65]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[66]:


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# In[67]:


dataset.tail()


# ## Cleaning the texts

# In[68]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# ## Creating the Bag of Words model

# In[69]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# ## Splitting the dataset into the Training set and Test set

# In[70]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ## Training the Naive Bayes model on the Training set

# In[71]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[72]:


print(X_test.size)


# ## Predicting the Test set results

# In[73]:


y_pred = classifier.predict(X_test)


# In[74]:


print(y_pred)


# ## Making the Confusion Matrix

# In[78]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[79]:


from sklearn.metrics import accuracy_score
print("Model accuracy using Naive Bayes model -- ",accuracy_score(y_test, y_pred))


# # Neural networks

# In[1]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#df.columns = ["label","text"]
x = df['Review'].values
y = df['Liked'].values

x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.1, random_state=123)
#print(x_test)

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(x)
xtrain= tokenizer.texts_to_sequences(x_train)
xtest= tokenizer.texts_to_sequences(x_test)

vocab_size=len(tokenizer.word_index)+1

maxlen=10
xtrain=pad_sequences(xtrain,padding='post', maxlen=maxlen)
xtest=pad_sequences(xtest,padding='post', maxlen=maxlen) 
 
print(x_train[3])
print(xtrain[3])


# # Model architecture

# In[2]:


embedding_dim=50

model=Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(layers.LSTM(units=50,return_sequences=True))
model.add(layers.LSTM(units=10))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8))
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
model.summary()


# # Model training

# In[3]:


model.fit(xtrain,y_train, epochs=100, batch_size=32, verbose=True)


# # Model predicions

# In[84]:


loss, acc = model.evaluate(xtrain, y_train, verbose=False)
print("Training Accuracy: ", acc.round(2))
loss, acc = model.evaluate(xtest, y_test, verbose=False)
print("Test Accuracy: ", acc.round(2))

ypred=model.predict(xtest)

ypred[ypred>0.5]=1 
ypred[ypred<=0.5]=0 
cm = confusion_matrix(y_test, ypred)
print(cm)

result=zip(x_test, y_test, ypred)

for i in result:
    print(i)


# In[5]:


model.save("model.hdf5")


# # predicitions for new text

# In[85]:


def predict_review_class(review: str):
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(x)
    review_seq= tokenizer.texts_to_sequences([review])
    vocab_size=len(tokenizer.word_index)+1
    maxlen=10
    review_seq=pad_sequences(review_seq,padding='post', maxlen=maxlen) 
    result = model.predict_classes(review_seq)
    if result == 1:
        print("Good review, well done.")
    else:
        print("Bad review.")


# In[86]:


predict_review_class("this is so cool")


# In[ ]:




