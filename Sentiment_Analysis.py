#!/usr/bin/env python
# coding: utf-8

# In[71]:


## Import Libaries

import numpy as np
import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


# In[72]:


lm = WordNetLemmatizer()


# In[73]:


## Stop words
sw = stopwords.words('english')


# In[74]:


## Reading the positive data
pos_rev = pd.read_csv(r'C:\Users\Darshana\Desktop\DSC_WKND20092020\NLP\netflix\pos.txt',header=None,sep='\n',encoding='latin-1')
pos_rev


# In[75]:


## Creating Target Column & renaming the column
pos_rev['mood'] = 1.0
pos_rev = pos_rev.rename(columns={0:'review'})
pos_rev


# In[76]:


## Reading the negative data
neg_rev = pd.read_csv(r'C:\Users\Darshana\Desktop\DSC_WKND20092020\NLP\netflix\negative.txt',header=None,sep='\n',encoding='latin-1')

## Creating Target Column & renaming the column
neg_rev['mood'] = 0.0
neg_rev = neg_rev.rename(columns={0:'review'})
neg_rev


# In[77]:


pos_rev.loc[:,'review'] = pos_rev.loc[:,'review'].apply(lambda x:x.lower())
pos_rev.loc[:,'review'] = pos_rev.loc[:,'review'].apply(lambda x:re.sub(r'@\S+','',x))
pos_rev.loc[:,'review'] = pos_rev.loc[:,'review'].apply(lambda x:x.translate(str.maketrans(dict.fromkeys((string.punctuation)))))
pos_rev.loc[:,'review'] = pos_rev.loc[:,'review'].apply(lambda x: " ".join([word for word in x.split() if word not in (sw)]))
pos_rev.loc[:,'review'] = pos_rev.loc[:,'review'].apply(lambda x:" ".join([lm.lemmatize(word,pos='v') for word in x.split()]))
pos_rev.loc[:,'review']


# In[78]:


neg_rev.loc[:,'review'] = neg_rev.loc[:,'review'].apply(lambda x:x.upper())
neg_rev.loc[:,'review']


# In[79]:


neg_rev.loc[:,'review'] = neg_rev.loc[:,'review'].apply(lambda x:x.lower())
neg_rev.loc[:,'review'] = neg_rev.loc[:,'review'].apply(lambda x:re.sub(r'@\S+','',x))
neg_rev.loc[:,'review'] = neg_rev.loc[:,'review'].apply(lambda x:x.translate(str.maketrans(dict.fromkeys((string.punctuation)))))
neg_rev.loc[:,'review'] = neg_rev.loc[:,'review'].apply(lambda x: " ".join([word for word in x.split() if word not in (sw)]))
neg_rev.loc[:,'review'] = neg_rev.loc[:,'review'].apply(lambda x:" ".join([lm.lemmatize(word,pos='v') for word in x.split()]))
neg_rev.loc[:,'review']


# In[80]:


com_rev = pd.concat([pos_rev,neg_rev],axis=0).reset_index()
com_rev


# In[81]:


# Train Test Split
xtrain,xtest,ytrain,ytest = train_test_split(com_rev['review'].values,com_rev['mood'].values,random_state=40,test_size=0.2)
# xtrain


# In[82]:


train_data = pd.DataFrame({'review':xtrain,'mood':ytrain})
test_data = pd.DataFrame({'review':xtest,'mood':ytest})


# In[83]:


train_data


# In[84]:


test_data


# In[85]:


# vectorizer = TfidfVectorizer()
# train_vectorizer = vectorizer.fit_transform(train_data['review'])
# test_vectorizer = vectorizer.transform(test_data['review'])


# In[86]:


# test_vectorizer.toarray()


# In[87]:


# train_vectorizer.toarray()


# In[88]:


# from sklearn import svm
# from sklearn.metrics import classification_report


# In[89]:


# classifier = svm.SVC(kernel='linear')
# classifier.fit(train_vectorizer,train_data['mood'])


# In[90]:


# prediction = classifier.predict(test_vectorizer)
# prediction


# In[91]:


# report = classification_report(test_data['mood'],prediction
# #                                ,output_dict=True
#                               )
# print(report)


# In[92]:


# vectorizer_cnt = CountVectorizer()
# train_vectorizer = vectorizer.fit_transform(train_data['review'])
# test_vectorizer = vectorizer.transform(test_data['review'])


# In[93]:


# from sklearn import svm
# from sklearn.metrics import classification_report


# In[94]:


# classifier = svm.SVC(kernel='linear')
# classifier.fit(train_vectorizer,train_data['mood'])


# In[95]:


# prediction = classifier.predict(test_vectorizer)
# prediction


# In[96]:


# report = classification_report(test_data['mood'],prediction
# #                                ,output_dict=True
#                               )
# print(report)


# In[97]:


vectorizer = TfidfVectorizer()
train_vectorizer = vectorizer.fit_transform(train_data['review'])
test_vectorizer = vectorizer.transform(test_data['review'])


# In[98]:


train_vectorizer.toarray()


# In[99]:


ytrain


# In[100]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
classifier = GaussianNB()
classifier.fit(train_vectorizer.toarray(),ytrain)


# In[101]:


predictions = classifier.predict(test_vectorizer.toarray())
predictions


# In[102]:


report = classification_report(ytest,predictions)
print(report)


# In[103]:


# a = input('write the review : ')
# vector = vectorizer.transform([a]).toarray()
# my_pred = classifier.predict(vector)
# print(my_pred)


# In[104]:


import joblib
model_filename = 'Sentiment_Analysis_model.pkl'
vector_filename = 'Sentiment_vector.pkl'
joblib.dump(classifier,model_filename)
joblib.dump(vectorizer,vector_filename)

