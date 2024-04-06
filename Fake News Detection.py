#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer ## Turn text to feed into machine learning model

#TF-IDF  Term frequency - Inverse Document Frequency

from sklearn.svm import LinearSVC  #for texts linear svc


# In[20]:


data = pd.read_csv('fake_or_real_news.csv')


# In[21]:


data


# In[22]:


# Binary label
data['FAKE'] = data['label'].apply(lambda x: 0 if x=='REAL' else 1)


# In[23]:


data


# In[24]:


data = data.drop('label',axis=1)


# In[25]:


data


# In[26]:


X, y =data['text'],data['FAKE']


# In[27]:


X


# In[28]:


y


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[31]:


len(X_train)


# In[32]:


len(X_test)


# In[33]:


X_train


# In[34]:


vectorizer = TfidfVectorizer(stop_words = "english", max_df=0.7)
X_train_vectorized  = vectorizer.fit_transform(X_train)
X_test_vectorized  = vectorizer.transform(X_test)


# In[35]:


clf = LinearSVC()
clf.fit(X_train_vectorized,y_train)


# In[36]:


clf.score(X_test_vectorized,y_test)


# In[37]:


len(y_test)


# In[39]:


with open("mytext.txt","w",encoding="utf-8") as f:
    f.write(X_test.iloc[10])


# In[40]:


with open("mytext.txt","r",encoding="utf-8") as f:
    text=f.read()


# In[41]:


text


# In[43]:


vectorized_text = vectorizer.transform([text])


# In[44]:


clf.predict(vectorized_text)


# In[45]:


y_test.iloc[10]


# In[ ]:




