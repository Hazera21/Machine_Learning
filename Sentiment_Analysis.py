
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

data=pd.read_csv(r'E:\Git_ML\LANGCHAIN\IMDB-Dataset.csv')

def clean_text(text):
    text=text.lower()
    lemmatizer=WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(word) for word in text.split() if word not in set(stopwords.words('english'))]
    return ' '.join(tokens)

data['clean_text']=data['review'].apply(clean_text)
print(data[['clean_text','sentiment']].head())

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import TfidfVectorizer

X=data['clean_text']
Y=data['sentiment']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

vectorizer=TfidfVectorizer()
Xtrain_vectorize=vectorizer.fit_transform(X_train)
Xtest_vectorize=vectorizer.transform(X_test)

model=MultinomialNB()
model.fit(Xtrain_vectorize,Y_train)
score=model.score(Xtest_vectorize,Y_test)

import streamlit as st
st.title('Sentiment Analysis of Movie Reviews')
movie_name=st.text_input('Enter the movie name')
user_review=st.text_area('Enter your review here')
with st.form('my_form'):
    submitted=st.form_submit_button('Submit')
    if submitted:
        clean_review=vectorizer.transform(user_review)
        prediction=model.predict(clean_review)
        st.write(f'The overall review for the movie is : {prediction[0]}')
