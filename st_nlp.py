import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

def nlp_predict(base,text):
    v = TfidfVectorizer()
    x = v.fit_transform(base)
    clf = TruncatedSVD(3)
    Xpca = clf.fit_transform(x)
    kmeans = KMeans(n_clusters=10).fit(Xpca)
    text_tfidf = v.transform([text])
    text_pca = clf.transform(text_tfidf)
    hasil = kmeans.predict(text_pca)
    return hasil[0]

def nlp_plot(dfbase):
    dfbase['data'] = dfbase['data'].astype('str')
    base = dfbase['data']
    v = TfidfVectorizer()
    x = v.fit_transform(base)
    clf = TruncatedSVD(3)
    Xpca = clf.fit_transform(x)
    return Xpca

# Load dataset
df = pd.read_csv('nlpbase.csv')
df['data'] = df['data'].astype('str')
dfbase = df['data']
dfwc = pd.read_csv('final10.csv')

st.set_page_config(layout="wide")
t1, t2 = st.beta_columns((1,2))
with t1:
    input_text = st.text_input(label="Kata Kunci",value="")
    output = None
    if st.button("Submit"):
        output = nlp_predict(dfbase,input_text)
    st.write(f'Klaster Hasil Prediksi NLP: {output}')   
with t2:
    data = dfwc
    if output == None:
        Xpca = nlp_plot(dfwc)
        fig0=px.scatter_3d(data, x=Xpca[:, 0], y=Xpca[:, 1], z=Xpca[:, 2]
                    ,color= 'label',title='Klaster Anggaran')
    else:
        output = int(output)
        data = data[data['label'].isin([output])]
        cek = data['data']
        wordcloud = WordCloud (
                    background_color = 'white',
                    width = 800,
                    stopwords =['belanja','pendapatan','lra','transfer','dan','kepada','yang'],
                    height = 500
                        ).generate(' '.join(cek))
        fig0 = px.imshow(wordcloud,title=f'Wordcloud Klaster Anggaran {output}')
    st.plotly_chart(fig0)
