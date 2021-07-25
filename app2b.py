# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 
pipe_lr = joblib.load(open("modelnlp.pkl","rb"))
pipe_ctm = joblib.load(open("model_custom.pkl","rb"))

import tweepy
# api key
api_key = "l5FGlSMhD3FOB1phnwB7I9sX5"
# api secret key
api_secret_key = "R55gay8XG4uz1VGns8BT87zzXBGftNxPMaS9nvUVOzRI8YNsP1"
# access token
access_token = "237213820-RbW5PBW76TqcbT1tiAGjdkiMV7LlPnRIb9oDHixg"
# access token secret
access_token_secret = "EkHdik9UpmPB8CP8g3kSip0RC30LqgSRdkuGrovUnNEyN"
auth = tweepy.OAuthHandler(api_key,api_secret_key)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

# Fxn
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

def predict_sentiment(docx):
    results = pipe_ctm.predict([docx])
    return results[0]

def get_sentiment_proba(docx):
	results = pipe_ctm.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}
emoji_sentiment = {"positif":"🤗","negatif":"😔","netral":"😐","tdk-relevan":"😮"}

# Timeseries model
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def create_model(col,seasonal):
    col = str(col)
    tr = ['add', 'mul']
    ss = ['add', 'mul']
    dp = [True, False]
    combs = {}
    aics = []

    # iterasi kombinasi option
    for i in tr:
        for j in ss:
            for k in dp:
                model = ExponentialSmoothing(data[col], trend=i, seasonal=j, seasonal_periods=seasonal, damped=k)
                model = model.fit()
                combs.update({model.aic : [i, j, k]})
                aics.append(model.aic)

    # forecasting dengan kombinasi terbaik            
    best_aic = min(aics)
    model = ExponentialSmoothing(data[col], trend=combs[best_aic][0], seasonal=combs[best_aic][1], seasonal_periods=12, damped=combs[best_aic][2])
            
    # output
    fit = model.fit()
    return fit

# Main Application
def main():
    # st.title("Machine Learning Analytics App")
    menu = ["Sentiment","TimeSeries"]
    choice = st.sidebar.selectbox("Menu",menu)
    # create_page_visited_table()
    # create_emotionclf_table()
    if choice == "Sentiment":
        # add_page_visited_details("Home",datetime.now())
        st.subheader("Sentiment & Emotion Prediction")

        with st.form(key='emotion_clf_form'):
            search_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:	
            hasilSearch = api.search(q=str(search_text),count=2)
            texts = []
            for tweet in hasilSearch:
                texts.append(tweet.text)
            # raw_text2 = texts[1]
            raw_text = texts[0]
            # translated = translator.translate(raw_text)
            # translated = raw_text
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            sentiment = predict_sentiment(raw_text)
            proba_sentiment = get_sentiment_proba(raw_text)
            col1,col2  = st.beta_columns(2)
            with col1:
                st.success("Search Result")
                st.write(raw_text)
                # st.write(raw_text2)
                # st.write(translated.text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
                st.write("{}:{}".format(sentiment,emoji_sentiment[sentiment]))
                st.write("Confidence:{}".format(np.max(proba_sentiment)))
            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                # st.write(proba_sentiment)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                proba_sent_df = pd.DataFrame(proba_sentiment,columns=pipe_ctm.classes_)
                # st.write(proba_df.T)
                # st.write(proba_sent_df.T)
                # proba_df_clean = proba_df.T.reset_index()
                # proba_df_clean.columns = ["emotions","probability"]
                proba_df_sent_clean = proba_sent_df.T.reset_index()
                proba_df_sent_clean.columns = ["sentiments","probability"]

                # fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                # st.altair_chart(fig,use_container_width=True)
                fig = alt.Chart(proba_df_sent_clean).mark_bar().encode(x='sentiments',y='probability',color='sentiments')
                st.altair_chart(fig,use_container_width=True)

    elif choice == "TimeSeries":
        import plotly.graph_objects as go
        import plotly.io as pio
        pio.templates.default = "seaborn"
        # Timeseries model
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        def create_model(col,seasonal):
            col = str(col)
            tr = ['add', 'mul']
            ss = ['add', 'mul']
            dp = [True, False]
            combs = {}
            aics = []

            # iterasi kombinasi option
            for i in tr:
                for j in ss:
                    for k in dp:
                        model = ExponentialSmoothing(data[col], trend=i, seasonal=j, seasonal_periods=seasonal, damped_trend=k)
                        model = model.fit()
                        combs.update({model.aic : [i, j, k]})
                        aics.append(model.aic)

            # forecasting dengan kombinasi terbaik            
            best_aic = min(aics)
            model = ExponentialSmoothing(data[col], trend=combs[best_aic][0], seasonal=combs[best_aic][1], seasonal_periods=12, damped_trend=combs[best_aic][2])
                    
            # output
            fit = model.fit()
            return fit

        st.subheader("Time Series Prediction")
        data = st.file_uploader("Upload Dataset", type=["xlsx"])
        kolom = []
        if data is None:
            st.write('Please upload timeseries file (xlsx)')
        else:
            df = pd.read_excel(data)
            data = df.dropna()
            st.write(data.head())
            kolom = data.columns.tolist()
            pilih = st.selectbox('Pilih Kolom',kolom)
            xaxis = data.iloc[:,0].astype('str')
            fig1 = px.line(x=xaxis,y =data[pilih])
            st.plotly_chart(fig1)
            seasonal = st.number_input('Seasonal_periods',value=12,max_value=48,min_value=1,step=1)
            pred_period = st.number_input('Prediction_periods',value=6,max_value=48,min_value=1,step=1)
            # submit_data = st.form_submit_button(label='Create_model')
            if st.button('Create_model and Run_Prediction'):
                st.success("Create Model Success")
                tsmodel = create_model(pilih,seasonal)
                prediksi = list(tsmodel.forecast(pred_period))
                yaxis = data[pilih].tolist()
                # st.write(prediksi)
                for p in prediksi:
                    yaxis.append(p)
                # st.write(yaxis)
                dfnew = df.drop(df.index[len(yaxis):84])
                dfnew['prediction'] = yaxis
                dfnew.iloc[:,0] = dfnew.iloc[:,0].astype('str')
                # dfnew = dfnew.dropna()
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=dfnew.iloc[:,0], y=dfnew['prediction'],
                                # line = dict(color='firebrick', width=4, dash='dot'),
                                mode='lines+markers',
                                name='prediction'))
                fig2.add_trace(go.Scatter(x=dfnew.iloc[:,0], y=dfnew[pilih],
                                # line = dict(color='firebrick', width=4, dash='dot'),
                                mode='lines+markers',
                                name='actual'))
                st.plotly_chart(fig2)



if __name__ == '__main__':
	main()