# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 
import plotly.graph_objects as go

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

# Utils
import joblib 
pipe_lr = joblib.load(open("modelnlp.pkl","rb"))
pipe_ctm = joblib.load(open("model_custom.pkl","rb"))

import tweepy
api_key = st.secrets(api_key)
# api_key = "l5FGlSMhD3FOB1phnwB7I9sX5"
api_secret_key = st.secrets(api_secret_key)
# api_secret_key = "R55gay8XG4uz1VGns8BT87zzXBGftNxPMaS9nvUVOzRI8YNsP1"
access_token = st.secrets(access_token)
# access_token = "237213820-RbW5PBW76TqcbT1tiAGjdkiMV7LlPnRIb9oDHixg"
access_token_secret = st.secrets(access_token_secret)
# access_token_secret = "EkHdik9UpmPB8CP8g3kSip0RC30LqgSRdkuGrovUnNEyN"
auth = tweepy.OAuthHandler(api_key,api_secret_key)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

# Image
from PIL import Image

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

# Main Application
def main():
    st.title("Machine Learning Web Application")
    menu = ["Sentiment","EDA","DataViz","Story","Classification","Timeseries"]
    choice = st.sidebar.selectbox("Select Menu", menu)
    if choice == "EDA":
        data = st.file_uploader("Upload Dataset", type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
        else:
            st.write("No Dataset To Show")
        st.subheader("Exploratory Data Analysis")
        if data is not None:
            if st.checkbox("Show Shape"):
                st.write(df.shape)
            if st.checkbox("Show Summary"):
                st.write(df.describe())
            if st.checkbox("Correlation Matrix"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()
            
    elif choice == "DataViz":
        data = st.file_uploader("Upload Dataset", type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
        else:
            st.write("No Dataset To Show")
        st.subheader("Data Visualization")
        if data is not None:
            all_columns = df.columns.to_list()
            if st.checkbox("Pie Chart"):
                columns_to_plot = st.selectbox("Select 1 Column to Visualize", all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()
            plot_type = st.selectbox("Select Type of Plot",["bar","line","area","hist","box"])
            selected_columns = st.multiselect("Select Columns To Visualize", all_columns)
            if st.button("Generate Plot"):
                st.success("Generating Custom Plot of {} for {}".format(plot_type,selected_columns))
                if plot_type == "bar":
                    cust_data = df[selected_columns]
                    st.bar_chart(cust_data)
                elif plot_type == "line":
                    cust_data = df[selected_columns]
                    st.line_chart(cust_data)
                elif plot_type == "area":
                    cust_data = df[selected_columns]
                    st.area_chart(cust_data)
                elif plot_type:
                    cust_plot = df[selected_columns].plot(kind=plot_type)
                    st.write(cust_plot)
                    st.pyplot()
    elif choice == "Story":
        data = st.file_uploader("Upload Dataset", type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.subheader("Storytelling with Data")
            if data.name == "gapminder.csv":
                fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
                # fig = px.scatter(px.data.gapminder(), x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
                size="pop", color="country", hover_name="country", log_x = True, 
                size_max=100, range_x=[100,100000], range_y=[25,90])
                fig.update_layout(height=650)
                st.write(fig)
            elif data.name == "stocks.csv":
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['date'], y=df['AAPL'], name="Apple"))
                fig.add_trace(go.Scatter(x=df['date'], y=df['AMZN'], name="Amazon"))
                fig.add_trace(go.Scatter(x=df['date'], y=df['FB'], name="Facebook"))
                fig.add_trace(go.Scatter(x=df['date'], y=df['GOOG'], name="Google"))
                fig.add_trace(go.Scatter(x=df['date'], y=df['NFLX'], name="Netflix"))
                fig.add_trace(go.Scatter(x=df['date'], y=df['MSFT'], name="Microsoft"))
                fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
                st.write(fig)
            elif data.name == "iris.csv":
                y1= df['sepal_length']
                x1= df['sepal_width']
                y2= df['petal_length']
                x2= df['petal_width']
                color = df['species']
                fig1 = px.scatter(df, x=x1, y=y1, color=color, marginal_y="violin",
                        marginal_x="box", trendline="ols", template="simple_white")
                fig2 = px.density_contour(df,x=x2,y=y2, color= color, marginal_y='histogram')
                st.write(fig1,fig2)
#         else:
# 		"No data to show"
#             st.subheader("Storytelling with Data")
# 	    story = ["gapminder","stocks","iris"]
# 	    selectedstory = st.sidebar.selectbox("Select Menu", story)
#             if selectedstory == "gapminder":
#                 fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
#                 # fig = px.scatter(px.data.gapminder(), x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
#                 size="pop", color="country", hover_name="country", log_x = True, 
#                 size_max=100, range_x=[100,100000], range_y=[25,90])
#                 fig.update_layout(height=650)
#                 st.write(fig)
#             elif selectedstory == "stocks":
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(x=df['date'], y=df['AAPL'], name="Apple"))
#                 fig.add_trace(go.Scatter(x=df['date'], y=df['AMZN'], name="Amazon"))
#                 fig.add_trace(go.Scatter(x=df['date'], y=df['FB'], name="Facebook"))
#                 fig.add_trace(go.Scatter(x=df['date'], y=df['GOOG'], name="Google"))
#                 fig.add_trace(go.Scatter(x=df['date'], y=df['NFLX'], name="Netflix"))
#                 fig.add_trace(go.Scatter(x=df['date'], y=df['MSFT'], name="Microsoft"))
#                 fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
#                 st.write(fig)
#             elif selectedstory == "iris":
#                 y1= df['sepal_length']
#                 x1= df['sepal_width']
#                 y2= df['petal_length']
#                 x2= df['petal_width']
#                 color = df['species']
#                 fig1 = px.scatter(df, x=x1, y=y1, color=color, marginal_y="violin",
#                         marginal_x="box", trendline="ols", template="simple_white")
#                 fig2 = px.density_contour(df,x=x2,y=y2, color= color, marginal_y='histogram')
#                 st.write(fig1,fig2)
        
    elif choice == "Classification":
        st.subheader("Classification Prediction")
        # if data is None:
        #     pass
        # elif data.name == "iris.csv":
        #     st.subheader("Iris flower Prediction from Machine Learning Model")
        iris= Image.open('iris.png')
        st.image(iris)

        model= open("model.pkl", "rb")
        knn_clf=joblib.load(model)
        #Loading images
        setosa= Image.open('setosa.png')
        versicolor= Image.open('versicolor.png')
        virginica = Image.open('virginica.png')

        st.sidebar.title("Features")
        #Intializing
        sl = st.sidebar.slider(label="Sepal Length (cm)",value=5.2,min_value=0.0, max_value=8.0, step=0.1)
        sw = st.sidebar.slider(label="Sepal Width (cm)",value=3.2,min_value=0.0, max_value=8.0, step=0.1)
        pl = st.sidebar.slider(label="Petal Length (cm)",value=4.2,min_value=0.0, max_value=8.0, step=0.1)
        pw = st.sidebar.slider(label="Petal Width (cm)",value=1.2,min_value=0.0, max_value=8.0, step=0.1)

        if st.button("Click Here to Classify"):
            dfvalues = pd.DataFrame(list(zip([sl],[sw],[pl],[pw])),columns =['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            input_variables = np.array(dfvalues[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
            prediction = knn_clf.predict(input_variables)
            if prediction == 1:
                st.image(setosa)
            elif prediction == 2:
                st.image(versicolor)
            elif prediction == 3:
                st.image(virginica)

    # st.title("Emotion Classifier App")
    # menu = ["Home"]
    # choice = st.sidebar.selectbox("Menu",menu)
    # create_page_visited_table()
    # create_emotionclf_table()
    elif choice == "Sentiment":
        # add_page_visited_details("Home",datetime.now())
        # data = " "
        st.subheader("Sentiment-Emotion Prediction")

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
       
    elif choice == "Timeseries":
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
            model = ExponentialSmoothing(data[col], trend=combs[best_aic][0], seasonal=combs[best_aic][1], seasonal_periods=seasonal, damped_trend=combs[best_aic][2])

            # output
            fit = model.fit()
            return fit

        st.subheader("Time Series Prediction")
        dt = st.file_uploader("Upload Dataset", type=["xlsx"])
        if dt is None:
            kolom = ['Pendapatan','Belanja']
            df = pd.read_excel('APBD Series.xlsx')
        else:
            kolom = dt.columns.tolist()
            df = pd.read_excel(dt)

        data = df.dropna()
        data['yyyy-mm'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m')
        st.write(data.head())
        pilih = st.selectbox('Pilih Kolom',kolom)
        xaxis = data.iloc[:,0].astype('str')
        fig1 = px.line(x=xaxis,y =data[pilih])
        st.plotly_chart(fig1)
        seasonal = st.number_input('Seasonal_periods',value=12,max_value=len(data),min_value=1,step=1)
        pred_period = st.number_input('Prediction_periods',value=6,max_value=len(data),min_value=1,step=1)
        # submit_data = st.form_submit_button(label='Create_model')
        if st.button('Create_model and Run_Prediction'):
            st.success("Create Model Success")
            tsmodel = create_model(pilih,seasonal)
            prediksi = list(tsmodel.forecast(pred_period))
            yaxis = data[pilih].tolist()
            # st.write(prediksi)
            for p in prediksi:
                yaxis.append(p)
            last = df.index[-1]
            dfnew = df.drop(df.index[len(yaxis):last+1])
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
