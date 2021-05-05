import streamlit as st

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
st.set_option('deprecation.showPyplotGlobalUse', False)

import seaborn as sns

import plotly_express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots

import joblib
from PIL import Image
import vaex
import dask.dataframe as dd

def main():
    """ML App with Streamlit"""
    st.title("Machine Learning Web Application")
    menu = ["EDA","DataViz","Story","Prediction"]
    data = st.file_uploader("Upload Dataset", type=["csv","txt"])
    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df.head())
    else:
        st.write("No Dataset To Show")
    choice = st.sidebar.selectbox("Select Menu", menu)
    if choice == "EDA":
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
        
    elif choice == "Prediction":
        st.subheader("Prediction from Model")
        if data is None:
            pass
        elif data.name == "iris.csv":
            st.subheader("Iris flower Prediction from Machine Learning Model")
        iris= Image.open('iris.png')

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

if __name__=='__main__':
    main()