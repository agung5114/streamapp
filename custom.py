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

"""ML App with Streamlit"""
st.title("Analytics Sample-Application")
menu = ["EDA","DataViz","Sample Story","Sample Prediction"]
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
        all_columns = df.columns.unique().to_list()
        # if st.checkbox("Pie Chart"):
        #     columns_to_plot = st.selectbox("Select 1 Column to Visualize", all_columns)
        #     pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
        #     st.write(pie_plot)
        #     st.pyplot()
        plot_type = st.selectbox("Select Type of Plot",["pie","bar","line","area","hist","box"])
        selected_columns = st.multiselect("Select Columns To Visualize", all_columns)
        if st.button("Generate Plot"):
            st.success("Generating Custom Plot of {} for {}".format(plot_type,selected_columns))
            if plot_type == "pie":
                # columns_to_plot = st.selectbox("Select 1 Column to Visualize", all_columns)
                pie_plot = df[selected_columns].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()
            elif plot_type == "bar":
                cust_data = df[selected_columns]
                st.bar_chart(cust_data)
            elif plot_type == "line":
                cust_data = df[selected_columns]
                st.line_chart(cust_data)
            elif plot_type == "area":
                cust_data = df[selected_columns]
                st.area_chart(cust_data)
            elif plot_type:
                selected_columns = st.multiselect("Select Columns To Visualize", all_columns)
                cust_plot = df[selected_columns].plot(kind=plot_type)
                st.write(cust_plot)
                st.pyplot()
elif choice == "Sample Story":
    st.subheader("Storytelling with Data")
    ops = st.selectbox("Select Sample Data",["","gapminder","stocks","iris"])
    if ops == "":
        pass
    elif ops == "gapminder":
        df = pd.read_csv('datasets/gapminder.csv')
        fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
        # fig = px.scatter(px.data.gapminder(), x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
        size="pop", color="country", hover_name="country", log_x = True, 
        size_max=100, range_x=[100,100000], range_y=[25,90])
        fig.update_layout(height=650)
        st.write(fig)
    elif ops == "stocks":
        df = pd.read_csv("datasets/stocks.csv")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['AAPL'], name="Apple"))
        fig.add_trace(go.Scatter(x=df['date'], y=df['AMZN'], name="Amazon"))
        fig.add_trace(go.Scatter(x=df['date'], y=df['FB'], name="Facebook"))
        fig.add_trace(go.Scatter(x=df['date'], y=df['GOOG'], name="Google"))
        fig.add_trace(go.Scatter(x=df['date'], y=df['NFLX'], name="Netflix"))
        fig.add_trace(go.Scatter(x=df['date'], y=df['MSFT'], name="Microsoft"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.write(fig)
    elif ops == "iris":
        df = pd.read_csv("datasets/iris.csv")
        y1= df['sepal_length']
        x1= df['sepal_width']
        y2= df['petal_length']
        x2= df['petal_width']
        color = df['species']
        fig1 = px.scatter(df, x=x1, y=y1, color=color, marginal_y="violin",
                marginal_x="box", trendline="ols", template="simple_white")
        fig2 = px.density_contour(df,x=x2,y=y2, color= color, marginal_y='histogram')
        st.write(fig1,fig2)
    
elif choice == "Sample Prediction":
    st.subheader("Prediction from Model")
    model= open("model.pkl", "rb")
    knn_clf=joblib.load(model)
    st.title("Iris flower species Classification App")

    #Loading images
    setosa= Image.open('setosa.png')
    versicolor= Image.open('versicolor.png')
    virginica = Image.open('virginica.png')

    st.sidebar.title("Features")
    #Intializing
    parameter_list=['Sepal length (cm)','Sepal Width (cm)','Petal length (cm)','Petal Width (cm)']
    parameter_input_values=[]
    parameter_default_values=['5.2','3.2','4.2','1.2']
    values=[]

    for parameter, parameter_df in zip(parameter_list, parameter_default_values):
        values= st.sidebar.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
        parameter_input_values.append(values)
        
    input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
    st.write('\n\n')

    if st.button("Click Here to Classify"):
        prediction = knn_clf.predict(input_variables)
        st.image(setosa) if prediction == 0 else st.image(versicolor)  if prediction == 1 else st.image(virginica)
