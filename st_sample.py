import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from PIL import Image

# load model 
import joblib

# linear programming
import pulp
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

def main():
    """App with Streamlit"""
    st.title("Hello Data Analyst!")
    menu = ["Predictive","Prescriptive"]
    
    choice = st.sidebar.selectbox("Select Menu", menu)
        
    if choice == "Predictive":
        # st.subheader("Prediction from Model")
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
    
    elif choice == "Prescriptive":
        st.subheader("Blue Ridge Prescriptive Linear Programming Model")
        st.sidebar.title("Constraints")
        p = st.sidebar.number_input(label="Pump",value=200,min_value=0, max_value=1000, step=10)
        l= st.sidebar.number_input(label="Labor",value=1566,min_value=0, max_value=5000, step=100)
        t = st.sidebar.number_input(label="Tubing",value=2880,min_value=0, max_value=5000, step=100)
        
        # Create the model
        prob = LpProblem(name="Blue_Ridge_LP_Problem",sense=LpMaximize)
        # Initialize the decision variables
        x1 = LpVariable(name="aqua", lowBound=0, cat="Integer")
        x2 = LpVariable(name="hydro", lowBound=0, cat="Integer")
        # Add the constraints to the model
        prob += (x1 + x2 <= p, "pump_constraint")
        prob += (9*x1 + 6*x2 <= l, "labor_constraint")
        prob += (12*x1 + 16*x2 <= t, "tubing_constraint")
        # Add the objective function to the model
        prob += 350*x1 + 300*x2
        # Solve the problem
        st.write(" How many products to produce to maximize Profit?")
        if st.button("Click Here to Solve"):
            status = prob.solve()
            st.write(f"Aqua-Spas: {pulp.value(x1):.0f}")
            st.write(f"Hydro-Luxes: {pulp.value(x2):.0f}")
            st.write(f"Profit: $ {(350*pulp.value(x1)+300*pulp.value(x2)):.0f}")


if __name__=='__main__':
    main()