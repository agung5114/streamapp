import streamlit as st
import plotly_express as px

def main():
    """App with Streamlit"""
    st.title("Hello Data Analyst!")

    df = px.data.gapminder().query("country == 'Indonesia' ")
    # fig = px.bar(df, y='pop', x='year', color='pop')

    # fig.update_layout(template='plotly_dark',title={'text':'Population Growth of Indonesia'},
    #                     xaxis_title='year',yaxis_title='population')
    # st.write(fig)

    import pandas as pd
    df2 = pd.read_csv("./datasets/gapminder.csv")
    if st.checkbox("Show Summary"):
        st.write(df.describe())
    
    st.subheader("Storytelling with Data")
    fig2 = px.scatter(df2, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
    size="pop", color="country", hover_name="country", log_x = True, 
    size_max=100, range_x=[100,100000], range_y=[25,90])
    fig2.update_layout(height=650)
    st.write(fig2)

if __name__=='__main__':
    main()
