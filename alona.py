import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from PIL import Image

# load model 
import joblib

# linear programming
# import pulp
# from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

def main():
    """Alona"""
    st.title("Allocation - Outcome - Anomali")
    menu = ["Allocation","Outcome","Anomali"]
    
    # df = pd.read_csv('belanja_apbd_full.csv',sep=";",error_bad_lines=False)
    df = pd.read_excel('belanja_apbd_full.xlsx')
    lokasi = df['nama_pemda'].unique()
    kota = st.selectbox('Pilih Daerah',lokasi)
    choice = st.sidebar.selectbox("Select Menu", menu)
    df = df[df['nama_pemda'].isin([kota])]
    df = df[df['Tahun'].isin([2019])]
    pop = df['Populasi'].sum()
    st.write(f"Total Populasi: {pop:.0f}")
    # st.write(f"Total Populasi (*2019):{pop:.0f}")
        
    if choice == "Allocation":
        # st.subheader("Prediction from Model")
        base = st.number_input(label="Total Anggaran",value=10000000000,min_value=0, max_value=1000000000000000, step=1)
        perkapita = st.write(f"Anggaran Perkapita: {base/pop:.2f}")
        st.write("Prediksi Perubahan Tingkat IPM Berdasarkan Alokasi Anggaran")

        model= open("ipm_gb.pkl", "rb")
        ipm=joblib.load(model)
        #Loading images

        st.sidebar.title("Alokasi Anggaran ( % )")
        #Intializing
        a = st.sidebar.number_input(label="Ekonomi",value=10,min_value=0, max_value=100, step=1)
        b = st.sidebar.number_input(label="Kesehatan",value=10,min_value=0, max_value=100, step=1)
        c = st.sidebar.number_input(label="Ketertiban",value=10,min_value=0, max_value=100, step=1)
        d = st.sidebar.number_input(label="Lingkungan",value=10,min_value=0, max_value=100, step=1)
        e = st.sidebar.number_input(label="Pariwisata Budaya",value=10,min_value=0, max_value=100, step=1)
        f = st.sidebar.number_input(label="Pelayanan",value=10,min_value=0, max_value=100, step=1)
        g = st.sidebar.number_input(label="Pendidikan",value=20,min_value=0, max_value=100, step=1)
        h = st.sidebar.number_input(label="Sosial",value=10,min_value=0, max_value=100, step=1)
        i = st.sidebar.number_input(label="Rumah_Fasum",value=10,min_value=0, max_value=100, step=1)
        j = st.sidebar.number_input(label="Base_IPM",value=60.0,min_value=0.0, max_value=100.0, step=1.0)

        if (a+b+c+d+e+f+g+h+i) >100:
            st.subheader("Total Alokasi Melebihi 100 persen anggaran")
        elif (a+b+c+d+e+f+g+h+i) <100:
            st.subheader("Total Alokasi Belum mencapai 100 persen anggaran")
        else:
            st.subheader("Anggaran sudah teralokasikan sepenuhnya")
        
        if st.button("Klik disini untuk mendapatkan Nilai Prediksi"):
            a = (base/pop)*(a/100)
            b = (base/pop)*(b/100)
            c = (base/pop)*(c/100)
            d = (base/pop)*(d/100)
            e = (base/pop)*(e/100)
            f = (base/pop)*(f/100)
            g = (base/pop)*(g/100)
            h = (base/pop)*(h/100)
            i = (base/pop)*(i/100)
            dfvalues = pd.DataFrame(list(zip([a],[b],[c],[d],[e],[f],[g],[h],[i],[j])),columns =[
                    'Ekonomip', 
                  'Kesehatanp', 
                  'Ketertibanp', 
                  'Lingkunganp', 
                  'ParBudp', 
                  'Pelayananp', 
                  'Pendidikanp', 
                  'Sosialp',
                  'Rumah_Fasump',
                  'Base_IPM',])
            input_variables = np.array(dfvalues[[
                    'Ekonomip', 
                  'Kesehatanp', 
                  'Ketertibanp', 
                  'Lingkunganp', 
                  'ParBudp', 
                  'Pelayananp', 
                  'Pendidikanp', 
                  'Sosialp',
                  'Rumah_Fasump',
                  'Base_IPM']])
            prediction = ipm.predict(input_variables)
            nilai = 100*float(prediction)
            st.title(f"Prediksi IPM: {nilai:.2f}")
            st.subheader(f"Perubahan IPM: {(nilai-j):.2f}")

if __name__=='__main__':
    main()