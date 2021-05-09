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

# def main():
    # """Alona"""
st.title("Allocation - Outcome - Anomali")
menu = ["IPM","AHH","HLS","RLS","PPK"]
choice = st.sidebar.selectbox("Select Menu", menu)

if choice == "IPM":
    st.subheader("Indeks Pembangunan Manusia (IPM)")
    st.write("Prediksi Perubahan Tingkat IPM Berdasarkan Alokasi Anggaran")
elif choice == "AHH":
    st.subheader("Angka Harapan Hidup Saat Lahir (AHH)")
    st.write("Prediksi Perubahan Tingkat AHH Berdasarkan Alokasi Anggaran")
elif choice == "HLS":
    st.subheader("Angka Harapan Lama Sekolah (HLS)")
    st.write("Prediksi Perubahan Tingkat HLS Berdasarkan Alokasi Anggaran")
elif choice == "RLS":
    st.subheader("Rata-rata Lama Sekolah (RLS)")
    st.write("Prediksi Perubahan Tingkat RLS Berdasarkan Alokasi Anggaran")
elif choice == "PPK":
    st.subheader("PEngeluaran Per Kapita Disesuaikan (PPK)")
    st.write("Prediksi Perubahan Tingkat PPK Berdasarkan Alokasi Anggaran")


# df = pd.read_csv('belanja_apbd_full.csv',sep=";",error_bad_lines=False)
if choice == "IPM":
    df = pd.read_excel('belanja_apbd_full.xlsx')
else:
    df = pd.read_excel('belanja_apbd_selisih.xlsx')

lokasi = df['nama_pemda'].unique()
kota = st.selectbox('Pilih Daerah',lokasi)
df = df[df['nama_pemda'].isin([kota])]
df = df[df['Tahun'].isin([2019])]
pop = df['Populasi'].sum()
base_ipm = df['Base_IPM'].sum()
base_ahh = df['Base_AHH'].sum()
base_hls = df['Base_HLS'].sum()
base_rls = df['Base_RLS'].sum()
base_ppk = df['Base_PPK'].sum()
st.write(f"Total Populasi: {pop:.0f} (2019)")

base = st.number_input(label="Total Anggaran",value=10000000000,min_value=0, max_value=1000000000000000, step=1)
perkapita = st.write(f"Anggaran Perkapita: {base/pop:.2f}")

#sidebar
if choice == "IPM":
    st.sidebar.title("Nilai IPM Awal")
    j = st.sidebar.number_input(label="Base_IPM",value=base_ipm,min_value=0.0, max_value=100.0, step=1.0)
elif choice == "AHH":
    st.sidebar.title("Nilai AHH Awal")
    j = st.sidebar.number_input(label="Base_AHH",value=base_ahh,min_value=0.0, max_value=100.0, step=1.0)
elif choice == "HLS":
    st.sidebar.title("Nilai HLS Awal")
    j = st.sidebar.number_input(label="Base_HLS",value=base_hls,min_value=0.0, max_value=100.0, step=1.0)
elif choice == "RLS":
    st.sidebar.title("Nilai RLS Awal")
    j = st.sidebar.number_input(label="Base_RLS",value=base_rls,min_value=0.0, max_value=100.0, step=1.0)
elif choice == "PPK":
    st.sidebar.title("Nilai PPK Awal")
    j = st.sidebar.number_input(label="Base_PPK",value=base_ppk,min_value=0, max_value=30000, step=100)

st.sidebar.title("Alokasi Anggaran ( % )")
a = st.sidebar.number_input(label="Ekonomi",value=10,min_value=0, max_value=100, step=1)
b = st.sidebar.number_input(label="Kesehatan",value=10,min_value=0, max_value=100, step=1)
c = st.sidebar.number_input(label="Ketertiban",value=10,min_value=0, max_value=100, step=1)
d = st.sidebar.number_input(label="Lingkungan",value=10,min_value=0, max_value=100, step=1)
e = st.sidebar.number_input(label="Pariwisata Budaya",value=10,min_value=0, max_value=100, step=1)
f = st.sidebar.number_input(label="Pelayanan",value=10,min_value=0, max_value=100, step=1)
g = st.sidebar.number_input(label="Pendidikan",value=20,min_value=0, max_value=100, step=1)
h = st.sidebar.number_input(label="Sosial",value=10,min_value=0, max_value=100, step=1)
i = st.sidebar.number_input(label="Rumah_Fasum",value=10,min_value=0, max_value=100, step=1)

if (a+b+c+d+e+f+g+h+i) >100:
    st.subheader("Total Alokasi Melebihi 100 persen anggaran")
elif (a+b+c+d+e+f+g+h+i) <100:
    st.subheader("Total Alokasi Belum mencapai 100 persen anggaran")
else:
    st.subheader("Anggaran sudah teralokasikan sepenuhnya")

if choice == "IPM":
    model= open("ipm_gb.pkl", "rb")
    ipm=joblib.load(model)
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
elif choice == "AHH":
    model= open("ahh_stack.pkl", "rb")
    ahh=joblib.load(model)
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
                'Base_AHH',])
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
                'Base_AHH']])
        prediction = ahh.predict(input_variables)
        nilai = 100*float(prediction)
        st.title(f"Prediksi AHH: {nilai:.2f}")
        st.subheader(f"Perubahan AHH: {(nilai-j):.2f}")
elif choice == "HLS":
    model= open("hls_knn.pkl", "rb")
    hls=joblib.load(model)
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
                'Base_HLS',])
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
                'Base_HLS']])
        prediction = hls.predict(input_variables)
        nilai = 100*float(prediction)
        st.title(f"Prediksi HLS: {nilai:.2f}")
        st.subheader(f"Perubahan HLS: {(nilai-j):.2f}")

elif choice == "RLS":
    model= open("rls_ridge.pkl", "rb")
    rls=joblib.load(model)
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
                'Base_RLS',])
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
                'Base_RLS']])
        prediction = rls.predict(input_variables)
        nilai = 100*float(prediction)
        st.title(f"Prediksi RLS: {nilai:.2f}")
        st.subheader(f"Perubahan RLS: {(nilai-j):.2f}")

elif choice == "PPK":
    model= open("ppk_stack.pkl", "rb")
    ppk=joblib.load(model)
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
                'Base_PPK',])
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
                'Base_PPK']])
        prediction = ppk.predict(input_variables)
        nilai = 100*float(prediction)
        st.title(f"Prediksi PPK: {nilai:.2f}")
        st.subheader(f"Perubahan PPK: {(nilai-j):.2f}")

# if __name__=='__main__':
#     main()