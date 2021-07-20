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
import locale
# locale.setlocale(locale.LC_ALL, 'en_us')
float_formatter = "{:.0f}".format

st.set_page_config(layout="wide")
st.title("Revenue Prediction based on Efficiency Model")
        
# df = pd.read_excel('DEA_ML.xlsx')
dfs = pd.read_excel('DEA_ML.xlsx')
# Sektor = df.Sektor.unique().tolist()
sector = ["Financials",
        "Infrastructures",
        "Consumer Non-Cyclicals",
        "Healthcare",
        "Properties & Real Estate",
        "Industrials",
        "Energy",
        "Basic Materials",
        "Consumer Cyclicals",
        "Technology",
        "Trasportation & Logistic"]

choice = st.sidebar.selectbox("Pilih Sektor", sector)
# 
# if choice == "Financials":
#     df = pd.read_excel('DEA_ML_Finc.xlsx')
#     model= open("dea_rf_finc.pkl", "rb")
# elif choice == "Infrastructures":
#     df = pd.read_excel('DEA_ML_Inft.xlsx')
#     model= open("dea_tree_inft.pkl", "rb")
# elif choice == "Consumer Non-Cyclicals":
#     df = pd.read_excel('DEA_ML_CNCy.xlsx')
#     model= open("dea_xbst_cncy.pkl", "rb")
# elif choice == "Healthcare":
#     df = pd.read_excel('DEA_ML_Hlth.xlsx')
#     model= open("dea_tree_hlth.pkl", "rb")
# elif choice == "Properties & Real Estate":
#     df = pd.read_excel('DEA_ML_Prop.xlsx')
#     model= open("dea_lbst_prop.pkl", "rb")
# elif choice == "Industrials":
#     df = pd.read_excel('DEA_ML_Inds.xlsx')
#     model= open("dea_xbst_inds.pkl", "rb")
# elif choice == "Energy":
#     df = pd.read_excel('DEA_ML_Enrg.xlsx')
#     model= open("dea_xbst_enrg.pkl", "rb")
# elif choice == "Basic Materials":
#     df = pd.read_excel('DEA_ML_BasM.xlsx')
#     model= open("dea_gb_basm.pkl", "rb")
# elif choice == "Consumer Cyclicals":
#     df = pd.read_excel('DEA_ML_CCyc.xlsx')
#     model= open("dea_rf_ccyc.pkl", "rb")
# elif choice == "Technology":
#     df = pd.read_excel('DEA_ML_Tech.xlsx')
#     model= open("dea_rf_tech.pkl", "rb")
# elif choice == "Trasportation & Logistic":
#     df = pd.read_excel('DEA_ML_Tran.xlsx')
#     model= open("dea_stack_tran.pkl", "rb")
df = dfs
# df = df[df['Sektor'].isin([choice])]
top = df['Efficiency'].max()
# filter frontier
topdf = df[df['Efficiency']>=top]

top_umum = float(topdf['Beban Umum dan Administrasi'].sum()/topdf['Beban Umum dan Administrasi'].count())
top_jual = float(topdf['Beban Penjualan'].sum()/topdf['Beban Penjualan'].count())
top_lain = float(topdf['Beban Lainnya'].sum()/topdf['Beban Lainnya'].count())

st.sidebar.subheader("Alokasi Beban (Rata-rata Frontier)")
# st.sidebar.subheader(df['DMU'].values)
sect1 = st.sidebar.number_input(label="Beban Umum Administrasi (%)",value=100*float(top_umum))
sect2 = st.sidebar.number_input(label="Beban Penjualan (%)",value=100*float(top_jual))
sect3 = st.sidebar.number_input(label="Beban Lainnya (%)",value=100*float(top_lain))

av_umum = float(df['Beban Umum dan Administrasi'].sum()/df['Beban Umum dan Administrasi'].count())
av_jual = float(df['Beban Penjualan'].sum()/df['Beban Penjualan'].count())
av_lain = float(df['Beban Lainnya'].sum()/df['Beban Lainnya'].count())

st.sidebar.subheader("Alokasi Beban (Rata-rata sektor)")
sect1 = st.sidebar.number_input(label="Beban Umum Administrasi (%) ",value=100*float(av_umum))
sect2 = st.sidebar.number_input(label="Beban Penjualan (%) ",value=100*float(av_jual))
sect3 = st.sidebar.number_input(label="Beban Lainnya (%) ",value=100*float(av_lain))

# dfs = pd.read_excel('DEA_ML.xlsx')
dfs = dfs[dfs['Sektor'].isin([choice])]
emiten = st.selectbox('Pilih Kode Emiten',["All"]+dfs.Nama_emiten.unique().tolist())

if emiten == "All":
    dfs = dfs
else:
    dfs = dfs[dfs['Nama_emiten'].isin([emiten])]

pend = int(dfs.Pendapatan.sum()/dfs.Pendapatan.count())
rev_val = locale.format_string('%.0f', int(dfs.Pendapatan.sum()/dfs.Pendapatan.count()),True)
cogs_val = locale.format_string('%.0f', int(dfs.HPP_nom.sum()/dfs.HPP_nom.count()),True)
exp_val = locale.format_string('%.0f', int(dfs.Beban_nom.sum()/dfs.Beban_nom.count()),True)

c1 ,c2= st.beta_columns((1,1))
with c1:
    revenue = st.text_input(label="Total Pendapatan (Rp)",value=rev_val)
#     revenue = st.text_input(label="Total Pendapatan (Rp)",value=f"Prediksi Index: {pend:.2f}")
    cogs = st.text_input(label="Total HPP (Rp)",value=cogs_val)
    expense = st.text_input(label="Total Beban (Rp)",value=exp_val)
with c2:
    eps_g = st.number_input(label="EPS Growth (%)",value=float(100*dfs['EPS_growth'].sum()/dfs['EPS_growth'].count()))
    eps_q = st.number_input(label="ROE (%)",value=100*dfs['ROE'].sum()/dfs['ROE'].count())
    eff = st.number_input(label="Efficiency Score (%)",value=float(100*dfs['Efficiency'].sum()/dfs['Efficiency'].count()))

umum = float(dfs['Beban Umum dan Administrasi'].sum()/dfs['Beban Umum dan Administrasi'].count())
jual = float(dfs['Beban Penjualan'].sum()/dfs['Beban Penjualan'].count())
lain = float(dfs['Beban Lainnya'].sum()/dfs['Beban Lainnya'].count())

st.write(" ")

p1 ,p2,p3,p4= st.beta_columns((1,1,1,1))
with p1:
    st.subheader("Alokasi Beban Saat ini")
    sect1 = st.number_input(label="Umum Administrasi (%)",value=100*float(umum))
    sect2 = st.number_input(label="Penjualan (%)",value=100*float(jual))
    sect3 = st.number_input(label="Lainnya (%)",value=100*float(lain))
with p2:
    st.subheader("Selisih dengan Frontier")
    gap1 = st.number_input(label="Umum Administrasi (%) ",value=100*float(top_umum)-100*float(umum))
    gap2 = st.number_input(label="Penjualan (%) ",value=100*float(top_jual)-100*float(jual))
    gap3 = st.number_input(label="Lainnya (%) ",value=100*float(top_lain)-100*float(lain))
with p3:
    st.subheader("Selisih dengan Rata-rata")
    sect1 = st.number_input(label="Umum Administrasi (%)  ",value=100*float(av_umum)-100*float(umum))
    sect2 = st.number_input(label="Penjualan (%)  ",value=100*float(av_jual)-100*float(jual))
    sect3 = st.number_input(label="Lainnya (%)  ",value=100*float(av_lain)-100*float(lain))
with p4:
    st.subheader("Alokasi untuk diprediksi")
    test1 = st.number_input(label="Umum Administrasi (%)   ",value=100*float(umum))
    test2 = st.number_input(label="Penjualan (%)   ",value=100*float(jual))
    test3 = st.number_input(label="Lainnya (%)   ",value=100*float(lain))

dmu = dfs['DMU'].tolist()

if emiten == "All":
    dfs = dfs
else:
    dfs = dfs[dfs['DMU'].isin([dmu])]
    dfs['Beban Umum dan Administrasi'] = [test1/100]
    dfs['Beban Penjualan'] = [test2/100]
    dfs['Beban Lainnya'] = [test3/100]

# df = pd.get_dummies(df, drop_first=True)
# import re
# df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# df = df.loc[:,~df.columns.duplicated()]
# df = df.dropna()
# df = df.drop('Efficiency', 1)
modelnew = open("dea_all_xg.pkl", "rb")
pkl=joblib.load(modelnew)
# pkl =  model
st.write(" ")
if st.button("Klik disini untuk mendapatkan Nilai Prediksi Pendapatan"):
    if emiten == "All":
        st.write('Silakan Pilih Emiten')
    else:
        dfsx = pd.DataFrame(dfs,columns =['Beban Lainnya','Beban Umum dan Administrasi','Beban Penjualan','Beban_nom','Q_EPS','Sektor_no'
                ])
        input_variables = np.array(dfsx[[
                        'Beban Lainnya','Beban Umum dan Administrasi','Beban Penjualan','Beban_nom','Q_EPS','Sektor_no'
                        ]])
        # input_variables = df.to_numpy()
        # input_variables = input_variables.reshape(-1, 1)
        prediction = pkl.predict(input_variables)
        nilai = locale.format_string('%.0f', int(prediction),True)
        st.subheader('Nilai Prediksi')
        st.title(nilai)
    # st.title(f"Prediksi Index: {nilai:.0f}")
    # st.subheader(f"Perubahan Index: {(nilai-rev_val):.0f}")
