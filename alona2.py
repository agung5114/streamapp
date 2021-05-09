import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from PIL import Image
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
# load model 
import joblib
import vaex as vx

st.set_page_config(layout="wide")
st.title('ALONA: Allocation - Outcome - Anomali')
# pages = ['Home','Anomali','Allocation','Outcome']
# jump = st.sidebar.selectbox('Select Pages', pages)

menu = ['Anomali','Allocation & Outcome']
choice = st.sidebar.selectbox('Select Menu', menu)

if choice == "Allocation & Outcome":
    st.subheader("Alokasi Anggaran dan Prediksi Perubahan Index")
    df = pd.read_excel('belanja_apbd_full.xlsx')
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
    c1, c2 = st.beta_columns((1, 1))
    with c1:
        st.subheader("Alokasi Anggaran ( % )")
        a = st.number_input(label="Ekonomi",value=10,min_value=0, max_value=100, step=1)
        b = st.number_input(label="Kesehatan",value=10,min_value=0, max_value=100, step=1)
        c = st.number_input(label="Ketertiban",value=10,min_value=0, max_value=100, step=1)
        d = st.number_input(label="Lingkungan",value=10,min_value=0, max_value=100, step=1)
        e = st.number_input(label="Pariwisata Budaya",value=10,min_value=0, max_value=100, step=1)
        f = st.number_input(label="Pelayanan",value=10,min_value=0, max_value=100, step=1)
        g = st.number_input(label="Pendidikan",value=20,min_value=0, max_value=100, step=1)
        h = st.number_input(label="Sosial",value=10,min_value=0, max_value=100, step=1)
        i = st.number_input(label="Rumah_Fasum",value=10,min_value=0, max_value=100, step=1)
    with c2:
        st.subheader("Prediksi Perubahan Index")
        index = st.selectbox('Pilih Index',['IPM','AHH','HLS','RLS','PPK'])
        if index == "IPM":
            st.subheader("Nilai IPM Awal")
            j = st.number_input(label="Base_IPM",value=base_ipm,min_value=0.0, max_value=100.0, step=1.0)
        elif index == "AHH":
            st.subheader("Nilai AHH Awal")
            j = st.number_input(label="Base_AHH",value=base_ahh,min_value=0.0, max_value=100.0, step=1.0)
        elif index == "HLS":
            st.subheader("Nilai HLS Awal")
            j = st.number_input(label="Base_HLS",value=base_hls,min_value=0.0, max_value=100.0, step=1.0)
        elif index == "RLS":
            st.subheader("Nilai RLS Awal")
            j = st.number_input(label="Base_RLS",value=base_rls,min_value=0.0, max_value=100.0, step=1.0)
        elif index == "PPK":
            st.subheader("Nilai PPK Awal")
            j = st.number_input(label="Base_PPK",value=base_ppk,min_value=0, max_value=30000, step=100)

        if (a+b+c+d+e+f+g+h+i) >100:
            st.subheader("Total Alokasi Melebihi 100 persen anggaran")
        elif (a+b+c+d+e+f+g+h+i) <100:
            st.subheader("Total Alokasi Belum mencapai 100 persen anggaran")
        else:
            st.subheader("Anggaran sudah teralokasikan sepenuhnya")

        if index == "IPM":
            model= open("ipm_gb.pkl", "rb")
        elif index == "AHH":
            model= open("ahh_stack.pkl", "rb")
        elif index == "HLS":
            model= open("hls_knn.pkl", "rb")
        elif index == "RLS":
            model= open("rls_ridge.pkl", "rb")
        elif index == "PPK":
            model= open("ppk_stack.pkl", "rb")
        
        pkl=joblib.load(model)
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
            prediction = pkl.predict(input_variables)
            nilai = 100*float(prediction)
            st.title(f"Prediksi Index: {nilai:.2f}")
            st.subheader(f"Perubahan Index: {(nilai-j):.2f}")

elif choice == 'Anomali':
    st.subheader('Analisis atas Anomali dalam Alokasi Anggaran dan Tingkat Perubahan Index')
    if st.checkbox("Exploratory Data Analysis"):
        df = pd.read_csv('belanja_apbd.csv',sep=";",usecols=["nama_pemda","Tahun","Ekonomi","Kesehatan","Ketertiban","Lingkungan","ParBud","Pelayanan","Pendidikan","Sosial","Rumah_Fasum","Populasi"])
        # st.subheader('Automated Exploratory Data Analysis')
        df.loc[:,'Total'] = df.sum(numeric_only=True, axis=1)
        st.dataframe(df.head())
        if st.checkbox('Show Shape'):
            st.write(df.shape)
        if st.checkbox('Show Summary'):
            st.write(df.describe())
        if st.checkbox('Correlation Matrix'):
            st.write(sns.heatmap(df.corr(),annot=True))
            st.pyplot()
        plot_type = st.selectbox('Select Type of Plot',["bar","line","area","hist","box"])
        all_columns = ["Total","Ekonomi","Kesehatan","Ketertiban","Lingkungan","ParBud","Pelayanan","Pendidikan","Sosial","Rumah_Fasum","Populasi"]
        pemda = st.multiselect('Pilih Pemda',df['nama_pemda'])
        df["Tahun"] = df["Tahun"].astype(str)
        # tahun = st.selectbox('Pilih Tahun',[2019,2018])
        # df = df[df['Tahun'].isin([tahun])]
        selected_columns = st.selectbox('Select Column To Visualize', all_columns)
        # dimension = st.selectbox('Add Dimension', all_columns)
        if pemda == None or pemda ==[]:
            dfp = df
            color_value = dfp['nama_pemda']
            mode = 'stack'
            xbox = None
        else:
            dfp = df[df['nama_pemda'].isin(pemda)]
            color_value = dfp['nama_pemda']
            mode = 'group'
            xbox = dfp['nama_pemda']

        if plot_type == "bar":
            fig = px.bar(dfp,x='Tahun',y=selected_columns,color=color_value,barmode=mode)
            st.plotly_chart(fig)
        elif plot_type == "line":
            fig = px.line(dfp,x='Tahun',y=selected_columns,color=color_value)
            st.plotly_chart(fig)
        elif plot_type == "area":
            fig = px.area(dfp,x='Tahun',y=selected_columns,color=color_value)
        elif plot_type == "hist":
            fig = px.histogram(dfp, x=selected_columns)
            st.plotly_chart(fig)
        elif plot_type == "box":
            fig = px.box(dfp, y=selected_columns, x=xbox)
            st.plotly_chart(fig)
        elif plot_type:
            cust_plot = dfp[selected_columns].plot(kind=plot_type)
            st.write(cust_plot)
            st.pyplot()
        
    if st.checkbox("Analisis Anggaran vs Index"):
        df = pd.read_csv('belanja_apbd.csv',sep=";")
        df.loc[:,'Total_anggaran'] = df.sum(numeric_only=True, axis=1)
        st.subheader('Anomali Detection from Trends and Outliers Analysis ')
        index = st.selectbox('Pilih Index',['IPM','AHH','HLS','RLS','PPK'])
        anggaran = st.selectbox('Pilih Jenis Fungsi',["All","Ekonomi","Kesehatan","Ketertiban","Lingkungan","ParBud","Pelayanan","Pendidikan","Sosial","Rumah_Fasum"])
        pemda = st.multiselect('Pilih Pemda',df['nama_pemda'])
        c1, c2 = st.beta_columns((1, 1))
        df["Tahun"] = df["Tahun"].astype(str)
        # st.write(pemda)
        with c1:
            if pemda == None or pemda ==[]:
                dfp = df
            else:
                dfp = df[df['nama_pemda'].isin(pemda)]

            if anggaran == "All":
                fig1 = px.scatter(dfp, x='nama_pemda', y='Total_anggaran',color='Tahun')
            elif anggaran == "Ekonomi":
                fig1 = px.scatter(dfp, x='nama_pemda', y="Ekonomi",color='Tahun')
            elif anggaran == "Kesehatan":
                fig1 = px.scatter(dfp, x='nama_pemda', y="Kesehatan",color='Tahun')
            elif anggaran == "Ketertiban":
                fig1 = px.scatter(dfp, x='nama_pemda', y="Ketertiban",color='Tahun')
            elif anggaran == "Lingkungan":
                fig1 = px.scatter(dfp, x='nama_pemda', y="Lingkungan",color='Tahun')
            elif anggaran == "ParBud":
                fig1 = px.scatter(dfp, x='nama_pemda', y="ParBud",color='Tahun')
            elif anggaran == "Pelayanan":
                fig1 = px.scatter(dfp, x='nama_pemda', y='Total_anggaran',color='Tahun')
            elif anggaran == "Pendidikan":
                fig1 = px.scatter(dfp, x='nama_pemda', y="Pelayanan",color='Tahun')
            elif anggaran == "Sosial":
                fig1 = px.scatter(dfp, x='nama_pemda', y="Sosial",color='Tahun')
            elif anggaran == "Rumah_Fasum":
                fig1 = px.scatter(dfp, x='nama_pemda', y="Rumah_Fasum",color='Tahun')
            st.plotly_chart(fig1)

        with c2:
            if pemda == None or pemda ==[]:
                dfp = df
            else:
                dfp = df[df['nama_pemda'].isin(pemda)]
                        
            if index == "IPM":
                fig2 = px.scatter(dfp, x='nama_pemda', y='Base_IPM',color='Tahun')
            elif index == "AHH":
                fig2 = px.scatter(dfp, x='nama_pemda', y='Base_AHH',color='Tahun')
            elif index == "HLS":
                fig2 = px.scatter(dfp, x='nama_pemda', y='Base_HLS',color='Tahun')
            elif index == "RLS":
                fig2 = px.scatter(dfp, x='nama_pemda', y='Base_RLS',color='Tahun')
            elif index == "PPK":
                fig2 = px.scatter(dfp, x='nama_pemda', y='Base_PPK',color='Tahun')
            fig2.update_layout(yaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig2)

