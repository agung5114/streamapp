import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from PIL import Image
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
# load model 
import joblib

st.set_page_config(layout="wide")
st.title('ALONA: Allocation - Outcome - Anomali')
# pages = ['Home','Anomali','Allocation','Outcome']
# jump = st.sidebar.selectbox('Select Pages', pages)

menu = ['Anomali','Allocation & Outcome']
alona= Image.open('logo_alona.png')
st.sidebar.image(alona)
choice = st.sidebar.selectbox('Select Menu', menu)

if choice == "Allocation & Outcome":
    st.subheader("Alokasi Anggaran dan Prediksi Perubahan Index")
    df = pd.read_excel('belanja_apbd_full.xlsx')
    new = df["nama_pemda"].str.split(" ", n = 1, expand = True)
    df["jenispemda"]= new[0]
    df["namapemda"]= new[1]
    # df.drop(columns =["nama_pemda"], inplace = True)
    df['wilayah'] = np.where(df['jenispemda'].str.contains("Provinsi"), "Provinsi", "Kab_Kota")
    p1, p2, p3 = st.beta_columns((4,0.5,4))
    with p1:
        wilayah = st.selectbox('Pilih Jenis Wilayah',['Provinsi','Kab_Kota'])
        if wilayah == 'Provinsi':
            df = df[df['wilayah'].isin(['Provinsi'])]
        else:
            df = df[df['wilayah'].isin(['Kab_Kota'])]
        lokasi = df['nama_pemda'].unique()
        kota = st.selectbox('Pilih Daerah',lokasi)
        df = df[df['nama_pemda'].isin([kota])]
        df = df[df['Tahun'].isin([2019])]
        lybudget = df.sum(axis=1)
        pop = df['Populasi'].sum()
        base_ipm = df['Base_IPM'].sum()
        base_ahh = df['Base_AHH'].sum()
        base_hls = df['Base_HLS'].sum()
        base_rls = df['Base_RLS'].sum()
        base_ppk = df['Base_PPK'].sum()
        df['Total_Anggaran'] = df.sum(axis=1)
        ekvalue=df['Ekonomi']/df["Total_Anggaran"]
        ksvalue=df['Kesehatan']/df["Total_Anggaran"]
        ktvalue=df['Ketertiban']/df["Total_Anggaran"]
        lkvalue=df['Lingkungan']/df["Total_Anggaran"]
        pbvalue=df['ParBud']/df["Total_Anggaran"]
        plvalue=df['Pelayanan']/df["Total_Anggaran"]
        pdvalue=df['Pendidikan']/df["Total_Anggaran"]
        sovalue=df['Sosial']/df["Total_Anggaran"]
        fsvalue=df['Rumah_Fasum']/df["Total_Anggaran"]
    with p2:
        st.write("")
    with p3:
        # st.write(f"Total Populasi: {pop:.0f} (2019)")
        lytotal = st.text_input(label="Total Anggaran Tahun Terakhir",value="Rp {:,.0f}".format(int(lybudget.values)))
        lypkp = st.text_input(label="Anggaran Perkapita",value="Rp {:,.0f}".format(int(lybudget.values/pop)))
        # perkapita = st.write(f"Anggaran Perkapita Akan Dialokasikan: {base/pop:.2f}")
    # with p3:
    #     st.write(" ")
    t1, t2 ,t3,t4= st.beta_columns((4,0.5,2,2))
    with t1:
        st.text_input(label="Total Populasi",value=int(pop))
        # st.write('(Populasi Penduduk Tahun 2019)')
    with t2:
        st.write(" ")
    with t3:
        # st.text_input(label="% Perubahan Anggaran: ",value=str(int(100*((base-lybudget.values)/lybudget.values)))+" %")
        # st.text_input(label="% Perubahan Perkapita: ",value=str(int(100*((base/pop-lybudget.values/pop)/lybudget.values/pop)))+" %")
        basechange = st.number_input(label="% Penyesuaian Anggaran",value=0,min_value=-100, max_value=100, step=5)
        allbase = int((basechange/100)*int(lybudget.values)+int(lybudget.values))
    with t4:
        # allbase = lybudget.values
        # alcpkp = lybudget.values/pop
        # base = st.number_input(label="Total Anggaran Akan Dialokasikan",value=allbase,min_value=0, max_value=1000000000000000)
#         base = st.number_input(label="Total Anggaran Akan Dialokasikan",value=allbase,min_value=0, max_value=1000000000000000,step=10000000000)
        base = int(allbase)
        st.text_input(label="Total Anggaran Akan Dialokasikan",value="Rp {:,.0f}".format(base))
        # pkp = st.text_input(label="Anggaran Perkapita Akan Dialokasikan",value=alpkp)
    # perkapita = st.write(f"Anggaran Perkapita Akan Dialokasikan: {base/pop:.2f}")

    k1,k2,k3= st.beta_columns((4,0.5,4))
    with k1:
        st.write('(Populasi Penduduk Tahun 2019)')
    with k2:
        st.write(" ")
    with k3:
        perkapita = st.write("Anggaran Perkapita Akan Dialokasikan: Rp {:,.0f}".format(int(base/pop)))

    c1, c2 ,c3, c4, c5= st.beta_columns((1,1,1,0.5,2))
    with c1:
        st.subheader("Alokasi Tahun Sebelumnya (Rp)")
        st.text_input(label="Ekonomi",value=df['Ekonomi'].sum())
        st.text_input(label="Kesehatan",value=df['Kesehatan'].sum())
        st.text_input(label="Ketertiban",value=df['Ketertiban'].sum())
        st.text_input(label="Lingkungan",value=df['Lingkungan'].sum())
        st.text_input(label="Pariwisata Budaya",value=df['ParBud'].sum())
        st.text_input(label="Pelayanan",value=df['Pelayanan'].sum())
        st.text_input(label="Pendidikan",value=df['Pendidikan'].sum())
        st.text_input(label="Sosial",value=df['Sosial'].sum())
        st.text_input(label="Rumah_Fasum",value=df['Rumah_Fasum'].sum())
    with c2:
        st.subheader("Proporsi Alokasi Anggaran (%)")
        a = st.number_input(label="Ekonomi",value=100*float(ekvalue.values),min_value=0.0, max_value=100.0, step=1.0)
        b = st.number_input(label="Kesehatan",value=100*float(ksvalue.values),min_value=0.0, max_value=100.0, step=1.0)
        c = st.number_input(label="Ketertiban",value=100*float(ktvalue.values),min_value=0.0, max_value=100.0, step=1.0)
        d = st.number_input(label="Lingkungan",value=100*float(lkvalue.values),min_value=0.0, max_value=100.0, step=1.0)
        e = st.number_input(label="Pariwisata Budaya",value=100*float(pbvalue.values),min_value=0.0, max_value=100.0, step=1.0)
        f = st.number_input(label="Pelayanan",value=100*float(plvalue.values),min_value=0.0, max_value=100.0, step=1.0)
        g = st.number_input(label="Pendidikan",value=100*float(pdvalue.values),min_value=0.0, max_value=100.0, step=1.0)
        h = st.number_input(label="Sosial",value=100*float(sovalue.values),min_value=0.0, max_value=100.0, step=1.0)
        i = st.number_input(label="Rumah_Fasum",value=100*float(fsvalue.values),min_value=0.0, max_value=100.0, step=1.0)
    
    with c3:
        st.subheader("Jumlah Anggaran Dialokasikan (Rp)")
        st.text_input(label="Ekonomi",value=int(base*a/100))
        st.text_input(label="Kesehatan",value=int(base*b/100))
        st.text_input(label="Ketertiban",value=int(base*c/100))
        st.text_input(label="Lingkungan",value=int(base*d/100))
        st.text_input(label="Pariwisata_Budaya",value=int(base*e/100))
        st.text_input(label="Pelayanan",value=int(base*f/100))
        st.text_input(label="Pendidikan",value=int(base*g/100))
        st.text_input(label="Sosial",value=int(base*h/100))
        st.text_input(label="Rumah_Fasum",value=int(base*i/100))
        
    with c4:
        st.write("")
    with c5:
        st.subheader("Prediksi Perubahan Index  \nBerdasarkan Machine-Learning Models")
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

        total = a+b+c+d+e+f+g+h+i
        # st.subheader("Total Alokasi Anggaran: "+str(total)+"%")
        st.subheader(f"Total Anggaran Dialokasikan: {total:.2f} %")

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
            nilai = j+float(prediction)
            st.title(f"Prediksi Index: {nilai:.2f}")
            st.subheader(f"Perubahan Index: {(nilai-j):.2f}")

elif choice == 'Anomali':
    st.subheader('Analisis atas Anomali dalam Alokasi Anggaran dan Tingkat Perubahan Index')
    if st.checkbox("Exploratory Data Analysis"):
        # df = pd.read_csv('belanja_apbd.csv',sep=";",usecols=["nama_pemda","Tahun","Ekonomi","Kesehatan","Ketertiban","Lingkungan","ParBud","Pelayanan","Pendidikan","Sosial","Rumah_Fasum","Populasi"])
        # st.subheader('Automated Exploratory Data Analysis')
        df = pd.read_excel('belanja_apbd_full.xlsx')
        df.loc[:,'Total'] = df.sum(numeric_only=True, axis=1)
#         df = df.style.format("{:,.0f}")
        st.dataframe(df.style.format(subset=["Total","Ekonomi","Kesehatan","Ketertiban","Lingkungan","ParBud","Pelayanan","Pendidikan","Sosial","Rumah_Fasum","Populasi"],formatter="Rp {:,.0f}"))
        if st.checkbox('Show Shape'):
            st.write(df.shape)
        if st.checkbox('Show Summary'):
            st.write(df.describe())
        if st.checkbox('Correlation Matrix'):
            st.write(sns.heatmap(df.corr(),annot=True,annot_kws={"size": 3.8}))
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
        # df = pd.read_csv('belanja_apbd.csv',sep=";")
        df = pd.read_excel('belanja_apbd_full.xlsx')
        df.loc[:,'Total_anggaran'] = df.sum(numeric_only=True, axis=1)
        st.subheader('Anomali Detection from Trends and Outliers Analysis ')
        index = st.selectbox('Pilih Index',['IPM','AHH','HLS','RLS','PPK'])
        anggaran = st.selectbox('Pilih Jenis Fungsi',["All","Ekonomi","Kesehatan","Ketertiban","Lingkungan","ParBud","Pelayanan","Pendidikan","Sosial","Rumah_Fasum"])
        pemda = st.multiselect('Pilih Pemda',df['nama_pemda'].unique())
        df['TotalPerkapita'] = df['Total_anggaran']/df['Populasi']
        df['ek'] = df['Ekonomi']/df['Populasi']
        df['ks'] = df['Kesehatan']/df['Populasi']
        df['kt'] = df['Ketertiban']/df['Populasi']
        df['lk'] = df['Lingkungan']/df['Populasi']
        df['pb'] = df['ParBud']/df['Populasi']
        df['pl'] = df['Pelayanan']/df['Populasi']
        df['pd'] = df['Pendidikan']/df['Populasi']
        df['so'] = df['Sosial']/df['Populasi']
        df['fs'] = df['Rumah_Fasum']/df['Populasi']

        c1, c2 = st.beta_columns((1, 1))
        df["Tahun"] = df["Tahun"].astype(str)
        # st.write(pemda)
        with c1:
            if pemda == None or pemda ==[]:
                dfp = df
            else:
                dfp = df[df['nama_pemda'].isin(pemda)]

            if anggaran == "All":
                fig1 = px.scatter(dfp, x='nama_pemda', y='TotalPerkapita',color='Tahun')
            elif anggaran == "Ekonomi":
                fig1 = px.scatter(dfp, x='nama_pemda', y="ek",color='Tahun')
            elif anggaran == "Kesehatan":
                fig1 = px.scatter(dfp, x='nama_pemda', y="ks",color='Tahun')
            elif anggaran == "Ketertiban":
                fig1 = px.scatter(dfp, x='nama_pemda', y="kt",color='Tahun')
            elif anggaran == "Lingkungan":
                fig1 = px.scatter(dfp, x='nama_pemda', y="lk",color='Tahun')
            elif anggaran == "ParBud":
                fig1 = px.scatter(dfp, x='nama_pemda', y="pb",color='Tahun')
            elif anggaran == "Pelayanan":
                fig1 = px.scatter(dfp, x='nama_pemda', y='pl',color='Tahun')
            elif anggaran == "Pendidikan":
                fig1 = px.scatter(dfp, x='nama_pemda', y="pd",color='Tahun')
            elif anggaran == "Sosial":
                fig1 = px.scatter(dfp, x='nama_pemda', y="so",color='Tahun')
            elif anggaran == "Rumah_Fasum":
                fig1 = px.scatter(dfp, x='nama_pemda', y="fs",color='Tahun')
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
            # fig2.update_layout(yaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig2)

