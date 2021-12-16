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
    st.subheader("Budget Allocations and Prediction of Indexes")
    df = pd.read_excel('belanja_apbd_full.xlsx')
    new = df["nama_pemda"].str.split(" ", n = 1, expand = True)
    df["jenispemda"]= new[0]
    df["namapemda"]= new[1]
    # df.drop(columns =["nama_pemda"], inplace = True)
    df['wilayah'] = np.where(df['jenispemda'].str.contains("Provinsi"), "Provinsi", "Kab_Kota")
    p1, p2, p3 = st.beta_columns((4,0.5,4))
    with p1:
        wilayah = st.selectbox('Choose Region',['Provinsi','Kab_Kota'])
        if wilayah == 'Provinsi':
            df = df[df['wilayah'].isin(['Provinsi'])]
        else:
            df = df[df['wilayah'].isin(['Kab_Kota'])]
        lokasi = df['nama_pemda'].unique()
        kota = st.selectbox('Choose Local Government',lokasi)
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
        lytotal = st.text_input(label="Last Year Budget",value="Rp {:,.0f}".format(int(lybudget.values)))
        lypkp = st.text_input(label="Per Capita Budget",value="Rp {:,.0f}".format(int(lybudget.values/pop)))
        # perkapita = st.write(f"Anggaran Perkapita Akan Dialokasikan: {base/pop:.2f}")
    # with p3:
    #     st.write(" ")
    t1, t2 ,t3,t4= st.beta_columns((4,0.5,2,2))
    with t1:
        st.text_input(label="Total Population",value=int(pop))
        # st.write('(Populasi Penduduk Tahun 2019)')
    with t2:
        st.write(" ")
    with t3:
        # st.text_input(label="% Perubahan Anggaran: ",value=str(int(100*((base-lybudget.values)/lybudget.values)))+" %")
        # st.text_input(label="% Perubahan Perkapita: ",value=str(int(100*((base/pop-lybudget.values/pop)/lybudget.values/pop)))+" %")
        basechange = st.number_input(label="% Budget Adjustment",value=0,min_value=-100, max_value=100, step=5)
        allbase = int((basechange/100)*int(lybudget.values)+int(lybudget.values))
    with t4:
        # allbase = lybudget.values
        # alcpkp = lybudget.values/pop
        # base = st.number_input(label="Total Anggaran Akan Dialokasikan",value=allbase,min_value=0, max_value=1000000000000000)
#         base = st.number_input(label="Total Anggaran Akan Dialokasikan",value=allbase,min_value=0, max_value=1000000000000000,step=10000000000)
        base = int(allbase)
        st.text_input(label="Planned Budget",value="Rp {:,.0f}".format(base))
        # pkp = st.text_input(label="Anggaran Perkapita Akan Dialokasikan",value=alpkp)
    # perkapita = st.write(f"Anggaran Perkapita Akan Dialokasikan: {base/pop:.2f}")

    k1,k2,k3= st.beta_columns((4,0.5,4))
    with k1:
        st.write('(Population in 2019)')
    with k2:
        st.write(" ")
    with k3:
        perkapita = st.write("Planned Per Capita Budget: Rp {:,.0f}".format(int(base/pop)))

    c1, c2 ,c3, c4, c5= st.beta_columns((1,1,1,0.5,2))
    with c1:
        st.subheader("Last Year Budget Allocation")
        st.text_input(label="Ekonomi",value="Rp {:,.0f}".format(df['Ekonomi'].sum()))
        st.text_input(label="Kesehatan",value="Rp {:,.0f}".format(df['Kesehatan'].sum()))
        st.text_input(label="Ketertiban",value="Rp {:,.0f}".format(df['Ketertiban'].sum()))
        st.text_input(label="Lingkungan",value="Rp {:,.0f}".format(df['Lingkungan'].sum()))
        st.text_input(label="Pariwisata Budaya",value="Rp {:,.0f}".format(df['ParBud'].sum()))
        st.text_input(label="Pelayanan",value="Rp {:,.0f}".format(df['Pelayanan'].sum()))
        st.text_input(label="Pendidikan",value="Rp {:,.0f}".format(df['Pendidikan'].sum()))
        st.text_input(label="Sosial",value="Rp {:,.0f}".format(df['Sosial'].sum()))
        st.text_input(label="Rumah_Fasum",value="Rp {:,.0f}".format(df['Rumah_Fasum'].sum()))
    with c2:
        st.subheader("Allocated Budget Distribution (%)")
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
        st.subheader("Allocated Budget Value")
        st.text_input(label="Ekonomi",value="Rp {:,.0f}".format(int(base*a/100)))
        st.text_input(label="Kesehatan",value="Rp {:,.0f}".format(int(base*b/100)))
        st.text_input(label="Ketertiban",value="Rp {:,.0f}".format(int(base*c/100)))
        st.text_input(label="Lingkungan",value="Rp {:,.0f}".format(int(base*d/100)))
        st.text_input(label="Pariwisata_Budaya",value="Rp {:,.0f}".format(int(base*e/100)))
        st.text_input(label="Pelayanan",value="Rp {:,.0f}".format(int(base*f/100)))
        st.text_input(label="Pendidikan",value="Rp {:,.0f}".format(int(base*g/100)))
        st.text_input(label="Sosial",value="Rp {:,.0f}".format(int(base*h/100)))
        st.text_input(label="Rumah_Fasum",value="Rp {:,.0f}".format(int(base*i/100)))
        
    with c4:
        st.write("")
    with c5:
        st.subheader("Index Change Prediction  \nBased on Machine-Learning Models")
        index = st.selectbox('Pilih Index',['IPM','AHH','HLS','RLS','PPK'])
        if index == "IPM":
            st.subheader("Base IPM Index")
            j = st.number_input(label="Base_IPM",value=base_ipm,min_value=0.0, max_value=100.0, step=1.0)
        elif index == "AHH":
            st.subheader("Base AHH Index")
            j = st.number_input(label="Base_AHH",value=base_ahh,min_value=0.0, max_value=100.0, step=1.0)
        elif index == "HLS":
            st.subheader("Base HLS Index")
            j = st.number_input(label="Base_HLS",value=base_hls,min_value=0.0, max_value=100.0, step=1.0)
        elif index == "RLS":
            st.subheader("Base RLS Index")
            j = st.number_input(label="Base_RLS",value=base_rls,min_value=0.0, max_value=100.0, step=1.0)
        elif index == "PPK":
            st.subheader("Base PPK Index")
            j = st.number_input(label="Base_PPK",value=base_ppk,min_value=0, max_value=30000, step=100)

        total = a+b+c+d+e+f+g+h+i
        # st.subheader("Total Alokasi Anggaran: "+str(total)+"%")
        st.subheader(f"Total Distributed Budget: {total:.2f} %")

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
        if st.button("Click to run Prediction"):
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
            st.title(f"Index Prediction: {nilai:.2f}")
            st.subheader(f"Index Change: {(nilai-j):.2f}")

elif choice == 'Anomali':
    st.subheader('Anomali Analysis in Budget Allocations')
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
            dfcor2 = df[['Base_IPM','Base_AHH','Base_HLS','Base_RLS','Base_PPK']]
            dfcor1 = df[["Ekonomi","Kesehatan","Ketertiban","Lingkungan","ParBud","Pelayanan","Pendidikan","Sosial","Rumah_Fasum"]]
            dfall = df[['Base_IPM','Base_AHH','Base_HLS','Base_RLS','Base_PPK',"Ekonomi","Kesehatan","Ketertiban","Lingkungan","ParBud","Pelayanan","Pendidikan","Sosial","Rumah_Fasum"]]
            z0, z1, z2 = st.beta_columns((1,1,1))
            with z0:
                st.write("Correlation Matrix of All Variables")
                st.write(sns.heatmap(dfall.corr(),annot=True,annot_kws={"size": 6}))
                st.pyplot()
            with z1:
                st.write("Correlation Matrix of Budget Functions")
                st.write(sns.heatmap(dfcor1.corr(),annot=True,annot_kws={"size": 10}))
                st.pyplot()
            with z2:
                st.write("Correlation Matrix of Index")
                st.write(sns.heatmap(dfcor2.corr(),annot=True,annot_kws={"size": 12}))
                st.pyplot()
        plot_type = st.selectbox('Select Type of Plot',["bar","line","area","hist","box"])
        all_columns = ["Total","Ekonomi","Kesehatan","Ketertiban","Lingkungan","ParBud","Pelayanan","Pendidikan","Sosial","Rumah_Fasum","Populasi"]
        pemda = st.multiselect('Choose Local Government',df['nama_pemda'])
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
        
    if st.checkbox("Budget Allocation Analysis"):
        # df = pd.read_csv('belanja_apbd.csv',sep=";")
        df = pd.read_excel('belanja_apbd_full.xlsx')
        df.loc[:,'Total_anggaran'] = df.sum(numeric_only=True, axis=1)
        st.subheader('Anomali Detection from Trends and Outliers Analysis ')
        index = st.selectbox('Choose Index',['IPM','AHH','HLS','RLS','PPK'])
        anggaran = st.selectbox('Choose Budget Function',["All","Ekonomi","Kesehatan","Ketertiban","Lingkungan","ParBud","Pelayanan","Pendidikan","Sosial","Rumah_Fasum"])
        pemda = st.multiselect('Choose Local Government',df['nama_pemda'].unique())
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
                msize=10
            else:
                dfp = df[df['nama_pemda'].isin(pemda)]
                msize=25

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
                
            fig1.update_traces(marker_size=msize, selector=dict(type='scatter'))
            st.plotly_chart(fig1)

        with c2:
            if pemda == None or pemda ==[]:
                dfp = df
                msize=10
            else:
                dfp = df[df['nama_pemda'].isin(pemda)]
                msize=25
                
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
            fig2.update_traces(marker_size=msize, selector=dict(type='scatter'))
            st.plotly_chart(fig2)

