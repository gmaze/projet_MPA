#fichier principal pour faire fonctionner application
# pour demarrer streamlit : streamlit run main.py
# pour se placer dans environnement MPA : conda activate MPA
# pour generer environnement.yml : conda env export > environment.yml
# pour mettre en place sur un nouvel ordinateur environment MPA avec le fichier environement.yml : conda env create -f environment.yml
# pour ouvrir  anaconda navigotor ( gestionnaire environemment) : anaconda-navigator

from random import randint
import gsw 
import streamlit as st
import xarray as xr
from argopy import DataFetcher as ArgoDataFetcher
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import numpy as np
import pandas as pd
import datetime as dt
import numpy as np
from pyxpcm.models import pcm
import xarray as xr
from argopy import DataFetcher as ArgoDataFetcher
#import cartopy.crs as ccrs 
import matplotlib.pyplot as plt
#import cartopy.feature as cfeature

def int_to_rgb(i: int):
    r = hex((i*i+10)%15).replace("0x","")
    g = hex((i*5)%15).replace("0x","")
    b = hex((i*i*i*2)%15).replace("0x","")
    return "#"+r*2+g*2+b*2

def recup_argo_data(llon:float,rlon:float, llat:float,ulat:float, depthmin:float, depthmax:float,time_in:str,time_f:str):

     # Temporarily redefine np.int to int -> car sinon on a une erreur np.int est deprecier dans NumPy 1.20
    np.int = int

    #recuperer les données avec argopy
    ds_points = ArgoDataFetcher(src='erddap', parallel=True).region([llon,rlon, llat,ulat, depthmin, depthmax,time_in,time_f]).to_xarray()
    #mettre en 2 dimensions ( N_PROf x N_LEVELS)
    ds = ds_points.argo.point2profile()
    """"""
    #recuperer les données SIG0 et N2(BRV2) avec teos 10
    ds.argo.teos10(['SIG0','N2'])

    # z est créé pour représenter des profondeurs de 0 à la profondeur maximale avec un intervalle de 5 mètres ( peux etre modifié)
    z=np.arange(0,depthmax,5)
    #interpole ds avec les profondeurs z
    ds2 = ds.argo.interp_std_levels(z)

    #Calculer la profondeur avec gsw.z_from_p a partir de la p (PRES) et de lat (LATITUDE)
    p=np.array(ds2.PRES)
    lat=np.array(ds2.LATITUDE)
    z=np.ones_like(p)
    nprof=np.array(ds2.N_PROF)

    for i in np.arange(0,len(nprof)):
        z[i,:]=gsw.z_from_p(p[i,:], lat[i])


    # Calcul de la profondeur à partir de la pression interpolée 
    p_interp=np.array(ds2.PRES_INTERPOLATED)
    z_interp=gsw.z_from_p(p_interp, 25) 


    #Créer un objet Dataset xarray pour stocker les données
    temp=np.array(ds2.TEMP)
    sal=np.array(ds2.PSAL)
    depth_var=z
    depth=z_interp
    lat=np.array(ds2.LATITUDE)
    lon=np.array(ds2.LONGITUDE)
    time=np.array(ds2.TIME)
    sig0 =np.array(ds2.SIG0)
    brv2 =np.array(ds2.N2)

    #ranger les données dans xarrays
    da=xr.Dataset(data_vars={
                            'TIME':(('N_PROF'),time),
                            'LATITUDE':(('N_PROF'),lat),
                            'LONGITUDE':(('N_PROF'),lon),
                            'TEMP':(('N_PROF','DEPTH'),temp),
                            'PSAL':(('N_PROF','DEPTH'),sal),
                            'SIG0':(('N_PROF','DEPTH'),sig0),
                            'BRV2':(('N_PROF','DEPTH'),brv2)
                            },
                            coords={'DEPTH':depth})
    return da


def pyxpcm_sal_temp(da):

    # Temporarily redefine np.int to int -> car sinon on a une erreur np.int est deprecier dans NumPy 1.20
    np.int = int


    z = np.arange(0.,-900,-10.) # depth array
    pcm_features = {'temperature': z, 'salinity':z} #features that vary in function of depth
    m = pcm(K=6, features=pcm_features) # create the 'basic' model
    
    
    
    features_in_ds = {'temperature': 'TEMP', 'salinity': 'PSAL'}
    features_zdim='DEPTH'
    m.fit(da, features=features_in_ds, dim=features_zdim)
    da['TEMP'].attrs['feature_name'] = 'temperature'
    da['PSAL'].attrs['feature_name'] = 'salinity'
    da['DEPTH'].attrs['axis'] = 'Z'

    m.predict(da, features=features_in_ds, dim=features_zdim,inplace=True)
 

    m.predict_proba(da, features=features_in_ds, inplace=True)

    for vname in ['TEMP', 'PSAL']:
        da = da.pyxpcm.quantile(m, q=[0.05, 0.5, 0.95], of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)


    # Reset np.int to its original value
    np.int = np.int_


    return da

if 'ds' not in st.session_state:
    st.session_state.ds = xr.Dataset()

def main():
    global ds
    df_points = pd.DataFrame()
    st.set_page_config(layout="wide")

    # parameters canvas
    # argopy parameters

    argopy_text = '<b><font color="blue" size="5">parametre argopy</font></b>'
    st.sidebar.markdown(argopy_text, unsafe_allow_html=True)

    longitude = st.sidebar.slider("longitude", min_value=-180.0, max_value=180.0, value=[-75.0, -45.0])
    latitude = st.sidebar.slider("Latitude ", min_value=-90.0, max_value=90.0, value=[30.0, 20.0])
    profondeur = st.sidebar.slider("Profondeur", min_value=0.0, max_value=1000.0, value=[0.0, 1000.0])
    date_debut = st.sidebar.date_input("date début", (dt.date(2010, 1, 1)), format="YYYY-MM-DD")
    date_fin = st.sidebar.date_input("date fin", (dt.date(2010, 12, 1)), format="YYYY-MM-DD")
    llon = longitude[0]
    rlon = longitude[1]
    llat = latitude[0]
    ulat = latitude[1]
    depthmin = profondeur[0]
    depthmax = profondeur[1]
    time_in = date_debut.strftime("%Y-%m-%d")
    time_f = date_fin.strftime("%Y-%m-%d")

    button_fetch_data = st.sidebar.button("Récupérer les données")

    # pyxpcm parameters
    pyxpcm_text = '<b><font color="blue" size="5">parametre pyxpcm</font></b>'
    st.sidebar.markdown(pyxpcm_text, unsafe_allow_html=True)
    clusters = st.sidebar.slider("nombre de clusters(K)", min_value=2, max_value=20, value=6)

    button_class_data = st.sidebar.button("classifier les données")

    if button_fetch_data:
        st.session_state.ds  = recup_argo_data(llon,rlon, llat,ulat, depthmin, depthmax,time_in,time_f)
        st.write(st.session_state.ds.to_dataframe())

    if button_class_data:
        print(st.session_state.ds)
        ds_py = pyxpcm_sal_temp(st.session_state.ds )
        ds_trier = ds_py.isel(DEPTH=0)
        ds_trier = ds_trier.isel(quantile=1)
        df_points = ds_trier.to_dataframe()
        st.write(ds_trier.to_dataframe())

    # Map canvas
    # créer un map folium avec la latitude et longitude 
    m = folium.Map(location=[np.mean(latitude), np.mean(longitude)], zoom_start=4)

    #maj emplacement si changement latitude ou longitude
    m.location = [np.mean(latitude), np.mean(longitude)]

    # ajouter un rectangle pour marquer l'emplacement des données.
    folium.Rectangle(bounds=[(latitude[0], longitude[0]), (latitude[1], longitude[1])],
                        color='red').add_to(m)

    # Vérifier si df_points n'est pas vide
    if not df_points.empty :
        #st.write(df_points)
        max_pcm_class = df_points['PCM_LABELS'].max()
        df_points['Color'] = df_points['PCM_LABELS'].apply(lambda x: "#{:06x}".format((x * 977) % 0x1000000))
        for index, row in df_points.iterrows():

            color = int_to_rgb(row['PCM_LABELS'])

            # Créer un popup avec les informations du DataFrame
            popup_content = """
                    DEPTH: {}<br> 
                    PCM_CLASS: {}<br>
                    LATITUDE : {}<br>
                    LONGITUDE : {}<br>
                    TIME : {}<br>
                    PSAL : {}<br>
                    TEMP : {}<br>
                """.format(row['DEPTH'], row['PCM_LABELS'],row['LATITUDE'], row['LONGITUDE'], row['TIME'], row['PSAL'], row['TEMP'])
            folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']],
                            radius=3,  # taille du cercle 
                            color=color,  # couleur du cercle
                            fill=True,
                            fill_color= color,  # couleur de remplissage du cercle
                            fill_opacity=1,
                            popup=folium.Popup(popup_content, max_width=300)).add_to(m)



        
    # Afficher la carte
    folium_static(m, width=1350, height=600)



    # graphique canvas
if __name__ == "__main__":
    main()
