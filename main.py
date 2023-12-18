#fichier principal pour faire fonctionner application
# pour demarrer streamlit : streamlit run main.py
# pour se placer dans environnement MPA : conda activate MPA
# pour generer environnement.yml : conda env export > environment.yml
# pour mettre en place sur un nouvel ordinateur environment MPA avec le fichier environement.yml : conda env create -f environment.yml
# pour ouvrir  anaconda navigotor ( gestionnaire environemment) : anaconda-navigator

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
#import pyxpcm
#from pyxpcm.models import pcm
import xarray as xr
from argopy import DataFetcher as ArgoDataFetcher
#import cartopy.crs as ccrs 
#import matplotlib.pyplot as plt
#import cartopy.feature as cfeature



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
date_fin = st.sidebar.date_input("date fin", (dt.date(2010, 1, 7)), format="YYYY-MM-DD")
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

if button_fetch_data:
    # Récupérer les données uniquement si le bouton est cliqué
    ds_points = ArgoDataFetcher(src='erddap').region([llon,rlon, llat,ulat, depthmin, depthmax,time_in,time_f]).to_xarray()
    #recuperer les données avec argopy
    ds_points = ArgoDataFetcher(src='erddap').region([llon,rlon, llat,ulat, depthmin, depthmax,time_in,time_f]).to_xarray()
    #mettre en 2 dimensions ( N_PROf x N_LEVELS)
    ds = ds_points.argo.point2profile()
    #recuperer les données SIG0 et N2(BRV2) avec teos 10
    ds.argo.teos10(['SIG0','N2'])

    # z est créé pour représenter des profondeurs de 0 à la profondeur maximale avec un intervalle de 5 mètres ( peux etre modifié)
    z=np.arange(0,depthmax,5)
    #interpole ds avec les profondeurs z
    ds2 = ds.argo.interp_std_levels(z)

    df_points = ds2.to_dataframe()




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
    st.write(df_points)
    marker_cluster = MarkerCluster().add_to(m)

    for index, row in df_points.iterrows():
                # Créer un popup avec les informations du DataFrame
                popup_content = """
                        Latitude: {}<br>
                        Longitude: {}<br>
                        Pres: {}<br>
                        Psal: {}<br>
                        Temp: {}<br>
                        Time: {}<br>
                        N2: {}<br>
                        SIG0: {}
                    """.format(row['LATITUDE'], row['LONGITUDE'], row['PRES'], 
                            row['PSAL'], row['TEMP'], row['TIME'], row['N2'], row['SIG0'])
                folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']],
                                radius=5,  # ajustez la taille du cercle selon vos préférences
                                color='blue',  # couleur du cercle
                                fill=True,
                                fill_color='blue',  # couleur de remplissage du cercle
                                fill_opacity=0.7,
                                popup=folium.Popup(popup_content, max_width=300)).add_to(marker_cluster)
 


    
# Afficher la carte
folium_static(m, width=1350, height=600)

# graphique canvas






    


    

    
