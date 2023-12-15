#fichier principal pour faire fonctionner application
# pour demarrer streamlit : streamlit run main.py
# pour se placer dans environnement MPA : conda activate MPA
# pour generer environnement.yml : conda env export > environment.yml
# pour mettre en place sur un nouvel ordinateur environment MPA avec le fichier environement.yml : conda env create -f environment.yml
# pour ouvrir  anaconda navigotor ( gestionnaire environemment) : anaconda-navigator

import streamlit as st
import xarray as xr
from argopy import DataFetcher as ArgoDataFetcher
import folium
from streamlit_folium import folium_static
import numpy as np
import pandas as pd
import datetime as dt

st.title("Application Streamlit argopy")

# parameters canvas
# argopy parameters


argopy_text = '<b><font color="blue" size="5">parametre argopy</font></b>'
st.sidebar.markdown(argopy_text, unsafe_allow_html=True)

longitude = st.sidebar.slider("longitude", min_value=-180.0, max_value=180.0, value=[-75.0, -45.0])
latitude = st.sidebar.slider("Latitude ", min_value=-90.0, max_value=90.0, value=[30.0, 20.0])
profondeur = st.sidebar.slider("Profondeur", min_value=0.0, max_value=1000.0, value=[0.0, 1000.0])
date_debut = st.sidebar.date_input("date début", (dt.date(2010, 1, 1)), format="YYYY-MM-DD")
date_fin = st.sidebar.date_input("date fin", (dt.date(2010, 1, 7)), format="YYYY-MM-DD")

#mettre en bon format les parametres
llon = longitude[0]
rlon = longitude[1]
llat = latitude[0]
ulat = latitude[1]
depthmin = profondeur[0]
depthmax = profondeur[1]
time_in = date_debut.strftime("%Y-%m-%d")
time_f = date_fin.strftime("%Y-%m-%d")

button_fetch_data = st.sidebar.button("Récupérer les données")

# Vérifier si le bouton est cliqué
if button_fetch_data:
    # Récupérer les données uniquement si le bouton est cliqué
    ds_points = ArgoDataFetcher(src='erddap').region([llon,rlon, llat,ulat, depthmin, depthmax,time_in,time_f]).to_xarray()
    df_points = ds_points.to_dataframe()
    # Afficher le DataFrame dans Streamlit penser à regarder si il y a un moyen de voir si erreur dans récupération ( st.erreur ?)
    st.write("Données récupérées avec succès!")
    st.dataframe(df_points)



# pyxpcm parameters
pyxpcm_text = '<b><font color="blue" size="5">parametre pyxpcm</font></b>'
st.sidebar.markdown(pyxpcm_text, unsafe_allow_html=True)
clusters = st.sidebar.slider("nombre de clusters(K)", min_value=2, max_value=20, value=6)

# Map canvas
col_map = st.container()
with col_map:
    # Create a Folium map centered around the selected coordinates
    m = folium.Map(location=[np.mean(latitude), np.mean(longitude)], zoom_start=4)


    m.location = [np.mean(latitude), np.mean(longitude)]
    m.zoom_start = 5
    # Add a rectangle to represent the selected area
    folium.Rectangle(bounds=[(latitude[0], longitude[0]), (latitude[1], longitude[1])],
                     color='red').add_to(m)

    # Display the map
    folium_static(m)

# graphique canvas
