"""
#fichier permettant de tester les fonctionnalités des librairies
import argopy
import pandas as pd

# Définisse une requête pour obtenir les données Argo souhaitées
# Exemple : Région géographique, période de temps et profondeur
argo_loader = argopy.DataFetcher()
ds = argo_loader.region([-75, -45, 20, 30, 0, 1000, "2010-01", "2010-12"]).to_xarray()

# Sélectionne uniquement les colonnes souhaitées
selected_columns = ['N_POINTS', 'LATITUDE', 'LONGITUDE', 'POSITION_QC','TIME','TIME_QC', 'PRES', 'PRES_QC', 'PSAL', 'PSAL_QC','TEMP' ,'TEMP_QC']
df = ds[selected_columns].to_dataframe()

# Affiche les premières lignes du DataFrame
#print(def.head())
#Afficher les 10 dernière lignes du DataFrame
print(df.tail(10))

# Exporte le DataFrame vers un fichier CSV
#df.to_csv('donnees_argo.csv', index=False)



#rendre les données compatible pour la librairie pyxpcm
import argopy
from argopy import DataFetcher 
import xarray as xr


ds = DataFetcher(src='argovis').region([-75,-55,30.,40.,0,100., '2011-01-01', '2011-01-15']).to_xarray()
ds.argo.teos10(['N2', 'SIG0'])
ds = ds.argo.point2profile()
print(ds)

"""
#mise en place des données pour la librairies pyxpcm avec argopy
import numpy as np
from argopy import DataFetcher as ArgoDataFetcher
import gsw
import xarray as xr


# créer une box pour dataFetcher de argopy
llon=-98;rlon=-80
ulat=31;llat=18
depthmin=0;depthmax=200
# Time range des donnnées
time_in='2010-01'
time_f='2017-01'

#recuperer les données avec argopy
ds_points = ArgoDataFetcher(src='erddap').region([llon,rlon, llat,ulat, depthmin, depthmax,time_in,time_f]).to_xarray()
ds = ds_points.argo.point2profile()
print(ds_points)