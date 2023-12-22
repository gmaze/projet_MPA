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


data = xr.Dataset()
# créer une box pour dataFetcher de argopy
llon=-90;rlon=-80
ulat=40;llat=18
depthmin=0;depthmax=200
# Time range des donnnées
time_in='2010-01'
time_f='2017-01'

#recuperer les données avec argopy
ds_points = ArgoDataFetcher(src='erddap', parallel=True).region([llon,rlon, llat,ulat, depthmin, depthmax,time_in,time_f]).to_xarray()
#mettre en 2 dimensions ( N_PROf x N_LEVELS)
ds = ds_points.argo.point2profile()
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
print(da)
data = da
print(da)