"""#fichier permettant de tester les fonctionnalités des librairies
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


from argopy import DataFetcher
import xarray as xr

f = DataFetcher(src='argovis').region([-75,-55,30.,40.,0,100., '2011-01-01', '2011-01-15'])

ds_points = f.data

ds_profiles = ds_points.argo.point2profile()





# Inversez le sens de la profondeur pour correspondre à votre convention
ds_profiles["DEPTH"] = ds_profiles.N_PROF *ds_profiles.N_LEVELS
ds_profiles['DEPTH'].attrs['axis'] = 'Z'
ds_profiles['DEPTH'].attrs['units'] = 'meters'
ds_profiles['DEPTH'].attrs['positive'] = 'up'
# Créez un nouvel xarray.Dataset avec les dimensions et les variables souhaitées
ds_profiles = xr.Dataset(
    data_vars={
        "LATITUDE": ds_profiles.LATITUDE,
        "LONGITUDE": ds_profiles.LONGITUDE,
        "TIME": ds_profiles.TIME,
        "DBINDEX": ds_profiles.CYCLE_NUMBER,
        "TEMP": ds_profiles.TEMP,
        "PSAL": ds_profiles.PSAL,
        "SIG0": None,
        "BRV2": None,
    },
    coords={"N_PROF": ds_profiles.N_PROF, "DEPTH": ds_profiles.DEPTH},
)


# Définissez les attributs pour le nouvel xarray.Dataset
ds_profiles.attrs["Sample test prepared by"] = "G. Maze"
ds_profiles.attrs["Institution"] = "Ifremer/LOPS"
ds_profiles.attrs["Data source DOI"] = "10.17882/42182"
ds_profiles['TEMP'].attrs['feature_name'] = 'temperature'
ds_profiles['PSAL'].attrs['feature_name'] = 'salinity'
ds_profiles['DEPTH'].attrs['axis'] = 'Z'
# Affichez le nouvel xarray.Dataset
"""
import argopy
from argopy import DataFetcher 



# Utilisez la méthode float() pour obtenir les flotteurs correspondant aux critères de recherche
ds = DataFetcher().float(2901623).to_xarray()
ds.argo.teos10(['N2', 'SIG0'])
ds = ds.argo.point2profile()
ds = ds.reset_coords("LATITUDE")
ds = ds.reset_coords("LONGITUDE")
ds = ds.reset_coords("TIME")
ds = ds.drop_vars("CYCLE_NUMBER")
ds = ds.drop_vars("DATA_MODE")
ds = ds.drop_vars("DIRECTION")
ds = ds.drop_vars("PLATFORM_NUMBER")
ds = ds.drop_vars("POSITION_QC")
ds = ds.drop_vars("PRES")
ds = ds.drop_vars("TEMP_ERROR")
ds = ds.drop_vars("TEMP_QC")
ds = ds.drop_vars("TIME_QC")
ds = ds.drop_vars("PRES_ERROR")
ds = ds.drop_vars("PRES_QC")
ds = ds.drop_vars("PSAL_ERROR")
ds = ds.drop_vars("PSAL_QC")

print(ds)


