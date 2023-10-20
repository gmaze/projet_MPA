#fichier permettant de tester les fonctionnalités des librairies
import argopy
import pandas as pd

# Définisse une requête pour obtenir les données Argo souhaitées
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

