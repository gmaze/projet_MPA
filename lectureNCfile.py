from netCDF4 import Dataset

# Ouvrez le fichier NetCDF
nc_file = Dataset('/home/mona/Téléchargements/argo_sample.nc', 'r')  # Remplacez 'votre_fichier.nc' par le chemin vers votre fichier NetCDF.
"""
for var_name in nc_file.variables:
    variable = nc_file.variables[var_name]
    
    # Accédez aux valeurs de la variable
    variable_values = variable[:]
    
    # Accédez aux valeurs valid_min et valid_max s'ils existent
    try:
        valid_min = variable.getncattr('valid_min')
        valid_max = variable.getncattr('valid_max')
    except AttributeError:
        valid_min = None
        valid_max = None
    
    # Vous pouvez traiter ou afficher les données ici en fonction de vos besoins
    print(f"Variable : {var_name}")
    print(f"Valid Min : {valid_min}")
    print(f"Valid Max : {valid_max}")
"""
depth_data = nc_file.variables['DEPTH']  
depth_values = depth_data[:]
lat_data = nc_file.variables['LATITUDE']  
lat_values = lat_data[:]
lon_data = nc_file.variables['LONGITUDE']  
lon_values = lon_data[:]
ti_data = nc_file.variables['TIME']  
ti_values = ti_data[:]
db_data = nc_file.variables['DBINDEX']  
db_values = db_data[:]
tem_data = nc_file.variables['TEMP']  
tem_values = tem_data[:]
sal_data = nc_file.variables['PSAL']  
sal_values = sal_data[:]
sig_data = nc_file.variables['SIG0']  
sig_values = sig_data[:]
br_data = nc_file.variables['BRV2']  
br_values = br_data[:]
print("depth")
print(depth_values)
print("latitude")
print(lat_values)
print("longitude")
print(lon_values)
print("time")
print(ti_values)
print("dbindex")
print(db_values)
print("temperature")
print(tem_values)
print("salinité")
print(sal_values)
print("sig0")
print(sig_values)
print("brv2")
print(br_values)


# N'oubliez pas de fermer le fichier NetCDF lorsque vous avez terminé
nc_file.close()