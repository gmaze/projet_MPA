import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Créer une carte avec la projection PlateCarree
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Tracer une carte du monde
ax.coastlines()

# Ajouter quelques points sur la carte
lons = [0, 45, -30, 150]
lats = [30, 10, -20, -40]
ax.scatter(lons, lats, color='red', marker='o', transform=ccrs.PlateCarree())

# Ajouter des frontières politiques
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Ajouter des noms aux pays
ax.text(0, 30, 'Point 1', transform=ccrs.PlateCarree(), color='blue')
ax.text(45, 10, 'Point 2', transform=ccrs.PlateCarree(), color='blue')
ax.text(-30, -20, 'Point 3', transform=ccrs.PlateCarree(), color='blue')
ax.text(150, -40, 'Point 4', transform=ccrs.PlateCarree(), color='blue')

# Afficher la carte
plt.show()

