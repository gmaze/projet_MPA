# ne fonctionne pas pour l'instant
import numpy as np

# Temporarily redefine np.int to int -> car sinon on a une erreur np.int est deprecier dans NumPy 1.20
np.int = int

import argopy
from pyxpcm.models import pcm

z = np.arange(0., -1000, -10.)
pcm_features = {'temperature': z, 'salinity': z}

m = pcm(K=8, features=pcm_features)
ds = argopy.DataFetcher().region([-75, -45, 20, 30, 0, 10, '2011-01', '2011-06']).data
ds = ds.argo.point2profile()


features_in_ds = {'temperature': 'TEMP', 'salinity': 'PSAL'}
features_zdim='DEPTH'
ds['TEMP'].attrs['feature_name'] = 'temperature'
ds['PSAL'].attrs['feature_name'] = 'salinity'
ds['N_LEVELS'].attrs['axis'] = 'Z'
m.fit(ds)



# Reset np.int to its original value
np.int = np.int_
