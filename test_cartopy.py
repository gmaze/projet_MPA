"""import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import cartopy.feature as cfeature

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Save the plot by calling plt.savefig() BEFORE plt.show()
plt.savefig('coastlines.pdf')
plt.savefig('coastlines.png')

plt.show()
"""
import argopy
import pandas as pd
import numpy as np
import xarray as xr
import gsw
from scipy.signal import filtfilt, butter

def lowpassFilt(dataVec, cutoff=10, deltaZ=1, removeNaNs=False):

    Wn = (2 * deltaZ) / cutoff  # Format is (2* delta_z)/low_pass
    b_Numerator, a_Denomoniator = butter(
        5, Wn
    )  # Fifth order butterworth filter with 10m low pass

    if removeNaNs:
        dataVec = dataVec[~np.isnan(dataVec)]

    return filtfilt(b_Numerator, a_Denomoniator, dataVec)



ds = argopy.DataFetcher(src="erddap").region([-75, -45, 20, 30, 0, 1000, "2010-01", "2010-12"]).to_xarray()
ds_profiles = ds.argo.point2profile()
num_profiles = ds_profiles.dims.get("N_PROF")
coords_names = list(ds_profiles.coords)
coords_names.remove("N_LEVELS")  # Levels is number of depth points, not profiles
coords_names.remove("TIME")
data_rows = []
for i in range(num_profiles):
    coord_dict = {
        ds_key: ds_profiles[ds_key][i].to_numpy() for ds_key in coords_names
    }
    data_dict = {
        ds_key: ds_profiles[ds_key][i].to_numpy()
        for ds_key in list(ds_profiles.keys())
    }
    time_dict = {"TIME": pd.to_datetime(ds_profiles["TIME"][i].values)}
    combined_dict = {**coord_dict, **data_dict, **time_dict}
    data_rows.append(combined_dict)





DF = pd.DataFrame(data_rows)

DF["Depth"] = pd.Series(dtype=object)
DF["Depth_uniform"] = pd.Series(dtype=object)
DF["TEMP_uniform"] = pd.Series(dtype=object)
DF["PRES_uniform"] = pd.Series(dtype=object)
DF["PSAL_uniform"] = pd.Series(dtype=object)
DF["Season"] = pd.Series(dtype="string")
DF["BV_cph"] = pd.Series(dtype=object)

interpVars = ["TEMP", "PRES", "PSAL"]
interpVarNames = ["TEMP_uniform", "PRES_uniform", "PSAL_uniform"]



# Loop over profiles
for i in range(DF.shape[0]):
    # Assign the season
    dayYear = DF["TIME"].iloc[i].dayofyear

    if dayYear in range(80, 172):
        season = "Spring"
    elif dayYear in range(172, 264):
        season = "Summer"
    elif dayYear in range(264, 355):
        season = "Fall"
    else:
        season = "Winter"
    DF["Season"].iloc[i] = season

    # Get the depth values
    depthVals = abs(gsw.z_from_p(DF["PRES"].iloc[i], DF["LATITUDE"].iloc[i]))
    DF["Depth"].iloc[i] = depthVals

    # Create the uniform depth profile
    depthUniform = np.linspace(0, 500, 500 + 1)
    DF["Depth_uniform"].iloc[i] = depthUniform

    # Interpolate onto 1m grid
    for idx, varName in enumerate(interpVars):
        varInterped = np.interp(depthUniform, depthVals, DF[varName].iloc[i])
        DF[interpVarNames[idx]].iloc[i] = varInterped


# zero pad the profiles
    PSAL_prof = np.pad(
        DF["PSAL_uniform"].iloc[i], 1, constant_values=DF["PSAL_uniform"].iloc[i][0]
    )
    Z_prof = np.pad(DF["Depth_uniform"].iloc[i], 1, constant_values=0)
    TEMP_prof = np.pad(
        DF["TEMP_uniform"].iloc[i], 1, constant_values=DF["TEMP_uniform"].iloc[i][0]
    )
    P_prof = np.pad(
        DF["PRES_uniform"].iloc[i], 1, constant_values=DF["PRES_uniform"].iloc[i][0]
    )

    # low pass filter the raw temp and salinity data

    SA = gsw.SA_from_SP(
        PSAL_prof, P_prof, DF["LONGITUDE"].iloc[i], DF["LATITUDE"].iloc[i]
    )

    # 2
    CT = gsw.CT_from_t(SA, TEMP_prof, P_prof)

    # 3 low pass filter the SA and CT profiles
    hasNaN = np.argwhere(np.isnan(SA) | np.isnan(CT))
    if len(hasNaN) > 0:
        firstNaN = int(hasNaN[0])
    else:
        firstNaN = len(SA)

    if firstNaN > 2:
        SA_filt = lowpassFilt(SA[0:firstNaN], cutoff=10, deltaZ=1)
        CT_filt = lowpassFilt(CT[0:firstNaN], cutoff=10, deltaZ=1)

        # 4
        N2, pmid = gsw.Nsquared(
            SA_filt, CT_filt, P_prof[0:firstNaN], lat=DF["LATITUDE"].iloc[i]
        )

        # 5
        BV_cph_pmid = (3600 / (2 * np.pi)) * np.sqrt(abs(N2))
        BV_cph_pmid[BV_cph_pmid < 0.3] = 0.3

        # 6
        zmid = abs(gsw.z_from_p(pmid, DF["LATITUDE"].iloc[i]))
        BV_cph = np.interp(
            depthUniform,
            zmid[(~np.isnan(BV_cph_pmid)) & np.isfinite(BV_cph_pmid)],
            BV_cph_pmid[(~np.isnan(BV_cph_pmid)) & np.isfinite(BV_cph_pmid)],
        )
        DF["BV_cph"].iloc[i] = BV_cph
    else:
        DF["BV_cph"].iloc[i] = np.nan(depthUniform.shape)

print(DF["Depth"])
