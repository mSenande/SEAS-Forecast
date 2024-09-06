# %% [markdown]
# # 2. Compute EOFs

# This script is used to compute EOFs and PCs. 
# from observational data for the forecast and hindcast period.
# 
# EOFs are calculted for ERA5 data with the Eof library.
# Seasonal forecasts are then projected into these EOFs to obtain the PCs.
#
# First we have to decide a valid year and month. 

#%%
print("2. Compute EOFs") 

import os
import sys
from dotenv import load_dotenv
import xarray as xr
import pandas as pd
import numpy as np
from eofs.xarray import Eof
import warnings
warnings.filterwarnings('ignore')

# If variables are introduced from the command line, read them
if len(sys.argv) > 2:
    endyear = int(sys.argv[1])
    aggr = str(sys.argv[2])
    endmonth = int(sys.argv[3])
    if aggr=='1m':
        # Valid season/month name
        meses = np.array(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])[endmonth-1]
    elif aggr=='3m':
        # Valid season/month name
        meses = np.array(['NDJ', 'DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND'])[endmonth-1]
# If no variables were introduced, ask for them
else:
    # Valid year 
    endyear = input("Resultados para el año: ")

    # Subset of plots to be produced
    aggr = input("Selecciona el tipo de agregación mensual [ 1m , 3m ]: ")

    if aggr=='1m':
        # Valid season/month name
        meses = input("Resultados para el mes [ Jan , Feb , Mar , Apr , May , Jun , Jul , Aug , Sep , Oct , Nov , Dec ]: ")
        # Valid season/month number
        endmonth = np.where(np.array(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']) == meses)[0][0]+1
    elif aggr=='3m':
        # Valid season/month name
        meses = input("Resultados para el trimestre [ NDJ , DJF , JFM , FMA , MAM , AMJ , MJJ , JJA , JAS , ASO , SON , OND ]: ")
        # Valid season/month number
        endmonth = np.where(np.array(['NDJ', 'DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']) == meses)[0][0]+1

# Here we save the configuration
config = dict(
    list_vars = 'geopotential',
    pressure_level = '500',
    endyear = endyear,
    hcstarty = 1993,
    hcendy = 2016,
    endmonth = endmonth,
)

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
HINDDIR = os.getenv('HIND_DIR')
FOREDIR = os.getenv('FORE_DIR')

# Which months do we have to read?
if aggr=='1m':
    months = [endmonth]
    years = [int(endyear)]
elif aggr=='3m':
    months = [endmonth-m if endmonth-m>=1 else endmonth-m+12 for m in reversed(range(3))]
    years = [int(endyear) if endmonth-m>=1 else int(endyear)-1 for m in reversed(range(3))]

# We check which files we have to read
val_fnames = []
for m in range(len(months)):
    month = months[m]
    year = years[m]

    # Base name for validation
    val_bname = f'era5_valid{year}-{month:02d}_monthly'
    # File name for validation
    val_fname = f'{FOREDIR}/{val_bname}.grib'
    val_fnames = val_fnames+[val_fname]

# File name for climatologies
clim_fname = f'{HINDDIR}/era5_monthly.grib'
clim_fname_sf = f'{HINDDIR}/era5_monthly_sf.grib'


# %% [markdown]
# ## 2.1 Observations anomalies

# We calculate the anomalies for the ERA5 data.

#%%
print("2.1 Observation anomalies")

# Reading observations from file
if aggr=='1m':
    era5_vali= xr.open_dataset(val_fnames, engine='cfgrib').rename({'latitude':'lat','longitude':'lon'})
elif aggr=='3m':
    era5_vali= xr.open_mfdataset(val_fnames, engine='cfgrib', combine='nested', concat_dim='time').rename({'latitude':'lat','longitude':'lon'})

# Reading climatology from file
era5_clim = xr.open_dataset(clim_fname, engine='cfgrib')
# Subsetting data
era5_clim = era5_clim.where(
          #era5_clim.time.dt.year > 1978, # &
          era5_clim.time.dt.month.isin(months),
          drop = True).sel(time=slice('1993-{:02d}-01'.format(months[0]),'2017-{:02d}-01'.format(months[-1]))).rename({'latitude':'lat','longitude':'lon'})

if aggr=='1m':
    clmean = era5_clim.mean('time')
    valianom = era5_vali - clmean
    hindanom = era5_clim - clmean
elif aggr=='3m':
    # Calculate 3-month aggregations
    era5_vali = era5_vali.rolling(time=3).mean()
    era5_vali = era5_vali.where(era5_vali['time.month']==months[-1], drop=True)
    era5_clim = era5_clim.rolling(time=3).mean()
    era5_clim = era5_clim.where(era5_clim['time.month']==months[-1], drop=True)
    # Calculate anomalies
    clmean = era5_clim.mean('time')
    valianom = era5_vali - clmean
    hindanom = era5_clim - clmean

# %% [markdown]
# ## 2.2 Climatology anomalies

# We calculate the monthly and 3-months anomalies for the ERA5 climatological data.

#%%
print("2.2 Climatology anomalies")  

# Reading climatology from file | This is for obtaining the eofs (just DJF)
era5_clim = xr.open_dataset(clim_fname, engine='cfgrib')
# Obtain desired months
d_months = [12, 1, 2]
# Subsetting data
era5_clim = era5_clim.where(
          #era5_clim.time.dt.year > 1978, # &
          era5_clim.time.dt.month.isin(d_months),
          drop = True).sel(time=slice('1941-{:02d}-01'.format(d_months[0]),'2020-{:02d}-01'.format(d_months[-1]))).rename({'latitude':'lat','longitude':'lon'})
# Calculate 3-month aggregations
era5_clim_3m = era5_clim.rolling(time=3).mean()
era5_clim_3m = era5_clim_3m.where(era5_clim_3m['time.month']==2, drop=True).drop(['number', 'step'])
# Calculate 3m anomalies
clmean_3m = era5_clim_3m.mean('time')
clanom_3m = era5_clim_3m - clmean_3m

# %% [markdown]
# ## 2.3  Climatological EOF and PCs

# We calculate the EOFs for the climatological ERA5 data.
# 
# To calculate the EOFs, we have to weight each grid point by its area.

#%%
print("2.3 Climatological EOF")  

# Square-root of cosine of latitude weights are applied before the computation of EOFs
coslat = np.cos(np.deg2rad(clanom_3m.coords['lat'].values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]

# Create an EOF solver to do the EOF analysis.
cl_array = clanom_3m.copy() #.where(clanom.forecastMonth==f, drop=True)
solver = Eof(cl_array.z, weights=wgts)

# %% [markdown]
# ## 2.4  Obtain PCs

# We project the ERA5 monthly data into the four main ERA5 EOFs, 
# obtaining the main climate variability modes: NAO, EA, EA/WR and SCA.

#%%
print("2.4 Obtain PCs")  

number_of_eofs = 4
pcs_validation = solver.projectField(valianom.z, neofs=number_of_eofs, eofscaling=1)
pcs_hindcast = solver.projectField(hindanom.z, neofs=number_of_eofs, eofscaling=1)

print(f'{endyear}-{meses}')
print(f'EOF1 - NAO value: {pcs_validation.sel(mode=0).values[0]:.2f}')
print(f'EOF2 - EA value: {pcs_validation.sel(mode=1).values[0]:.2f}')
print(f'EOF3 - EAWR value: {pcs_validation.sel(mode=2).values[0]:.2f}')
print(f'EOF4 - SCA value: {pcs_validation.sel(mode=3).values[0]:.2f}')
