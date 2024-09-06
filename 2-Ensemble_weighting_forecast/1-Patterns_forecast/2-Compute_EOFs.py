# %% [markdown]
# # 2. Compute EOFs

# This script is used to compute the EOFs and PCs. 
# from monthly seasonal forescasts for the forecast and hindcast period.
# 
# EOFs are calculted for ERA5 data with the Eof library.
# Seasonal forecasts are then projected into these EOFs to obtain the PCs.
#
# First we have to decide a forecast system (institution and system name) and a start month. 

#%%
print("2. Compute EOFs") 

import os
import sys
from dotenv import load_dotenv
import xarray as xr
import pandas as pd
import numpy as np
from eofs.xarray import Eof
import locale
import calendar
from dateutil.relativedelta import relativedelta
import time
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# Read the file with model names
models = pd.read_csv('../../models.csv')

# If variables are introduced from the command line, read them
if len(sys.argv) > 2:
    institution = str(sys.argv[1]).replace('"', '')
    name = str(sys.argv[2]).replace('"', '')
    year = int(sys.argv[3])
    startmonth = int(sys.argv[4])
# If no variables were introduced, ask for them
else:
    # Which model institution
    institutions = [inst for inst in models.institution.unique()]
    institution = input(f"Usar modelo del siguiente organismo {institutions}: ")

    # Which model system
    names = [name for name in models[models["institution"]==institution]["name"]]
    name = input(f"Sistema del modelo {names}: ")

    # Which year of initialization
    year = int(input("Año de inicio del forecast: "))

    # Which start month
    startmonth = int(input("Mes de inicialización (en número): "))

# Save the simplier model and system name
model = str(models[(models["institution"]==institution) & (models["name"]==name)]["short_institution"].values[0])
system = str(models[(models["institution"]==institution) & (models["name"]==name)]["short_name"].values[0])

# Here we save the configuration
config = dict(
    list_vars = 'geopotential',
    pressure_level = '500',
    fcy = year,
    hcstarty = 1993,
    hcendy = 2016,
    start_month = startmonth,
    origin = model,
    system = system,
    isLagged = False if model in ['ecmwf', 'meteo_france', 'dwd', 'cmcc', 'eccc'] else True
)

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
HINDDIR = os.getenv('HIND_DIR')
FOREDIR = os.getenv('FORE_DIR')

MODES_HINDDIR = HINDDIR + '/modes'
SCORE_HINDDIR = HINDDIR + '/scores'
MODES_FOREDIR = FOREDIR + '/modes'
SCORE_FOREDIR = FOREDIR + '/scores'

# Directory creation
for directory in [MODES_HINDDIR, SCORE_HINDDIR, MODES_FOREDIR, SCORE_FOREDIR]:
    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

# Base name for hindcast
fcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_forecast{fcy}_monthly'.format(**config)
# File name for hindcast
fcst_fname = f'{FOREDIR}/{fcst_bname}.grib'
# Base name for hindcast
hcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)
# File name for hindcast
hcst_fname = f'{HINDDIR}/{hcst_bname}.grib'
# Base name for observations
obs_bname = 'era5_monthly_stmonth{start_month:02d}_{hcstarty}-{hcendy}'.format(**config)
# File name for observations
obs_fname = f'{HINDDIR}/{obs_bname}.grib'
# File name for climatologies
clim_fname = f'{HINDDIR}/era5_monthly.grib'
clim_fname_sf = f'{HINDDIR}/era5_monthly_sf.grib'

# Check if files exist
if not os.path.exists(obs_fname):
    print('No se descargaron aún los datos de ERA5')
    sys.exit()
elif not os.path.exists(hcst_fname):
    print('No se descargaron aún los datos de este modelo y sistema')
    sys.exit()
elif not os.path.exists(fcst_fname):
    print('No se descargaron aún los datos de este modelo y sistema')
    sys.exit()
elif not os.path.exists(clim_fname):
    print('No se descargaron aún los datos de ERA5')
    sys.exit()
elif not os.path.exists(clim_fname_sf):
    print('No se descargaron aún los datos de ERA5')
    sys.exit()



# %% [markdown]
# ## 2.1 Hindcast anomalies

# We calculate the monthly and 3-months anomalies for the hindcast data.

#%%
print("2.1 Hindcast anomalies")

# For the re-shaping of time coordinates in xarray.Dataset we need to select the right one 
#  -> burst mode ensembles (e.g. ECMWF SEAS5) use "time". This is the default option in this notebook
#  -> lagged start ensembles (e.g. MetOffice GloSea6) use "indexing_time" (see CDS documentation about nominal start date)
st_dim_name = 'time' if not config.get('isLagged',False) else 'indexing_time'

# Reading hindcast data from file
hcst = xr.open_dataset(hcst_fname,engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', st_dim_name)))
# We use dask.array with chunks on leadtime, latitude and longitude coordinate
hcst = hcst.chunk({'forecastMonth':1, 'latitude':'auto', 'longitude':'auto'})  #force dask.array using chunks on leadtime, latitude and longitude coordinate
# Reanme coordinates to match those of observations
hcst = hcst.rename({'latitude':'lat','longitude':'lon', st_dim_name:'start_date'})

# Add start_month to the xr.Dataset
start_month = pd.to_datetime(hcst.start_date.values[0]).month
hcst = hcst.assign_coords({'start_month':start_month})
# Add valid_time to the xr.Dataset
vt = xr.DataArray(dims=('start_date','forecastMonth'), coords={'forecastMonth':hcst.forecastMonth,'start_date':hcst.start_date})
vt.data = [[pd.to_datetime(std)+relativedelta(months=fcmonth-1) for fcmonth in vt.forecastMonth.values] for std in vt.start_date.values]
hcst = hcst.assign_coords(valid_time=vt)

# Calculate 3-month aggregations
hcst_3m = hcst.rolling(forecastMonth=3).mean()
# rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped
hcst_3m = hcst_3m.where(hcst_3m.forecastMonth>=3,drop=True)

# Calculate 1m anomalies
hcmean = hcst.mean(['number','start_date'])
hcanom = hcst - hcmean
hcanom = hcanom.assign_attrs(reference_period='{hcstarty}-{hcendy}'.format(**config))
# Calculate 3m anomalies
hcmean_3m = hcst_3m.mean(['number','start_date'])
hcanom_3m = hcst_3m - hcmean_3m
hcanom_3m = hcanom_3m.assign_attrs(reference_period='{hcstarty}-{hcendy}'.format(**config))

# %% [markdown]
# ## 2.2 Forecast anomalies

# We calculate the monthly and 3-months anomalies for the forecast data.

#%%
print("2.2 Forecast anomalies")

# For the re-shaping of time coordinates in xarray.Dataset we need to select the right one 
#  -> burst mode ensembles (e.g. ECMWF SEAS5) use "time". This is the default option in this notebook
#  -> lagged start ensembles (e.g. MetOffice GloSea6) use "indexing_time" (see CDS documentation about nominal start date)
st_dim_name = 'time' if not config.get('isLagged',False) else 'indexing_time'

# Reading hindcast data from file
fcst = xr.open_dataset(fcst_fname,engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', st_dim_name)))
# We use dask.array with chunks on leadtime, latitude and longitude coordinate
fcst = fcst.chunk({'forecastMonth':1, 'latitude':'auto', 'longitude':'auto'})  #force dask.array using chunks on leadtime, latitude and longitude coordinate
# Reanme coordinates to match those of observations
fcst = fcst.rename({'latitude':'lat','longitude':'lon', st_dim_name:'start_date'})

# Add start_month to the xr.Dataset
start_month = pd.to_datetime(fcst.start_date.values).month
fcst = fcst.assign_coords({'start_month':start_month})
# Add valid_time to the xr.Dataset
vt = xr.DataArray(dims=('forecastMonth',), coords={'forecastMonth': fcst.forecastMonth})
vt.data = [pd.to_datetime(fcst.start_date.values)+relativedelta(months=fcmonth-1) for fcmonth in fcst.forecastMonth.values]
fcst = fcst.assign_coords(valid_time=vt)

# Calculate 3-month aggregations
fcst_3m = fcst.rolling(forecastMonth=3).mean()
# rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped
fcst_3m = fcst_3m.where(fcst_3m.forecastMonth>=3,drop=True)

# Calculate 1m anomalies
fcanom = fcst - hcmean
# Calculate 3m anomalies
fcanom_3m = fcst_3m - hcmean_3m

# %% [markdown]
# ## 2.3 Observations anomalies

# We calculate the monthly and 3-months anomalies for the ERA5 data.

#%%
print("2.3 Observations anomalies")  

# Reading observations from file
era5_1deg = xr.open_dataset(obs_fname, engine='cfgrib')

# Renaming to match hindcast names 
era5_1deg = era5_1deg.rename({'latitude':'lat','longitude':'lon','time':'start_date'}).swap_dims({'start_date':'valid_time'})
# Assign 'forecastMonth' coordinate values
fcmonths = [mm+1 if mm>=0 else mm+13 for mm in [t.month - config['start_month'] for t in pd.to_datetime(era5_1deg.valid_time.values)] ]
era5_1deg = era5_1deg.assign_coords(forecastMonth=('valid_time',fcmonths))
# Drop obs values not needed (earlier than first start date) - this is useful to create well shaped 3-month aggregations from obs.
era5_1deg = era5_1deg.where(era5_1deg.valid_time>=np.datetime64('{hcstarty}-{start_month:02d}-01'.format(**config)),drop=True)

# Calculate 3-month aggregations
era5_1deg_3m = era5_1deg.rolling(valid_time=3).mean()
# rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped
era5_1deg_3m = era5_1deg_3m.where(era5_1deg_3m.forecastMonth>=3,drop=True)

# As we don't need it anymore at this stage, we can safely remove 'forecastMonth'
era5_1deg = era5_1deg.drop('forecastMonth')
era5_1deg_3m = era5_1deg_3m.drop('forecastMonth')

# Calculate 1m anomalies
obmean = era5_1deg.groupby('valid_time.month').mean('valid_time')
obanom = era5_1deg.groupby('valid_time.month') - obmean
# Calculate 3m anomalies
obmean_3m = era5_1deg_3m.groupby('valid_time.month').mean('valid_time')
obanom_3m = era5_1deg_3m.groupby('valid_time.month') - obmean_3m

# We can remove unneeded coordinates
obanom = obanom.drop(['number', 'step', 'start_date', 'isobaricInhPa', 'month'])
obanom_3m = obanom_3m.drop(['number', 'step', 'start_date', 'isobaricInhPa', 'month'])

# %% [markdown]
# ## 2.4 Climatology anomalies

# We calculate the monthly and 3-months anomalies for the ERA5 climatological data.

#%%
print("2.4 Climatology anomalies")  

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

# Reading climatology from file | This is for projecting the eofs (corresponding 3-months season)
era5_clim = xr.open_dataset(clim_fname, engine='cfgrib')
# Obtain desired months
d_months = [(config['start_month']+leadm)%12 if config['start_month']+leadm!=12 else 12 for leadm in range(6)]
# Subsetting data
era5_clim = era5_clim.where(
          #era5_clim.time.dt.year > 1978, # &
          era5_clim.time.dt.month.isin(d_months),
          drop = True).sel(time=slice('1941-{:02d}-01'.format(d_months[0]),'2020-{:02d}-01'.format(d_months[-1]))).rename({'latitude':'lat','longitude':'lon'})
# Assign 'forecastMonth' coordinate values
fcmonths = [mm+1 if mm>=0 else mm+13 for mm in [t.month - config['start_month'] for t in pd.to_datetime(era5_clim.valid_time.values)] ]
era5_clim = era5_clim.assign_coords(forecastMonth=('time',fcmonths))
# Calculate 3-month aggregations
era5_clim_3m = era5_clim.rolling(time=3).mean()
# rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped
era5_clim_3m = era5_clim_3m.where(era5_clim_3m.forecastMonth>=3, drop=True).drop(['number', 'step'])
# Calculate 1m anomalies
clmean = era5_clim.groupby('time.month').mean('time')
clanom_pl = era5_clim.groupby('time.month') - clmean
# Calculate 3m anomalies
clmean_3m = era5_clim_3m.mean('time')
clanom_pl_3m = era5_clim_3m - clmean_3m

# Reading surface climatology from file
era5_clim_sf_notp = xr.open_dataset(clim_fname_sf, engine='cfgrib', backend_kwargs={'filter_by_keys': {'step': 0}})
era5_clim_sf_tp = xr.open_dataset(clim_fname_sf, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})
era5_clim_sf_tp = era5_clim_sf_tp.assign_coords(time=era5_clim_sf_notp.time.values)
era5_clim_sf = xr.merge([era5_clim_sf_notp,era5_clim_sf_tp],compat='override')
del era5_clim_sf_notp, era5_clim_sf_tp
# Obtain desired months
d_months = [(config['start_month']+leadm)%12 if config['start_month']+leadm!=12 else 12 for leadm in range(6)]
# Subsetting data
era5_clim_sf = era5_clim_sf.where(
          #era5_clim.time.dt.year > 1978, # &
          era5_clim_sf.time.dt.month.isin(d_months),
          drop = True).sel(time=slice('1941-{:02d}-01'.format(d_months[0]),'2020-{:02d}-01'.format(d_months[-1]))).rename({'latitude':'lat','longitude':'lon'})
# Assign 'forecastMonth' coordinate values
fcmonths = [mm+1 if mm>=0 else mm+13 for mm in [t.month - config['start_month'] for t in pd.to_datetime(era5_clim_sf.valid_time.values)] ]
era5_clim_sf = era5_clim_sf.assign_coords(forecastMonth=('time',fcmonths))

# Calculate 3-month aggregations
era5_clim_sf_3m = era5_clim_sf.rolling(time=3).mean()
# rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped
era5_clim_sf_3m = era5_clim_sf_3m.where(era5_clim_sf_3m.forecastMonth>=3, drop=True).drop(['number', 'step'])

# Calculate 1m anomalies
clmean_sf = era5_clim_sf.groupby('time.month').mean('time')
clanom_sf = era5_clim_sf.groupby('time.month') - clmean_sf
# Calculate 3m anomalies
clmean_sf_3m = era5_clim_sf_3m.groupby('time.month').mean('time')
clanom_sf_3m = era5_clim_sf_3m.groupby('time.month') - clmean_sf_3m

# %% [markdown]
# ## 2.5  Climatological EOF and PCs

# We calculate the EOFs for the climatological ERA5 data, and their corresponding PCs.
# 
# To calculate the EOFs, we have to weight each grid point by its area.

#%%
print("2.5 Climatological EOF and PCs")  

# Square-root of cosine of latitude weights are applied before the computation of EOFs
coslat = np.cos(np.deg2rad(clanom_3m.coords['lat'].values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]

list_solvers = {"forecastMonth":[],"validMonth":[],"EOFs":[]} 
# For each forecast month
for f in hcanom.forecastMonth:
    # Create an EOF solver to do the EOF analysis.
    cl_array = clanom_3m.copy() #.where(clanom.forecastMonth==f, drop=True)
    cl_array_pl = clanom_pl.where(clanom_pl.forecastMonth==f, drop=True)
    solver = Eof(cl_array.z, weights=wgts)
    list_solvers["forecastMonth"].append(int(f.values))
    list_solvers["validMonth"].append(int(cl_array_pl['valid_time.month'].values[0]))
    list_solvers["EOFs"].append(solver)

    validmonth = config['start_month']+f if config['start_month']+f<=12 else config['start_month']+f-12
    explained_var = solver.varianceFraction()
    variance = pd.DataFrame({'EOF': range(1,5), 'VAR':explained_var[0:4]})
    variance.to_csv(f'{MODES_HINDDIR}/ERA5_VAR_{validmonth:02d}.csv')


list_solvers_3m = {"forecastMonth":[],"validMonth":[],"EOFs":[]} 
# For each forecast month
for f in hcanom_3m.forecastMonth:
    # Create an EOF solver to do the EOF analysis.
    cl_array = clanom_3m.copy() #.where(clanom_3m.forecastMonth==f, drop=True)
    cl_array_pl = clanom_pl_3m.where(clanom_pl_3m.forecastMonth==f, drop=True)
    cl_array_sf = clanom_sf_3m.where(clanom_sf_3m.forecastMonth==f, drop=True)
    solver = Eof(cl_array.z, weights=wgts)
    list_solvers_3m["forecastMonth"].append(int(f.values))
    list_solvers_3m["validMonth"].append(int(cl_array_pl['valid_time.month'].values[0]))
    list_solvers_3m["EOFs"].append(solver)
    # Retrieve the leading EOF, expressed as the correlation between the leading PC time series and the input SLP anomalies at each grid point
    eofs_corr = solver.eofsAsCorrelation(neofs=4)
    explained_var = solver.varianceFraction()
    # Extracting the principal component
    #pcs = solver.pcs(npcs=4, pcscaling=1)
    pcs = solver.projectField(cl_array_pl.z, neofs=4, eofscaling=1)

    # Months string
    locale.setlocale(locale.LC_ALL, 'en_GB')
    validmonths = [vm if vm<=12 else vm-12 for vm in [config['start_month'] + (int(f.values)-1) - shift for shift in range(3)]]
    validmonths = [calendar.month_abbr[vm][0] for vm in reversed(validmonths)]
    tit_line = "".join(validmonths)

    variance = pd.DataFrame({'EOF': range(1,5), 'VAR':explained_var[0:4]})
    variance.to_csv(f'{MODES_HINDDIR}/ERA5_VAR_{tit_line}.csv')

# %% [markdown]
# ## 2.6  Hindcast PCs

# We project the seasonal forecast data into the four main ERA5 EOFs, 
# obtaining the main climate variability modes: NAO, EA, EA/WR and SCA.

#%%
print("2.6 Hindcast PCs")

if not (os.path.exists(f'{MODES_HINDDIR}/{hcst_bname}.1m.PCs.nc') & os.path.exists(f'{MODES_HINDDIR}/{hcst_bname}.3m.PCs.nc')):
    print('- Computing PCs for 1m aggregation (hindcast)')
    number_of_eofs = 4
    nn = 0
    list1_hcpcs = list()
    TIME = time.time()
    hcanom = hcanom.load()
    # For each model member
    for n in hcanom.number:
        print('{:.2f}%'.format((float(nn)/float(hcanom.number.size)*100.)))
        list2_hcpcs = list() 
        # For each forecast month
        for f in hcanom.forecastMonth:
            list3_hcpcs = list() 
            # For each year
            for t in hcanom.start_date:
                # Project the z500hPa field in the EOF solver
                solver = list_solvers["EOFs"][list_solvers["forecastMonth"].index(int(f.values))]
                pcs = solver.projectField(hcanom.sel(number=n,forecastMonth=f,start_date=t).z, neofs=number_of_eofs, eofscaling=1)
                list3_hcpcs.append(pcs.assign_coords({'start_date':t}))
            list3 = xr.concat(list3_hcpcs,dim='start_date')                    
            list2_hcpcs.append(list3.assign_coords({'forecastMonth':f}))
        list2 = xr.concat(list2_hcpcs,dim='forecastMonth')                    
        list1_hcpcs.append(list2.assign_coords({'number':n}))
        nn+=1
    hcpcs = xr.concat(list1_hcpcs,dim='number').assign_coords({'valid_time':hcanom.valid_time})
    print("TIME: --- {} ---".format(dt.timedelta(seconds=(time.time() - TIME)))) 

    print('- Computing PCs for 3m aggregation (hindcast)')
    nn = 0
    list1_hcpcs = list()
    TIME = time.time()
    hcanom_3m = hcanom_3m.load()
    # For each model member
    for n in hcanom_3m.number:
        print('{:.2f}%'.format((float(nn)/float(hcanom_3m.number.size)*100.)))
        list2_hcpcs = list()
        # For each forecast month
        for f in hcanom_3m.forecastMonth:
            list3_hcpcs = list()
            # For each year
            for t in hcanom_3m.start_date:
                # Project the z500hPa field in the EOF solver
                solver = list_solvers_3m["EOFs"][list_solvers_3m["forecastMonth"].index(int(f.values))]
                pcs = solver.projectField(hcanom_3m.sel(number=n,forecastMonth=f,start_date=t).z, neofs=number_of_eofs, eofscaling=1)
                list3_hcpcs.append(pcs.assign_coords({'start_date':t}))
            list3 = xr.concat(list3_hcpcs,dim='start_date')                    
            list2_hcpcs.append(list3.assign_coords({'forecastMonth':f}))
        list2 = xr.concat(list2_hcpcs,dim='forecastMonth')                    
        list1_hcpcs.append(list2.assign_coords({'number':n}))
        nn+=1
    hcpcs_3m = xr.concat(list1_hcpcs,dim='number').assign_coords({'valid_time':hcanom_3m.valid_time})
    print("TIME: --- {} ---".format(dt.timedelta(seconds=(time.time() - TIME)))) 

    # Saving pcs to netCDF files
    hcpcs.to_netcdf(f'{MODES_HINDDIR}/{hcst_bname}.1m.PCs.nc')
    hcpcs_3m.to_netcdf(f'{MODES_HINDDIR}/{hcst_bname}.3m.PCs.nc')
else:
    print(f'Las PCs del hindcast ya están calculadas')

# %% [markdown]
# ## 2.7  Forecast PCs

# We project the seasonal forecast data into the four main ERA5 EOFs, 
# obtaining the main climate variability modes: NAO, EA, EA/WR and SCA.

#%%
print("2.7 Forecast PCs")  

if not (os.path.exists(f'{MODES_FOREDIR}/{fcst_bname}.1m.PCs.nc') & os.path.exists(f'{MODES_FOREDIR}/{fcst_bname}.3m.PCs.nc')):
    print('- Computing PCs for 1m aggregation (forecast)')
    number_of_eofs = 4
    nn = 0
    list1_fcpcs = list()
    TIME = time.time()
    fcanom = fcanom.load()
    # For each model member
    for n in fcanom.number:
        print('{:.2f}%'.format((float(nn)/float(fcanom.number.size)*100.)))
        list2_fcpcs = list() 
        # For each forecast month
        for f in fcanom.forecastMonth:
            # Project the z500hPa field in the EOF solver
            solver = list_solvers["EOFs"][list_solvers["forecastMonth"].index(int(f.values))]
            pcs = solver.projectField(fcanom.sel(number=n,forecastMonth=f).z, neofs=number_of_eofs, eofscaling=1)
            list2_fcpcs.append(pcs.assign_coords({'forecastMonth':f}))
        list2 = xr.concat(list2_fcpcs,dim='forecastMonth')                    
        list1_fcpcs.append(list2.assign_coords({'number':n}))
        nn+=1
    fcpcs = xr.concat(list1_fcpcs,dim='number').assign_coords({'valid_time':fcanom.valid_time})
    print("TIME: --- {} ---".format(dt.timedelta(seconds=(time.time() - TIME)))) 

    print('- Computing PCs for 3m aggregation (forecast)')
    nn = 0
    list1_fcpcs = list()
    TIME = time.time()
    fcanom_3m = fcanom_3m.load()
    # For each model member
    for n in fcanom_3m.number:
        print('{:.2f}%'.format((float(nn)/float(fcanom_3m.number.size)*100.)))
        list2_fcpcs = list()
        # For each forecast month
        for f in fcanom_3m.forecastMonth:
            # Project the z500hPa field in the EOF solver
            solver = list_solvers_3m["EOFs"][list_solvers_3m["forecastMonth"].index(int(f.values))]
            pcs = solver.projectField(fcanom_3m.sel(number=n,forecastMonth=f).z, neofs=number_of_eofs, eofscaling=1)
            list2_fcpcs.append(pcs.assign_coords({'forecastMonth':f}))
        list2 = xr.concat(list2_fcpcs,dim='forecastMonth')                    
        list1_fcpcs.append(list2.assign_coords({'number':n}))
        nn+=1
    fcpcs_3m = xr.concat(list1_fcpcs,dim='number').assign_coords({'valid_time':fcanom_3m.valid_time})
    print("TIME: --- {} ---".format(dt.timedelta(seconds=(time.time() - TIME)))) 

    # Saving pcs to netCDF files
    fcpcs.to_netcdf(f'{MODES_FOREDIR}/{fcst_bname}.1m.PCs.nc')
    fcpcs_3m.to_netcdf(f'{MODES_FOREDIR}/{fcst_bname}.3m.PCs.nc')
else:
    print(f'Las PCs del forecast ya están calculadas')

# %% [markdown]
# ## 2.8  Observed PCs

# The same as in the previous section but with the ERA5 data.

#%%
print("2.8 Obseved PCs")

if not (os.path.exists(f'{MODES_HINDDIR}/{obs_bname}.1m.PCs.nc') & os.path.exists(f'{MODES_HINDDIR}/{obs_bname}.3m.PCs.nc')):
    print('- Computing PCs for 1m aggregation (observation)')
    list1_obpcs = list()
    obanom = obanom.load()
    # For each year
    for t in obanom.valid_time:
        # Project the z500hPa field in the EOF solver
        solver = list_solvers["EOFs"][list_solvers["validMonth"].index(int(t['valid_time.month'].values))]
        pcs = solver.projectField(obanom.sel(valid_time=t).z, neofs=number_of_eofs, eofscaling=1)
        list1_obpcs.append(pcs.assign_coords({'valid_time':t}))
    obpcs = xr.concat(list1_obpcs,dim='valid_time')      

    print('- Computing PCs for 3m aggregation (observation)')
    list1_obpcs = list() 
    obanom_3m = obanom_3m.load()
    # For each year
    for t in obanom_3m.valid_time:
        # Project the z500hPa field in the EOF solver
        solver = list_solvers_3m["EOFs"][list_solvers_3m["validMonth"].index(int(t['valid_time.month'].values))]
        pcs = solver.projectField(obanom_3m.sel(valid_time=t).z, neofs=number_of_eofs, eofscaling=1)
        list1_obpcs.append(pcs.assign_coords({'valid_time':t}))
    obpcs_3m = xr.concat(list1_obpcs,dim='valid_time')                   

    # Saving pcs to netCDF files
    obpcs.to_netcdf(f'{MODES_HINDDIR}/{obs_bname}.1m.PCs.nc')
    obpcs_3m.to_netcdf(f'{MODES_HINDDIR}/{obs_bname}.3m.PCs.nc')
else:
    print(f'Las PCs de la observación ya están calculadas')
