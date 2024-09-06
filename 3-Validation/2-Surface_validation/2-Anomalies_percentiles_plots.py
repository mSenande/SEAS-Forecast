# %% [markdown]
# # 3. Plot forecasts

# This script creates different plots for a specific forecast.
# 
# First we have to decide a forecast system (institution and system name), a start year and month, a month aggregation, a region and a variable. 

#%%
print("3. Plot forecasts")

import os
import sys
from dotenv import load_dotenv
import pandas as pd
import xarray as xr
import numpy as np
from scipy import stats
from dateutil.relativedelta import relativedelta
import matplotlib
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

# Read the file with model names
models = pd.read_csv('../../models.csv')

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
    elif aggr=='5m':
        # Valid season/month name
        meses = np.array(['SONDJ' , 'ONDJF' , 'NDJFM' , 'DJFMA' , 'JFMAM' , 'FMAMJ' , 'MAMJJ' , 'AMJJA' , 'MJJAS' , 'JJASO' , 'JASON' , 'ASOND' ])[endmonth-1]
    region = str(sys.argv[4])
    reference = str(sys.argv[5])

# If no variables were introduced, ask for them
else:
    # Valid year 
    endyear = input("Resultados para el año: ")

    # Subset of plots to be produced
    aggr = input("Selecciona el tipo de agregación mensual [ 1m , 3m , 5m ]: ")

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
    elif aggr=='5m':
        # Valid season/month name
        meses = input("Resultados para los meses [ SONDJ , ONDJF , NDJFM , DJFMA , JFMAM , FMAMJ , MAMJJ , AMJJA , MJJAS , JJASO , JASON , ASOND ]: ")
        # Valid season/month number
        endmonth = np.where(np.array(['SONDJ','ONDJF','NDJFM','DJFMA','JFMAM','FMAMJ','MAMJJ','AMJJA','MJJAS','JJASO','JASON','ASOND']) == meses)[0][0]+1

    # Region selected for plots
    region = input("Selecciona qué región representar [ Iberia , MedCOF ]: ")

    # Reference period for plots
    reference = input("Selecciona qué periodo de referencia usar [ Eg. 1993-2016 ]: ")

# Here we save the configuration
config = dict(
    list_vars = ['2m_temperature', 'total_precipitation'],
    endyear = endyear,
    starty = reference.split('-')[0],
    endy = reference.split('-')[1],
    endmonth = endmonth,
)

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
HINDDIR = os.getenv('HIND_DIR') # Directory where hindcast surface data files are located
FOREDIR = os.getenv('FORE_DIR') # Directory where forecast surface data files are located

# Which months do we have to read?
if aggr=='1m':
    months = [endmonth]
    years = [int(endyear)]
elif aggr=='3m':
    months = [endmonth-m if endmonth-m>=1 else endmonth-m+12 for m in reversed(range(3))]
    years = [int(endyear) if endmonth-m>=1 else int(endyear)-1 for m in reversed(range(3))]
elif aggr=='5m':
    months = [endmonth-m if endmonth-m>=1 else endmonth-m+12 for m in reversed(range(5))]
    years = [int(endyear) if endmonth-m>=1 else int(endyear)-1 for m in reversed(range(5))]

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
clim_fname = f'{HINDDIR}/era5_monthly_sf.grib'

# We select a directory to save the results
PLOTSDIR = f'./PLOTS/{endyear}-{endmonth:02d}_{meses}'
# Check if directory exists
if not os.path.exists(PLOTSDIR):
    # If it doesn't exist, create it
    try:
        os.system(f'mkdir -p {PLOTSDIR}')
    except FileExistsError:
        pass

# %% [markdown]
# ## 2.1 Observations anomalies

# We calculate the anomalies for the ERA5 data.

#%%
print("2.1 Observation anomalies")

# Temperature
# Reading observations from file
if aggr=='1m':
    era5_vali_t2m= xr.open_dataset(val_fnames[0], engine='cfgrib', backend_kwargs={'filter_by_keys': {'step': 0}}).rename({'latitude':'lat','longitude':'lon'})['t2m']
else:
    era5_vali_t2m= xr.open_mfdataset(val_fnames, engine='cfgrib', combine='nested', concat_dim='time', backend_kwargs={'filter_by_keys': {'step': 0}}).rename({'latitude':'lat','longitude':'lon'})['t2m']
# Reading climatology from file
era5_clim_t2m = xr.open_dataset(clim_fname, engine='cfgrib', backend_kwargs={'filter_by_keys': {'step': 0}})['t2m']
# Subsetting data
era5_clim_t2m = era5_clim_t2m.where(
          #era5_clim.time.dt.year > 1978, # &
          era5_clim_t2m.time.dt.month.isin(months),
          drop = True).sel(time=slice('{}-{:02d}-01'.format(config['starty'],months[0]),'{}-{:02d}-01'.format(int(config['endy'])+1,months[-1]))).rename({'latitude':'lat','longitude':'lon'})
if aggr=='1m':
    clmean_t2m = era5_clim_t2m.mean('time')
    valianom_t2m = era5_vali_t2m - clmean_t2m
    valianom_t2m = valianom_t2m.squeeze()
elif aggr=='3m':
    # Calculate 3-month aggregations
    era5_vali_t2m = era5_vali_t2m.rolling(time=3).mean()
    era5_vali_t2m = era5_vali_t2m.where(era5_vali_t2m['time.month']==months[-1], drop=True)
    era5_clim_t2m = era5_clim_t2m.rolling(time=3).mean()
    era5_clim_t2m = era5_clim_t2m.where(era5_clim_t2m['time.month']==months[-1], drop=True)
    # Calculate anomalies
    clmean_t2m = era5_clim_t2m.mean('time')
    valianom_t2m = era5_vali_t2m - clmean_t2m
    valianom_t2m = valianom_t2m.squeeze()
elif aggr=='5m':
    # Calculate 5-month aggregations
    era5_vali_t2m = era5_vali_t2m.rolling(time=5).mean()
    era5_vali_t2m = era5_vali_t2m.where(era5_vali_t2m['time.month']==months[-1], drop=True)
    era5_clim_t2m = era5_clim_t2m.rolling(time=5).mean()
    era5_clim_t2m = era5_clim_t2m.where(era5_clim_t2m['time.month']==months[-1], drop=True)
    # Calculate anomalies
    clmean_t2m = era5_clim_t2m.mean('time')
    valianom_t2m = era5_vali_t2m - clmean_t2m
    valianom_t2m = valianom_t2m.squeeze()

# Precipitation
# Reading observations from file
if aggr=='1m':
    era5_vali_tp= xr.open_dataset(val_fnames[0], engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'tp'}}).rename({'latitude':'lat','longitude':'lon'})['tp']
    era5_vali_tp['time'] = era5_vali_tp['valid_time']
else:
    era5_vali_tp= xr.open_mfdataset(val_fnames, engine='cfgrib', combine='nested', concat_dim='time', backend_kwargs={'filter_by_keys': {'shortName': 'tp'}}).rename({'latitude':'lat','longitude':'lon'})['tp']
    era5_vali_tp['time'] = era5_vali_tp['valid_time']
# Reading climatology from file
era5_clim_tp = xr.open_dataset(clim_fname, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})['tp']
era5_clim_tp['time'] = era5_clim_tp['valid_time']
# Subsetting data
era5_clim_tp = era5_clim_tp.where(
          #era5_clim.time.dt.year > 1978, # &
          era5_clim_tp.time.dt.month.isin(months),
          drop = True).sel(time=slice('{}-{:02d}-01'.format(config['starty'],months[0]),'{}-{:02d}-01'.format(int(config['endy'])+1,months[-1]))).rename({'latitude':'lat','longitude':'lon'})
if aggr=='1m':
    clmean_tp = era5_clim_tp.mean('time')
    valianom_tp = (era5_vali_tp - clmean_tp)/clmean_tp*100.
    valianom_tp = valianom_tp.squeeze()
elif aggr=='3m':
    # Calculate 3-month aggregations
    era5_vali_tp = era5_vali_tp.rolling(time=3).mean()
    era5_vali_tp = era5_vali_tp.where(era5_vali_tp['time.month']==months[-1], drop=True)
    era5_clim_tp = era5_clim_tp.rolling(time=3).mean()
    era5_clim_tp = era5_clim_tp.where(era5_clim_tp['time.month']==months[-1], drop=True)
    # Calculate anomalies
    clmean_tp = era5_clim_tp.mean('time')
    valianom_tp = (era5_vali_tp - clmean_tp)/clmean_tp*100.
    valianom_tp = valianom_tp.squeeze()
elif aggr=='5m':
    # Calculate 3-month aggregations
    era5_vali_tp = era5_vali_tp.rolling(time=5).mean()
    era5_vali_tp = era5_vali_tp.where(era5_vali_tp['time.month']==months[-1], drop=True)
    era5_clim_tp = era5_clim_tp.rolling(time=5).mean()
    era5_clim_tp = era5_clim_tp.where(era5_clim_tp['time.month']==months[-1], drop=True)
    # Calculate anomalies
    clmean_tp = era5_clim_tp.mean('time')
    valianom_tp = (era5_vali_tp - clmean_tp)/clmean_tp*100.
    valianom_tp = valianom_tp.squeeze()

# %% [markdown]
# ## 2.2 Percentiles

# We calculate the percentiles for the ERA5 data.

#%%
print("2.2 Percentiles")

era5_vali_t2m = era5_vali_t2m.compute().squeeze()
era5_vali_tp = era5_vali_tp.compute().squeeze()
era5_clim_t2m = era5_clim_t2m.compute().squeeze()
era5_clim_tp = era5_clim_tp.compute().squeeze()

# Empty arrays
percentiles_t2m = xr.full_like(era5_vali_t2m,fill_value=0.)
percentiles_tp = xr.full_like(era5_vali_tp,fill_value=0.)

t = 0
# For each latitude and longitude
for ilat in era5_vali_t2m.lat.values:
   for ilon in era5_vali_t2m.lon.values:
        # Observed values for hindcast period
        serie_t2m = era5_clim_t2m.sel(lat=ilat,lon=ilon).values
        serie_tp = era5_clim_tp.sel(lat=ilat,lon=ilon).values
        # Observed values for forecast period
        score_t2m = era5_vali_t2m.sel(lat=ilat,lon=ilon).values
        score_tp = era5_vali_tp.sel(lat=ilat,lon=ilon).values
        # Percentile 
        perce_t2m = stats.percentileofscore(serie_t2m, score_t2m)
        perce_tp = stats.percentileofscore(serie_tp, score_tp)
        # Filling the array
        percentiles_t2m.loc[dict(lon=ilon, lat=ilat)] = perce_t2m
        percentiles_tp.loc[dict(lon=ilon, lat=ilat)] = perce_tp
        t+=1
        print(f'{t/(era5_vali_t2m.lat.size*era5_vali_t2m.lon.size)*100.:.2f}%')
# %% [markdown]
# ## 2.3 Plots

# We represent the anomaly and percentiles.

#%%
print("2.3 Plots")

# Some predefined options to plot each variable
var_options = {'t2m': [np.linspace(-3.,3.,13), plt.colormaps['RdYlBu_r'], r'2m temperature anomaly ($^\circ C$)',np.linspace(5.,20.,16), r'2m temperature'],
               'tp': [np.linspace(-50.,50.,21), plt.colormaps['BrBG'], r'total precipitation relative anomaly (%)',np.linspace(0.,1500.,16), r'total precipitation']
              }

# Region definition
if region=='Iberia':
    box_limits = [-30, 5, 25, 50] # [West, East, South, North]
elif region=='MedCOF':
    box_limits = [-30, 50, 14, 55] # [West, East, South, North]

# Create a figure 
fig = plt.figure(figsize=(18,10))
# Subdivide the figure (rows x columns)
gs = fig.add_gridspec(2,2)

# A map with projection ccrs.PlateCarree()
ax1 = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
# We define the map extension
ax1.set_extent(box_limits, crs=ccrs.PlateCarree())
# Borders and coastlines
ax1.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
ax1.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
# Lat Lon gridlines and labels (but not right and top labels)
gl = ax1.gridlines(draw_labels=True)
gl.right_labels = False
gl.top_labels = False
# Filled contour 
cs1 = ax1.contourf(valianom_t2m.lon,valianom_t2m.lat,valianom_t2m,levels=var_options['t2m'][0],cmap=var_options['t2m'][1],extend='both')
# Colorbar position
cax1 = ax1.inset_axes([1.05, 0., 0.05, 1.])
cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
# Include title
plt.title('ERA5 '+var_options['t2m'][2],loc='center',fontsize=12)      

# A map with projection ccrs.PlateCarree()
ax2 = fig.add_subplot(gs[1,0],projection=ccrs.PlateCarree())
# We define the map extension
ax2.set_extent(box_limits, crs=ccrs.PlateCarree())
# Borders and coastlines
ax2.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
ax2.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
# Lat Lon gridlines and labels (but not right and top labels)
gl = ax2.gridlines(draw_labels=True)
gl.right_labels = False
gl.top_labels = False
# Filled contour 
cs2 = ax2.contourf(valianom_tp.lon,valianom_tp.lat,valianom_tp,levels=var_options['tp'][0],cmap=var_options['tp'][1],extend='both')
# Colorbar position
cax2 = ax2.inset_axes([1.05, 0., 0.05, 1.])
cb2 = plt.colorbar(cs2, cax=cax2, orientation='vertical')
# Include title
plt.title('ERA5 '+var_options['tp'][2],loc='center',fontsize=12)      

# A map with projection ccrs.PlateCarree()
ax3 = fig.add_subplot(gs[0,1],projection=ccrs.PlateCarree())
# We define the map extension
ax3.set_extent(box_limits, crs=ccrs.PlateCarree())
# Borders and coastlines
ax3.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
ax3.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
# Lat Lon gridlines and labels (but not right and top labels)
gl = ax3.gridlines(draw_labels=True)
gl.right_labels = False
gl.top_labels = False
# Filled contour
cs3 = ax3.contourf(percentiles_t2m.lon,percentiles_t2m.lat,percentiles_t2m,levels=np.linspace(0.,100.,11),cmap=plt.colormaps['RdBu_r'])
# Colorbar position
cax3 = ax3.inset_axes([1.05, 0., 0.05, 1.])
cb3 = plt.colorbar(cs3, cax=cax3, orientation='vertical')
cb3.set_ticks([i for i in cb3.get_ticks()])
cb3.ax.set_yticklabels(["{}th".format(int(i)) for i in cb3.get_ticks()])
# Include title
plt.title(var_options['t2m'][4]+' corresponding percentile',loc='center',fontsize=12)      

# A map with projection ccrs.PlateCarree()
ax4 = fig.add_subplot(gs[1,1],projection=ccrs.PlateCarree())
# We define the map extension
ax4.set_extent(box_limits, crs=ccrs.PlateCarree())
# Borders and coastlines
ax4.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
ax4.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
# Lat Lon gridlines and labels (but not right and top labels)
gl = ax4.gridlines(draw_labels=True)
gl.right_labels = False
gl.top_labels = False
# Filled contour
cs4 = ax4.contourf(percentiles_tp.lon,percentiles_tp.lat,percentiles_tp,levels=np.linspace(0.,100.,11),cmap=plt.colormaps['BrBG'])
# Colorbar position
cax4 = ax4.inset_axes([1.05, 0., 0.05, 1.])
cb4 = plt.colorbar(cs4, cax=cax4, orientation='vertical')
cb4.set_ticks([i for i in cb4.get_ticks()])
cb4.ax.set_yticklabels(["{}th".format(int(i)) for i in cb4.get_ticks()])
# Include title
plt.title(var_options['tp'][4]+' corresponding percentile',loc='center',fontsize=12)      

# Include figure title
fig.suptitle('ERA5 {}-{}\n Reference period: {}-{}'.format(endyear,meses,config['starty'],config['endy']), fontsize=18)
# Save figure
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.07)
fig.savefig(PLOTSDIR+'/ERA5_{}-{}_ref{}-{}_{}.png'.format(endyear,meses,config['starty'],config['endy'],region))  
