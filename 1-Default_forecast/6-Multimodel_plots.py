# %% [markdown]
# # 6. Multimodel forecast plots

# This script creates different plots for the multimodel forecast.
# 
# First we have to decide a start year and month, a month aggregation, a region and a variable. 

#%%
print("6. Multimodel forecast plots")

import os
import sys
from dotenv import load_dotenv
import pandas as pd
import xarray as xr
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

# Read the file with model names
models = pd.read_csv('./models_running.csv')

# If variables are introduced from the command line, read them
if len(sys.argv) > 2:
    year = int(sys.argv[1])
    startmonth = int(sys.argv[2])
    aggr = str(sys.argv[3])
    fcmonth = int(sys.argv[4])
    region = str(sys.argv[5])
    var = str(sys.argv[6])
    if aggr=='1m':
        # Valid season/month number
        endmonth = (startmonth+fcmonth)-1 if (startmonth+fcmonth)<=12 else (startmonth+fcmonth)-1-12
        # Valid season/month name
        meses = np.array(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])[endmonth-1]
    elif aggr=='3m':
        # Valid season/month number
        endmonth = (startmonth+fcmonth)-1 if (startmonth+fcmonth)<=12 else (startmonth+fcmonth)-1-12
        # Valid season/month name
        meses = np.array(['NDJ', 'DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND'])[endmonth-1]
    elif aggr=='5m':
        # Valid season/month number
        endmonth = (startmonth+fcmonth)-1 if (startmonth+fcmonth)<=12 else (startmonth+fcmonth)-1-12
        # Valid season/month name
        meses = np.array(['SONDJ' , 'ONDJF' , 'NDJFM' , 'DJFMA' , 'JFMAM' , 'FMAMJ' , 'MAMJJ' , 'AMJJA' , 'MJJAS' , 'JJASO' , 'JASON' , 'ASOND' ])[endmonth-1]

# If no variables were introduced, ask for them
else:
    # Which year of initialization
    year = int(input("Año de inicio del forecast: "))

    # Which start month
    startmonth = int(input("Mes de inicialización (en número): "))

    # Montly aggregation
    aggr = input("Selecciona el tipo de agregación mensual [ 1m , 3m , 5m ]: ")

    if aggr=='1m':
        # Valid season/month name
        meses = input("Resultados para el mes [ Jan , Feb , Mar , Apr , May , Jun , Jul , Aug , Sep , Oct , Nov , Dec ]: ")
        # Valid season/month number
        endmonth = np.where(np.array(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']) == meses)[0][0]+1
        # Forecast month number
        fcmonth = (endmonth-startmonth)+1 if (endmonth-startmonth)>=0 else (endmonth-startmonth)+13
    elif aggr=='3m':
        # Valid season/month name
        meses = input("Resultados para el trimestre [ NDJ , DJF , JFM , FMA , MAM , AMJ , MJJ , JJA , JAS , ASO , SON , OND ]: ")
        # Valid season/month number
        endmonth = np.where(np.array(['NDJ', 'DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']) == meses)[0][0]+1
        # Forecast month number
        fcmonth = (endmonth-startmonth)+1 if (endmonth-startmonth)>=0 else (endmonth-startmonth)+13
    elif aggr=='5m':
        # Valid season/month name
        meses = input("Resultados para los meses [ SONDJ , ONDJF , NDJFM , DJFMA , JFMAM , FMAMJ , MAMJJ , AMJJA , MJJAS , JJASO , JASON , ASOND ]: ")
        # Valid season/month number
        endmonth = np.where(np.array(['SONDJ','ONDJF','NDJFM','DJFMA','JFMAM','FMAMJ','MAMJJ','AMJJA','MJJAS','JJASO','JASON','ASOND']) == meses)[0][0]+1
        # Forecast month number
        fcmonth = (endmonth-startmonth)+1 if (endmonth-startmonth)>=0 else (endmonth-startmonth)+13

    # Region selected for plots
    region = input("Selecciona qué región representar [ Iberia , MedCOF ]: ")

    # Variable
    var = input("Usar la siguiente variable [ t2m , tprate ]: ")

# Here we save the configuration
config = dict(
    list_vars = ['2m_temperature', 'total_precipitation'],
    fcy = year,
    hcstarty = 1993,
    hcendy = 2016,
    start_month = startmonth,
)

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
HINDDIR = os.getenv('HIND_DIR')
FOREDIR = os.getenv('FORE_DIR')
SCOREDIR = HINDDIR + '/scores'

# If the end month is not in the same year than the start month, we create a new variable with the valid year number
if endmonth<startmonth:
    year_end = str(int(year)+1)
else:
    year_end = year
# We select a directory to save the results
PLOTSDIR = f'./PLOTS/{year_end}-{endmonth:02d}_{meses}'
# Check if directory exists
if not os.path.exists(PLOTSDIR):
    # If it doesn't exist, create it
    try:
        os.system(f'mkdir -p {PLOTSDIR}')
    except FileExistsError:
        pass

# Base name for multimodel forecast
multi_fcst_bname = 'multimodel_stmonth{start_month:02d}_forecast{fcy}_monthly'.format(**config)
# Base name for multimodel hindcast
multi_hcst_bname = 'multimodel_stmonth{start_month:02d}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)

# %% [markdown]
# ## 6.1 Compute anomalies and probabilities

# We calculate the monthly and 3-months anomalies and probabilities for the hindcast and forecast data.

#%%
print("6.1 Compute anomalies and probabilities")

# We define a function to calculate the boundaries of forecast categories defined by quantiles
def get_thresh(icat,quantiles,xrds,dims=['number','start_date']):

    if icat == 0:
        xrds_lo = -np.inf
        xrds_hi = xrds.quantile(quantiles[icat],dim=dims,skipna=True)      
        
    elif icat == len(quantiles):
        xrds_lo = xrds.quantile(quantiles[icat-1],dim=dims,skipna=True)
        xrds_hi = np.inf
        
    else:
        xrds_lo = xrds.quantile(quantiles[icat-1],dim=dims,skipna=True)
        xrds_hi = xrds.quantile(quantiles[icat],dim=dims,skipna=True)
      
    return xrds_lo,xrds_hi

l_probs_models=list()
l_anoms_models=list()
# For each model 
for m in models.index:

    model = models['short_institution'][m]
    system = models['short_name'][m]

    config['origin'] = model
    config['system'] = system
    config['isLagged'] = False if model in ['ecmwf', 'meteo_france', 'dwd', 'cmcc', 'eccc'] else True

    # Base name for hindcast
    hcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)
    # File name for hindcast
    hcst_fname = f'{HINDDIR}/{hcst_bname}.grib'

    if not os.path.exists(hcst_fname):
        print('No se descargaron aún los datos de hindcast de este modelo y sistema')
        sys.exit()

    # For the re-shaping of time coordinates in xarray.Dataset we need to select the right one 
    #  -> burst mode ensembles (e.g. ECMWF SEAS5) use "time". This is the default option in this notebook
    #  -> lagged start ensembles (e.g. MetOffice GloSea6) use "indexing_time" (see CDS documentation about nominal start date)
    st_dim_name = 'time' if not config.get('isLagged',False) else 'indexing_time'

    # Reading hindcast data from file
    hcst = xr.open_dataset(hcst_fname,engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', st_dim_name)))
    # We use dask.array with chunks on leadtime, latitude and longitude coordinate
    hcst = hcst.chunk({'forecastMonth':1, 'latitude':'auto', 'longitude':'auto'})
    # Reanme coordinates to match those of observations
    hcst = hcst.rename({'latitude':'lat','longitude':'lon', st_dim_name:'start_date'})

    # Add start_month to the xr.Dataset
    start_month = pd.to_datetime(hcst.start_date.values[0]).month
    hcst = hcst.assign_coords({'start_month':start_month})
    # Add valid_time to the xr.Dataset
    vt = xr.DataArray(dims=('start_date','forecastMonth'), coords={'forecastMonth':hcst.forecastMonth,'start_date':hcst.start_date})
    vt.data = [[pd.to_datetime(std)+relativedelta(months=fcmonth-1) for fcmonth in vt.forecastMonth.values] for std in vt.start_date.values]
    hcst = hcst.assign_coords(valid_time=vt)

    if aggr=='3m':
        # Calculate 3-month aggregations
        hcst = hcst.rolling(forecastMonth=3).mean()
        # rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped
        hcst = hcst.where(hcst.forecastMonth>=3,drop=True)
    elif aggr=='5m':
        # Calculate 5-month aggregations
        hcst = hcst.rolling(forecastMonth=5).mean()
        # rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped
        hcst = hcst.where(hcst.forecastMonth>=5,drop=True)

    # Base name for forecast
    fcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_forecast{fcy}_monthly'.format(**config)
    # File name for forecast
    fcst_fname = f'{FOREDIR}/{fcst_bname}.grib'

    # Check if files exist
    if not os.path.exists(fcst_fname):
        print('No se descargaron aún los datos de hindcast de este modelo y sistema')
        sys.exit()

    # For the re-shaping of time coordinates in xarray.Dataset we need to select the right one 
    #  -> burst mode ensembles (e.g. ECMWF SEAS5) use "time". This is the default option in this notebook
    #  -> lagged start ensembles (e.g. MetOffice GloSea6) use "indexing_time" (see CDS documentation about nominal start date)
    st_dim_name = 'time' if not config.get('isLagged',False) else 'indexing_time'

    # Reading hindcast data from file
    fcst = xr.open_dataset(fcst_fname,engine='cfgrib', backend_kwargs=dict(time_dims=('forecastMonth', st_dim_name)))
    # We use dask.array with chunks on leadtime, latitude and longitude coordinate
    fcst = fcst.chunk({'forecastMonth':1, 'latitude':'auto', 'longitude':'auto'})
    # Reanme coordinates to match those of observations
    fcst = fcst.rename({'latitude':'lat','longitude':'lon', st_dim_name:'start_date'})

    # Add start_month to the xr.Dataset
    start_month = pd.to_datetime(fcst.start_date.values).month
    fcst = fcst.assign_coords({'start_month':start_month})
    # Add valid_time to the xr.Dataset
    vt = xr.DataArray(dims=('forecastMonth',), coords={'forecastMonth': fcst.forecastMonth})
    vt.data = [pd.to_datetime(fcst.start_date.values)+relativedelta(months=fcmonth-1) for fcmonth in fcst.forecastMonth.values]
    fcst = fcst.assign_coords(valid_time=vt)

    if aggr=='3m':
        # Calculate 3-month aggregations
        fcst = fcst.rolling(forecastMonth=3).mean()
        # rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped
        fcst = fcst.where(hcst.forecastMonth>=3,drop=True)
    elif aggr=='5m':
        # Calculate 5-month aggregations
        fcst = fcst.rolling(forecastMonth=5).mean()
        # rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped
        fcst = fcst.where(hcst.forecastMonth>=5,drop=True)

    # We select the forecast month
    hcst = hcst.sel(forecastMonth=fcmonth).drop('forecastMonth')
    fcst = fcst.sel(forecastMonth=fcmonth).drop('forecastMonth')

    # Compute the hindcast average
    hindcast_mean = hcst.mean(['start_date'])

    # Compute the forecast anomaly from the hindcast mean
    if var=='t2m':
        anom = fcst[var] - hindcast_mean[var]
    elif var=='tprate':
        # For precipitation, we use the relative anomaly (expressed as a precentage)
        anom = (fcst[var] - hindcast_mean[var])/hindcast_mean[var]*100.
    
    # Compute the ensemble mean (for each spatial point)
    anom_mean = anom.mean(dim='number').squeeze()

    # We will create a new coordinate called 'model' and concatenae the results
    l_anoms_models.append(anom_mean.assign_coords({'model':model}))

    # Calculate probabilities for tercile categories by counting members within each category
    quantiles = [1/3., 2/3.]
    numcategories = len(quantiles)+1
    l_probs_forecast=list()
    # For each quantile
    for icat in range(numcategories):
        # Get the lower and higher threshold
        h_lo,h_hi = get_thresh(icat, quantiles, hcst[var])
        # Count the number of member between the threshold
        probf = np.logical_and(fcst[var]>h_lo, fcst[var]<=h_hi).sum('number')/float(fcst.number.size)
        # Instead of using the coordinate 'quantile' coming from the hindcast xr.Dataset
        # we will create a new coordinate called 'category'
        if 'quantile' in list(probf.coords):
            probf = probf.drop('quantile')
        l_probs_forecast.append(probf.assign_coords({'category':icat}))
    # Concatenating tercile probs categories
    probs_forecast = xr.concat(l_probs_forecast,dim='category').squeeze().compute()
    
    # We will create a new coordinate called 'model' and concatenae the results
    l_probs_models.append(probs_forecast.assign_coords({'model':model}))


# We average the results of all models
anom_mean = xr.concat(l_anoms_models,dim='model').mean(dim='model').compute()
probs_forecast = xr.concat(l_probs_models,dim='model').mean(dim='model').compute()
# Assign attribute indicating models
anom_mean = anom_mean.assign_attrs({'Models': " | ".join(models['institution']+'-'+models['name'])})
probs_forecast = probs_forecast.assign_attrs({'Models': " | ".join(models['institution']+'-'+models['name'])})
# Save to netcdf file 
anom_mean.to_netcdf(f'{FOREDIR}/{multi_fcst_bname}-{meses}_{var}-anomaly.nc')               
probs_forecast.to_netcdf(f'{FOREDIR}/{multi_fcst_bname}-{meses}_{var}-probability.nc')
# Selecting the tercile with maximum probability                 
probs_forecast_max = probs_forecast.where(probs_forecast==probs_forecast.max(dim='category'))

# %% [markdown]
# ## 6.2 Plots

# We represent the anomaly and forecast probabilities with some verification scores (Correlation, RPSS and ROC area) of the hindcast.

#%%
print("6.2 Plots")

# Some predefined options to plot each variable
var_options = {'t2m': [np.linspace(-3.,3.,13), plt.colormaps['RdYlBu_r'], r'2m temperature anomaly ($^\circ C$)',np.linspace(5.,20.,16), r'2m temperature'],
               'tprate': [np.linspace(-50.,50.,21), plt.colormaps['BrBG'], r'total precipitation relative anomaly (%)',np.linspace(0.,1500.,16), r'total precipitation']
              }
# Some predefined options to plot each tercile
if var=='t2m':
    ter_options = {'lower': [np.linspace(0.4,0.9,6), plt.colormaps['Blues']],
                'middle': [np.linspace(0.4,0.9,6), plt.colormaps['Greys']],
                'upper': [np.linspace(0.4,0.9,6), plt.colormaps['Reds']]
                }
elif var=='tprate':
    ter_options = {'lower': [np.linspace(0.4,0.9,6), plt.colormaps['Oranges']],
                'middle': [np.linspace(0.4,0.9,6), plt.colormaps['Greys']],
                'upper': [np.linspace(0.4,0.9,6), plt.colormaps['Greens']]
                }

# Region definition
if region=='Iberia':
    box_limits = [-30, 5, 25, 50] # [West, East, South, North]
elif region=='MedCOF':
    box_limits = [-30, 50, 14, 55] # [West, East, South, North]

# Create a figure 
fig = plt.figure(figsize=(15,16))
# Subdivide the figure (rows x columns)
gs = fig.add_gridspec(4,2)

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
cs1 = ax1.contourf(hindcast_mean.lon,hindcast_mean.lat,anom_mean,levels=var_options[var][0],cmap=var_options[var][1],extend='both')
# Colorbar position
cax1 = ax1.inset_axes([1.05, 0., 0.05, 1.])
cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
# Include title
plt.title('Forecasted '+var_options[var][2],loc='center',fontsize=12)      

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
# Filled contour of each tercile probabilities
cs1 = ax2.contourf(hindcast_mean.lon,hindcast_mean.lat,probs_forecast_max.sel(category=0),levels=ter_options['lower'][0],cmap=ter_options['lower'][1],extend='max')
#cs2 = ax2.contourf(hindcast_mean.lon,hindcast_mean.lat,probs_forecast_max.sel(category=1),levels=ter_options['middle'][0],cmap=ter_options['middle'][1],extend='max')
cs3 = ax2.contourf(hindcast_mean.lon,hindcast_mean.lat,probs_forecast_max.sel(category=2),levels=ter_options['upper'][0],cmap=ter_options['upper'][1],extend='max')
# Colorbar position 1
cax1 = ax2.inset_axes([1.05, 0., 0.05, 1.])
cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
cb1.set_ticks([])
cb1.ax.set_title('lower', loc='left', rotation=45)
# Colorbar position 2
# cax2 = ax2.inset_axes([1.1, 0., 0.05, 1.])
# cb2 = plt.colorbar(cs2, cax=cax2, orientation='vertical')
# cb2.set_ticks([])
# cb2.ax.set_title('middle', loc='left', rotation=45)
# Colorbar position 3
cax3 = ax2.inset_axes([1.10, 0., 0.05, 1.])
cb3 = plt.colorbar(cs3, cax=cax3, orientation='vertical')
cb3.set_ticks([i for i in cb3.get_ticks()])
cb3.ax.set_yticklabels(["{:.1%}".format(i) for i in cb3.get_ticks()])
cb3.ax.set_title('upper', loc='left', rotation=45)
# Include title
plt.title('Forecasted tercile probabilities',loc='center',fontsize=12)      

# A map with projection ccrs.PlateCarree()
ax3 = fig.add_subplot(gs[2,0],projection=ccrs.PlateCarree())
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
cs1 = ax3.contourf(hindcast_mean.lon,hindcast_mean.lat,probs_forecast.sel(category=0),levels=np.linspace(0.,0.9,10),cmap=plt.colormaps['viridis'],extend='max')
# Colorbar position
cax1 = ax3.inset_axes([1.05, 0., 0.05, 1.])
cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
cb1.set_ticks([i for i in cb1.get_ticks()])
cb1.ax.set_yticklabels(["{:.1%}".format(i) for i in cb1.get_ticks()])
cb1.ax.set_title('lower', loc='left', rotation=45)
# Include title
plt.title('Lower tercile forecasted probabilities',loc='center',fontsize=12)      

# A map with projection ccrs.PlateCarree()
ax4 = fig.add_subplot(gs[3,0],projection=ccrs.PlateCarree())
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
cs1 = ax4.contourf(hindcast_mean.lon,hindcast_mean.lat,probs_forecast.sel(category=2),levels=np.linspace(0.,0.9,10),cmap=plt.colormaps['viridis'],extend='max')
# Colorbar position
cax1 = ax4.inset_axes([1.05, 0., 0.05, 1.])
cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
cb1.set_ticks([i for i in cb1.get_ticks()])
cb1.ax.set_yticklabels(["{:.1%}".format(i) for i in cb1.get_ticks()])
cb1.ax.set_title('upper', loc='left', rotation=45)
# Include title
plt.title('Upper tercile forecasted probabilities',loc='center',fontsize=12)      

# Some predefined options to plot each verification score
score_options = {'bs': [np.linspace(0.,0.5,11), plt.colormaps['YlGn'], 3, 'max', 'Brier Score (BS)'],
                 'corr': [np.linspace(-1.,1.,11), plt.colormaps['RdYlBu_r'], 1, 'both', 'Spearmans Rank Correlation (stippling where significance below 95%)'],
                 'roc': [np.linspace(0.,1.,11), plt.colormaps['BrBG'], 3, 'both', 'Area under Relative Operating Characteristic (ROC) curve'],
                 'rocss': [np.linspace(-0.5,0.5,9), plt.colormaps['BrBG'], 3, 'both', 'Relative Operating Characteristic Skill Score (ROCSS)'],
                 'rps': [np.linspace(0.3,0.5,11), plt.colormaps['YlGn_r'], 1, 'max', 'Ranked Probability Score (RPS)'],
                 'rpss': [np.linspace(-0.5,0.5,11), plt.colormaps['BrBG'], 1, 'both', 'Ranked Probability Skill Score (RPSS)'],
                }

# Read the data file
corr = xr.open_dataset(f'{SCOREDIR}/{multi_hcst_bname}.{aggr}.corr.nc')
corr_pval = xr.open_dataset(f'{SCOREDIR}/{multi_hcst_bname}.{aggr}.corr_pval.nc')
# Select forecast month
thiscorr = corr.sel(forecastMonth=fcmonth)[var]
thiscorrpval = corr_pval.sel(forecastMonth=fcmonth)[var]

# A map with projection ccrs.PlateCarree()
ax5 = fig.add_subplot(gs[0,1],projection=ccrs.PlateCarree())
# We define the map extension
ax5.set_extent(box_limits, crs=ccrs.PlateCarree())
# Borders and coastlines
ax5.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
ax5.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
# Lat Lon gridlines and labels (but not right and top labels)
gl = ax5.gridlines(draw_labels=True)
gl.right_labels = False
gl.top_labels = False
# Filled contour
cs1 = ax5.contourf(hindcast_mean.lon,hindcast_mean.lat,thiscorr,levels=score_options['corr'][0],cmap=score_options['corr'][1],extend=score_options['corr'][3])
# Hatched contour
ax5.contourf(hindcast_mean.lon,hindcast_mean.lat,thiscorrpval,levels=[0.05,np.inf],hatches=['...',None],colors='none')
# Colorbar position
cax1 = ax5.inset_axes([1.05, 0., 0.05, 1.])
cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
# Include title
plt.title(score_options['corr'][4],loc='center',fontsize=12)      

# Read the data file
rpss = xr.open_dataset(f'{SCOREDIR}/{multi_hcst_bname}.{aggr}.rpss.nc')
# Select forecast month
thisrpss = rpss.sel(forecastMonth=fcmonth)[var]

# A map with projection ccrs.PlateCarree()
ax6 = fig.add_subplot(gs[1,1],projection=ccrs.PlateCarree())
# We define the map extension
ax6.set_extent(box_limits, crs=ccrs.PlateCarree())
# Borders and coastlines
ax6.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
ax6.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
# Lat Lon gridlines and labels (but not right and top labels)
gl = ax6.gridlines(draw_labels=True)
gl.right_labels = False
gl.top_labels = False
# Filled contour
cs1 = ax6.contourf(hindcast_mean.lon,hindcast_mean.lat,thisrpss,levels=score_options['rpss'][0],cmap=score_options['rpss'][1],extend=score_options['rpss'][3])
# Colorbar position
cax1 = ax6.inset_axes([1.05, 0., 0.05, 1.])
cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
# Include title
plt.title(score_options['rpss'][4],loc='center',fontsize=12)      

# Read the data file
roc = xr.open_dataset(f'{SCOREDIR}/{multi_hcst_bname}.{aggr}.roc.nc')
# Select forecast month
thisroc = roc.sel(forecastMonth=fcmonth)[var]

# A map with projection ccrs.PlateCarree()
ax7 = fig.add_subplot(gs[2,1],projection=ccrs.PlateCarree())
# We define the map extension
ax7.set_extent(box_limits, crs=ccrs.PlateCarree())
# Borders and coastlines
ax7.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
ax7.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
# Lat Lon gridlines and labels (but not right and top labels)
gl = ax7.gridlines(draw_labels=True)
gl.right_labels = False
gl.top_labels = False
# Filled contour
cs1 = ax7.contourf(hindcast_mean.lon,hindcast_mean.lat,thisroc.sel(category=0),levels=score_options['roc'][0],cmap=score_options['roc'][1],extend=score_options['roc'][3])
# Colorbar position
cax1 = ax7.inset_axes([1.05, 0., 0.05, 1.])
cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
# Include title
plt.title('Lower tercile '+score_options['roc'][4],loc='center',fontsize=12)      

# A map with projection ccrs.PlateCarree()
ax8 = fig.add_subplot(gs[3,1],projection=ccrs.PlateCarree())
# We define the map extension
ax8.set_extent(box_limits, crs=ccrs.PlateCarree())
# Borders and coastlines
ax8.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
ax8.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
# Lat Lon gridlines and labels (but not right and top labels)
gl = ax8.gridlines(draw_labels=True)
gl.right_labels = False
gl.top_labels = False
# Filled contour
cs1 = ax8.contourf(hindcast_mean.lon,hindcast_mean.lat,thisroc.sel(category=2),levels=score_options['roc'][0],cmap=score_options['roc'][1],extend=score_options['roc'][3])
# Colorbar position
cax1 = ax8.inset_axes([1.05, 0., 0.05, 1.])
cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
# Include title
plt.title('Upper tercile '+score_options['roc'][4],loc='center',fontsize=12)      

# Include figure title
fig.suptitle(f'Multimodel\n Forecast start: {year}-{startmonth:02d}, Valid time: {meses}\n Variable: {var_options[var][4]}', fontsize=18)
# Save figure
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.07)
fig.savefig(PLOTSDIR+f'/{multi_fcst_bname}-{meses}_{var}_{region}.png')  
