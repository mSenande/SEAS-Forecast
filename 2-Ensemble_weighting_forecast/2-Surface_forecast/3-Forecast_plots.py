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
    institution = str(sys.argv[1]).replace('"', '')
    name = str(sys.argv[2]).replace('"', '')
    year = int(sys.argv[3])
    startmonth = int(sys.argv[4])
    aggr = str(sys.argv[5])
    fcmonth = int(sys.argv[6])
    region = str(sys.argv[7])
    var = str(sys.argv[8])
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

    # Region selected for plots
    region = input("Selecciona qué región representar [ Iberia , MedCOF ]: ")

    # Variable
    var = input("Usar la siguiente variable [ t2m , tprate ]: ")

# Save the simplier model and system name
model = str(models[(models["institution"]==institution) & (models["name"]==name)]["short_institution"].values[0])
system = str(models[(models["institution"]==institution) & (models["name"]==name)]["short_name"].values[0])

# Here we save the configuration
config = dict(
    list_vars = ['2m_temperature', 'total_precipitation'],
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
HINDDIR = os.getenv('HIND_DIR') # Directory where hindcast surface data files are located
FOREDIR = os.getenv('FORE_DIR') # Directory where forecast surface data files are located
NEW_FOREDIR = os.getenv('NEW_FORE_DIR') # Directory where forecast outputs will be located
POSTDIR_VER = os.getenv('POST_DIR_VER')
NEW_SCORES_HINDDIR = POSTDIR_VER + '/scores' # Directory where skill scores files are located
MODES_HIND_DIR = os.getenv('MODES_HIND_DIR')  # Directory where hindcast variability patterns files are located
MODES_FORE_DIR = os.getenv('MODES_FORE_DIR' ) # Directory where forecast variability patterns files are located
MODES_FORECASTED_DIR = os.getenv('MODES_FORECASTED_DIR') # Directory where forecasted variability patterns are ocated

# Base name for hindcast
hcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)
# File name for hindcast
hcst_fname = f'{HINDDIR}/{hcst_bname}.grib'
# Base name for forecast
fcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_forecast{fcy}_monthly'.format(**config)
# File name for forecast
fcst_fname = f'{FOREDIR}/{fcst_bname}.grib'

# Check if files exist
if not os.path.exists(fcst_fname):
    print('No se descargaron aún los datos de forecast de este modelo y sistema')
    sys.exit()
elif not os.path.exists(hcst_fname):
    print('No se descargaron aún los datos de hindcast de este modelo y sistema')
    sys.exit()

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

# %% [markdown]
# ## 3.1 Read hindcast

# We calculate the monthly and 3-months anomalies for the hindcast data.

#%%
print("3.1 Read hindcast")

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

# %% [markdown]
# ## 3.2 Read forecast

# We calculate the monthly and 3-months anomalies for the hindcast data.

#%%
print("3.2 Read forecast")

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


# %% [markdown]
# ## 3.3 Ensemble weighting

# We chose the best variability pattern forecast (among different models) for this valid month / season.
#
# Then we weight the ensemble members of the model according to the distance of their variability pattern values with the variability pattern forecast
#
# We would have four weighting functions (one for each variability pattern considered), 
# so we finally need to compute a weighted average of these four weighting functions with the explained variance of each pattern as the weights

#%%
print("3.3 Ensemble weighting")

# Read forecast PCs
fpcs_fname = f'{MODES_FORE_DIR}/{fcst_bname}.{aggr}.PCs.nc'
# Reading data from file
fpcs = xr.open_dataarray(fpcs_fname).sel(forecastMonth=fcmonth)

# Read forecasted PCs
forecasted_eof1 = pd.read_csv(f'{MODES_FORECASTED_DIR}/{year_end}-{endmonth:02d}_{meses}/Forecasted_EOF1.csv',index_col=0)
forecasted_eof2 = pd.read_csv(f'{MODES_FORECASTED_DIR}/{year_end}-{endmonth:02d}_{meses}/Forecasted_EOF2.csv',index_col=0)
forecasted_eof3 = pd.read_csv(f'{MODES_FORECASTED_DIR}/{year_end}-{endmonth:02d}_{meses}/Forecasted_EOF3.csv',index_col=0)
forecasted_eof4 = pd.read_csv(f'{MODES_FORECASTED_DIR}/{year_end}-{endmonth:02d}_{meses}/Forecasted_EOF4.csv',index_col=0)

# Select the value of the model with greater skill
forecasted_eof1 = forecasted_eof1.sort_values('RPSS',ascending=False).reset_index(drop=True)
forecasted_eof1_value = forecasted_eof1.loc[0]['PC_value']
forecasted_eof2 = forecasted_eof2.sort_values('RPSS',ascending=False).reset_index(drop=True)
forecasted_eof2_value = forecasted_eof2.loc[0]['PC_value']
forecasted_eof3 = forecasted_eof3.sort_values('RPSS',ascending=False).reset_index(drop=True)
forecasted_eof3_value = forecasted_eof3.loc[0]['PC_value']
forecasted_eof4 = forecasted_eof4.sort_values('RPSS',ascending=False).reset_index(drop=True)
forecasted_eof4_value = forecasted_eof4.loc[0]['PC_value']

if not len(sys.argv) > 2:
    forecasted_eof1_newvalue = input("Cambiar manualmente valor para PC de EOF1: ")
    if not len(forecasted_eof1_newvalue)==0:
        forecasted_eof1_value=float(forecasted_eof1_newvalue)
    forecasted_eof2_newvalue = input("Cambiar manualmente valor para PC de EOF2: ")
    if not len(forecasted_eof2_newvalue)==0:
        forecasted_eof2_value=float(forecasted_eof2_newvalue)
    forecasted_eof3_newvalue = input("Cambiar manualmente valor para PC de EOF3: ")
    if not len(forecasted_eof3_newvalue)==0:
        forecasted_eof3_value=float(forecasted_eof3_newvalue)
    forecasted_eof4_newvalue = input("Cambiar manualmente valor para PC de EOF4: ")
    if not len(forecasted_eof4_newvalue)==0:
        forecasted_eof4_value=float(forecasted_eof4_newvalue)

# Compute anomaly
pcs_anom = fpcs.copy()
pcs_anom = xr.where(pcs_anom.mode==0,(pcs_anom-forecasted_eof1_value)**2., pcs_anom)
pcs_anom = xr.where(pcs_anom.mode==1,(pcs_anom-forecasted_eof2_value)**2., pcs_anom)
pcs_anom = xr.where(pcs_anom.mode==2,(pcs_anom-forecasted_eof3_value)**2., pcs_anom)
pcs_anom = xr.where(pcs_anom.mode==3,(pcs_anom-forecasted_eof4_value)**2., pcs_anom)

# Variances
# Explained variance percentages are stored in csv files
if aggr=='1m':
    variances = pd.read_csv(f'{MODES_HIND_DIR}/ERA5_VAR_{endmonth:02d}.csv',index_col=0)
else:
    variances = pd.read_csv(f'{MODES_HIND_DIR}/ERA5_VAR_{meses}.csv',index_col=0)

# We can delete any EOF from analysis
if not len(sys.argv) > 2:
    delete_EOF = input("Eliminar algún patrón de variabilidad del pesado de miembros: [ EOF1 , EOF2 , EOF3 , EOF4 ]: ")
    if not len(delete_EOF)==0:
        variances['VAR'][variances['EOF']==int(delete_EOF[-1])]=0.
        delete_EOF = input("Eliminar otro patrón de variabilidad del pesado de miembros: [ EOF1 , EOF2 , EOF3 , EOF4 ]: ")
        if not len(delete_EOF)==0:
            variances['VAR'][variances['EOF']==int(delete_EOF[-1])]=0.
            delete_EOF = input("Eliminar otro patrón de variabilidad del pesado de miembros: [ EOF1 , EOF2 , EOF3 , EOF4 ]: ")
            if not len(delete_EOF)==0:
                variances['VAR'][variances['EOF']==int(delete_EOF[-1])]=0.

# We create an array with all the percentages of explained variance associated with each variability pattern
eof_variances = xr.DataArray(np.zeros([4]),coords={'mode':[0,1,2,3]})
eof_variances.values=variances['VAR']

# We calculate the weighting funtions
weights = 1./(1.+pcs_anom.weighted(eof_variances).mean(dim='mode'))
# We normalize the weighting funtions
weights_norm = (weights/weights.sum(dim='number'))
# We standarize the weighting funtions
weights_off = (weights-weights.min(dim='number'))/(weights.max(dim='number')-weights.min(dim='number'))
weights_scal = weights_off/weights_off.sum(dim='number')
# Control for non-negative values
weights_norm = xr.where(weights_norm<0.,0.,weights_norm)
weights_scal = xr.where(weights_scal<0.,0.,weights_scal)
# Control for non-nan values
weights_norm = weights_norm.fillna(1./weights_norm.number.size)
weights_scal = weights_scal.fillna(1./weights_scal.number.size)

# %% [markdown]
# ## 3.4 Anomalies

# Compute the anomaly of the forecast from the hindcast.

#%%
print("3.4 Anomalies")

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
anom_mean = anom.weighted(weights_scal).mean(dim='number').squeeze()
anom_mean.to_netcdf(f'{NEW_FOREDIR}/{fcst_bname}-{meses}_{var}-anomaly.nc')

# %% [markdown]
# ## 3.5 Probabilities

# Compute the tercile probabilities of the forecast.

#%%
print("3.5 Probabilities")

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

# Calculate probabilities for tercile categories by counting members within each category
quantiles = [1/3., 2/3.]
numcategories = len(quantiles)+1
l_probs_forecast=list()
# One's array
f_ones = xr.full_like(fcst[var], 1)
# For each quantile
for icat in range(numcategories):
    # Get the lower and higher threshold
    h_lo,h_hi = get_thresh(icat, quantiles, hcst[var])
    # Count the number of member between the threshold
    probf = f_ones.where((fcst[var]>h_lo) & (fcst[var]<=h_hi)).weighted(weights_scal).sum('number')#.sum('number')/float(h.number.size)
    # Instead of using the coordinate 'quantile' coming from the hindcast xr.Dataset
    # we will create a new coordinate called 'category'
    if 'quantile' in list(probf.coords):
        probf = probf.drop('quantile')
    l_probs_forecast.append(probf.assign_coords({'category':icat}))
# Concatenating tercile probs categories
probs_forecast = xr.concat(l_probs_forecast,dim='category').squeeze().compute()                   
# Selecting the tercile with maximum probability                 
probs_forecast_max = probs_forecast.where(probs_forecast==probs_forecast.max(dim='category'))
# Save to netcdf file                 
probs_forecast.to_netcdf(f'{NEW_FOREDIR}/{fcst_bname}-{meses}_{var}-probability.nc')

# %% [markdown]
# ## 3.6 Plots

# We represent the anomaly and forecast probabilities with some verification scores (Correlation, RPSS and ROC area) of the hindcast.

#%%
print("3.6 Plots")

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
corr = xr.open_dataset(f'{NEW_SCORES_HINDDIR}/{hcst_bname}.{aggr}.corr.nc')
corr_pval = xr.open_dataset(f'{NEW_SCORES_HINDDIR}/{hcst_bname}.{aggr}.corr_pval.nc')
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
rpss = xr.open_dataset(f'{NEW_SCORES_HINDDIR}/{hcst_bname}.{aggr}.rpss.nc')
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
roc = xr.open_dataset(f'{NEW_SCORES_HINDDIR}/{hcst_bname}.{aggr}.roc.nc')
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
fig.suptitle(f'{institution} {name}\n Forecast start: {year}-{startmonth:02d}, Valid time: {meses}\n Variable: {var_options[var][4]}', fontsize=18)
# Save figure
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.07)
fig.savefig(PLOTSDIR+f'/{fcst_bname}-{meses}_{var}_{region}.png')  
