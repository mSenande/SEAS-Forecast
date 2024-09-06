# %% [markdown]
# # 4. Plot multisystem forecasts

# This script creates plots of different systems forecasts.
# 
# First we have to decide a start year and month, a month aggregation, a region and a variable. 

#%%
print("4. Plot multisystem forecasts")

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
    aggr = input("Selecciona el tipo de agregación mensual [ 1m , 3m, 5m ]: ")

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


# Some predefined options to plot each variable
var_options = {'t2m': [np.linspace(-3.,3.,13), plt.colormaps['RdYlBu_r'], r'2m temperature anomaly ($^\circ C$)',np.linspace(5.,20.,16), r'2m temperature ($^\circ C$)'],
               'tprate': [np.linspace(-50.,50.,21), plt.colormaps['BrBG'], r'total precipitation relative anomaly (%)',np.linspace(0.,1500.,16), r'total precipitation ($mm$)']
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

# Some predefined options to plot each verification score
score_options = {'bs': [np.linspace(0.,0.5,11), plt.colormaps['YlGn'], 3, 'max', 'Brier Score (BS)'],
                'corr': [np.linspace(-1.,1.,11), plt.colormaps['RdYlBu_r'], 1, 'both', 'Spearmans Rank Correlation (stippling where significance below 95%)'],
                'roc': [np.linspace(0.,1.,11), plt.colormaps['BrBG'], 3, 'both', 'Area under Relative Operating Characteristic (ROC) curve'],
                'rocss': [np.linspace(-0.5,0.5,9), plt.colormaps['BrBG'], 3, 'both', 'Relative Operating Characteristic Skill Score (ROCSS)'],
                'rps': [np.linspace(0.3,0.5,11), plt.colormaps['YlGn_r'], 1, 'max', 'Ranked Probability Score (RPS)'],
                'rpss': [np.linspace(-0.5,0.5,11), plt.colormaps['BrBG'], 1, 'both', 'Ranked Probability Skill Score (RPSS)'],
                }

# Region definition
if region=='Iberia':
    box_limits = [-30, 5, 25, 50] # [West, East, South, North]
elif region=='MedCOF':
    box_limits = [-30, 50, 14, 55] # [West, East, South, North]


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

# %% [markdown]
# ## 4.1 Plots

# We represent forecast probabilities with RPSS verification score.

#%%
print("4.1 Plots")

# Create a figure 
fig = plt.figure(figsize=(24,6))
# Subdivide the figure (rows x columns)
gs = fig.add_gridspec(2,8)
# For each model in list
for m in models.index:

    # Save the simplier model and system name
    model = str(models.iloc[m]['short_institution'])
    system = str(models.iloc[m]['short_name'])

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

    # Base name for hindcast
    hcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)
    # Base name for forecast
    fcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_forecast{fcy}_monthly'.format(**config)

    # Read forecasted anomaly
    anom_mean = xr.open_dataarray(f'{FOREDIR}/{fcst_bname}-{meses}_{var}-anomaly.nc')
    # Read forecasted probabilities
    probs_forecast = xr.open_dataarray(f'{FOREDIR}/{fcst_bname}-{meses}_{var}-probability.nc')               
    probs_forecast_max = probs_forecast.where(probs_forecast==probs_forecast.max(dim='category'))

    # A map with projection ccrs.PlateCarree()
    ax1 = fig.add_subplot(gs[0,m],projection=ccrs.PlateCarree())
    # We define the map extension
    ax1.set_extent(box_limits, crs=ccrs.PlateCarree())
    # Borders and coastlines
    ax1.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
    ax1.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
    # Lat Lon gridlines and labels (but not right and top labels)
    gl = ax1.gridlines(draw_labels=True)
    gl.right_labels = False
    gl.top_labels = False
    # Left labels only for left panel
    if m!=0:
        gl.left_labels = False  
    # Filled contour of each tercile probabilities
    cs1 = ax1.contourf(anom_mean.lon,anom_mean.lat,probs_forecast_max.sel(category=0),levels=ter_options['lower'][0],cmap=ter_options['lower'][1],extend='max')
    #cs2 = ax1.contourf(hindcast_mean.lon,hindcast_mean.lat,probs_forecast_max.sel(category=1),levels=ter_options['middle'][0],cmap=ter_options['middle'][1],extend='max')
    cs3 = ax1.contourf(anom_mean.lon,anom_mean.lat,probs_forecast_max.sel(category=2),levels=ter_options['upper'][0],cmap=ter_options['upper'][1],extend='max')
    # Unique colorbar
    if m==7:
        # Colorbar position 1
        cax1 = ax1.inset_axes([1.05, 0., 0.05, 1.])
        cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
        cb1.set_ticks([])
        cb1.ax.set_title('lower', loc='left', rotation=45,fontsize=8)
        # Colorbar position 2
        # cax2 = ax1.inset_axes([1.1, 0., 0.05, 1.])
        # cb2 = plt.colorbar(cs2, cax=cax2, orientation='vertical')
        # cb2.set_ticks([])
        # cb2.ax.set_title('middle', loc='left', rotation=45,fontsize=8)
        # Colorbar position 3
        cax3 = ax1.inset_axes([1.10, 0., 0.05, 1.])
        cb3 = plt.colorbar(cs3, cax=cax3, orientation='vertical')
        cb3.set_ticks([i for i in cb3.get_ticks()])
        cb3.ax.set_yticklabels(["{:.1%}".format(i) for i in cb3.get_ticks()])
        cb3.ax.set_title('upper', loc='left', rotation=45,fontsize=8)
    # Include title
    plt.title(str(models.iloc[m]['institution'])+' '+str(models.iloc[m]['name']),loc='center',fontsize=10)      

    # Read the score data file
    rpss = xr.open_dataset(f'{SCOREDIR}/{hcst_bname}.{aggr}.rpss.nc')
    # Select forecast month
    thisrpss = rpss.sel(forecastMonth=fcmonth)[var]

    # A map with projection ccrs.PlateCarree()
    ax2 = fig.add_subplot(gs[1,m],projection=ccrs.PlateCarree())
    # We define the map extension
    ax2.set_extent(box_limits, crs=ccrs.PlateCarree())
    # Borders and coastlines
    ax2.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
    ax2.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=2.)
    # Lat Lon gridlines and labels (but not right and top labels)
    gl = ax2.gridlines(draw_labels=True)
    gl.right_labels = False
    gl.top_labels = False
    # Left labels only for left panel
    if m!=0:
        gl.left_labels = False  
    # Filled contour
    cs1 = ax2.contourf(anom_mean.lon,anom_mean.lat,thisrpss,levels=score_options['rpss'][0],cmap=score_options['rpss'][1],extend=score_options['rpss'][3])
    # Unique colorbar
    if m==7:
        cax1 = ax2.inset_axes([1.05, 0., 0.05, 1.])
        cb1 = plt.colorbar(cs1, cax=cax1, orientation='vertical')
# Include figure title
scorename = score_options['rpss'][4]
fig.suptitle(f'Forecasted probabilities and {scorename}\n Forecast start: {year}-{startmonth:02d}, Valid time: {meses}\n Variable: {var_options[var][4]}', fontsize=18)

# Save figure
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1, hspace=0.01, wspace=0.01)
figure_bname = 'Allmodels_stmonth{start_month:02d}_forecast{fcy}_monthly'.format(**config)
fig.savefig(PLOTSDIR+f'/{figure_bname}-{meses}_{var}_{region}.png')  
