# %% [markdown]
# # 4. Forecast plots

# This script is used to forecast variability patterns
# 
# First we have to decide a forecast system (institution and system name), a start year and month, an aggregation and a forecast month. 

#%%
print("4. Plot forecasts")

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

    # Subset of plots to be produced
    aggr = input("Selecciona el tipo de agregación mensual [ 1m , 3m ]: ")

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
)

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
HINDDIR = os.getenv('HIND_DIR')
FOREDIR = os.getenv('FORE_DIR')
MODES_HINDDIR = HINDDIR + '/modes'
SCORE_HINDDIR = HINDDIR + '/scores'
MODES_FOREDIR = FOREDIR + '/modes'
SCORE_FOREDIR = FOREDIR + '/scores'

# Base name for hindcast
hcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)
# File name for hindcast PCs
hpcs_fname = f'{MODES_HINDDIR}/{hcst_bname}.{aggr}.PCs.nc'
# Base name for forecast
fcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_forecast{fcy}_monthly'.format(**config)
# File name for forecast PCs
fpcs_fname = f'{MODES_FOREDIR}/{fcst_bname}.{aggr}.PCs.nc'
# File name for hindcast scores
corr_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.corr.nc'
rpss_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.rpss.nc'

# Check if files exist
if not os.path.exists(hpcs_fname):
    print('No se calcularon aún las PCs del hindcast de este modelo y sistema')
    sys.exit()
elif not os.path.exists(fpcs_fname):
    print('No se calcularon aún las PCs del forecast de este modelo y sistema')
    sys.exit()
elif not (os.path.exists(corr_fname) & os.path.exists(rpss_fname)):
    print('No se realizó aún la verificación del hindcast de este modelo y sistema')
    sys.exit()

# If the end month is not in the same year than the start month, we take it into account
if endmonth<startmonth:
    year_end = str(int(year)+1)
else:
    year_end = year
# We select a directory to save the results
PLOTSDIR = f'./PLOTS/{year_end}-{endmonth:02d}_{meses}'
# Miramos si existe el directorio
if not os.path.exists(PLOTSDIR):
    # Si no existe lo creamos
    try:
        os.system(f'mkdir -p {PLOTSDIR}')
    except FileExistsError:
        pass

# %% [markdown]
# ## 4.1 Calculate forecasted PCs 

# We calculate forecasted PCs values by comparing forecast and hindcast values.

#%%
print("4.1 Calculate forecasted PCs")


# For each eof (eof1, eof2, eof3, eof4) 
for m in range(4):

    # Read the csv where PCs forecasts are saved
    try:
        forecast_eofx = pd.read_csv(PLOTSDIR+f'/Forecasted_EOF{m+1}.csv',index_col=0)
    # If it hasn't been created yet, we create it
    except:
        forecast_eofx = pd.DataFrame({})

    # Reading hindcast data from file
    hpcs_eofx = xr.open_dataarray(hpcs_fname).sel(forecastMonth=fcmonth,mode=m)
    # Reading forecast data from file
    fpcs_eofx = xr.open_dataarray(fpcs_fname).sel(forecastMonth=fcmonth,mode=m)
    # Reading scores data from file
    corr_eofx = xr.open_dataarray(corr_fname).sel(forecastMonth=fcmonth,mode=m)
    rpss_eofx = xr.open_dataarray(rpss_fname).sel(forecastMonth=fcmonth,mode=m)

    # Quantify tercile thresholds
    low = hpcs_eofx.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
    high = hpcs_eofx.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
    # Quantify tercile averages
    low_value = hpcs_eofx.where(hpcs_eofx<low).mean(dim=['number','start_date'])
    med_value = hpcs_eofx.where((hpcs_eofx<=high) & (hpcs_eofx>=low)).mean(dim=['number','start_date'])
    high_value = hpcs_eofx.where(hpcs_eofx>high).mean(dim=['number','start_date'])
    # Calculate the probability of each tercile (number of ensemble members above the threshold divided by the ensemble size), and assigning a representative value of each tercile
    fpcs_eofx_low = (fpcs_eofx.where(fpcs_eofx<low).count(dim='number')/float(fpcs_eofx.number.size)).assign_coords({'category':float(low_value)})
    fpcs_eofx_med = (fpcs_eofx.where((fpcs_eofx<=high) & (fpcs_eofx>=low)).count(dim='number')/float(fpcs_eofx.number.size)).assign_coords({'category':float(med_value)})
    fpcs_eofx_high = (fpcs_eofx.where(fpcs_eofx>high).count(dim='number')/float(fpcs_eofx.number.size)).assign_coords({'category':float(high_value)})
    # Concatenate the three terciles
    fpcs_eofx_ter = xr.concat([fpcs_eofx_low,fpcs_eofx_med,fpcs_eofx_high],dim='category')
    # Find the most probable tercile and assign the corresponding value
    fpcs_eofx_val = fpcs_eofx_ter.idxmax(dim='category').drop('mode')

    # Create a DataFrame row with the corresponding information
    fpcs_eofx_new = pd.DataFrame({'Initialization': [f'{year}-{startmonth:02d}'], 'Model': [institution+'-'+name], 'institution': [institution], 'name': [name], 'PC_value': [round(float(fpcs_eofx_val),4)], 'RPSS': [round(float(rpss_eofx),4)], 'Correlation': [round(float(corr_eofx),4)]})
    # Concatenate the present row with the former PCs forecasts
    forecast_eofx = pd.concat([forecast_eofx,fpcs_eofx_new]).reset_index(drop=True)
    # Remove duplicated rows
    forecast_eofx = forecast_eofx.drop_duplicates().reset_index(drop=True)
    # Save the DataFrame
    forecast_eofx.to_csv(PLOTSDIR+f'/Forecasted_EOF{m+1}.csv')

# %% [markdown]
# ## 4.2 Boxplots

# We represent the forecasted PCs with boxplots.

#%%
print("4.2 Boxplots")


# Create a figure 
fig = plt.figure(figsize=(12,10))
# Subdivide the figure (rows x columns)
gs = fig.add_gridspec(2,2)

# For each eof (eof1, eof2, eof3, eof4) 
for m in range(4):

    # Reading hindcast data from file
    hpcs_eofx = xr.open_dataarray(hpcs_fname).sel(forecastMonth=fcmonth,mode=m)
    # Reading forecast data from file
    fpcs_eofx = xr.open_dataarray(fpcs_fname).sel(forecastMonth=fcmonth,mode=m)
    # Reading scores data from file
    corr_eofx = xr.open_dataarray(corr_fname).sel(forecastMonth=fcmonth,mode=m)
    rpss_eofx = xr.open_dataarray(rpss_fname).sel(forecastMonth=fcmonth,mode=m)

    # Quantify tercile thresholds
    low = hpcs_eofx.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
    high = hpcs_eofx.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
    # Quantify tercile averages
    low_value = hpcs_eofx.where(hpcs_eofx<low).mean(dim=['number','start_date'])
    med_value = hpcs_eofx.where((hpcs_eofx<=high) & (hpcs_eofx>=low)).mean(dim=['number','start_date'])
    high_value = hpcs_eofx.where(hpcs_eofx>high).mean(dim=['number','start_date'])
    # Calculate the probability of each tercile (number of ensemble members above the threshold divided by the ensemble size), and assigning a representative value of each tercile
    fpcs_eofx_low = (fpcs_eofx.where(fpcs_eofx<low).count(dim='number')/float(fpcs_eofx.number.size)).assign_coords({'category':float(low_value)})
    fpcs_eofx_med = (fpcs_eofx.where((fpcs_eofx<=high) & (fpcs_eofx>=low)).count(dim='number')/float(fpcs_eofx.number.size)).assign_coords({'category':float(med_value)})
    fpcs_eofx_high = (fpcs_eofx.where(fpcs_eofx>high).count(dim='number')/float(fpcs_eofx.number.size)).assign_coords({'category':float(high_value)})

    # One subplot
    ax1 = fig.add_subplot(gs[m-2*(m//2),m//2])
    # Convert data to Pandas Series
    fpcs_eofx_values = pd.Series(fpcs_eofx.values)
    # Define x limits
    bound_h = np.max([np.abs(hpcs_eofx.min().values),np.abs(hpcs_eofx.max().values)])
    bound_f = np.max([np.abs(fpcs_eofx.min().values),np.abs(fpcs_eofx.max().values)])
    bound = np.max([bound_h,bound_f])+0.1
    # Pdfs 
    bins = np.linspace(-1*bound,bound,20)
    n, bins, patch = plt.hist(fpcs_eofx_values, bins=bins, density=True, color='grey', alpha=0.4, rwidth=0.85,zorder=-1)
    fpcs_eofx_values.plot.density(ax=ax1, color='grey', lw=2., label='Lower') # identical to s.plot.kde(...)
    # Boxplot
    height = round(n.max()*1.8)
    medianprops = dict(linestyle='-', linewidth=2., color='k')
    meanprops = dict(marker='o', markeredgecolor='black', markerfacecolor='w')
    bplot = ax1.boxplot(fpcs_eofx_values, positions=[height*0.75-height*0.07], widths=[height*0.04], vert=False, showfliers=True, patch_artist=True, notch=True, showmeans=True, meanprops=meanprops, medianprops=medianprops)
    bplot['boxes'][0].set_facecolor('grey')
    ax1.text(fpcs_eofx.mean(), height*0.8-height*0.07, '{:.3f}'.format(fpcs_eofx.mean()),family='sans-serif',weight='bold',size=11, horizontalalignment='center', verticalalignment='top')
    # Low tercile separation
    ax1.vlines(low, 0., height, color='dodgerblue', linestyle='--', lw=1.5, alpha=0.6, zorder=0)
    # Zero line
    ax1.vlines(0., 0., height, color='k', linestyle='-', lw=1., alpha=0.6, zorder=0)
    # High tercile separation
    ax1.vlines(high, 0., height, color='firebrick', linestyle='--', lw=1.5, alpha=0.6, zorder=0)
    # Subplot limits
    ax1.set_ylim([0.,height])
    ax1.set_xlim([-1*bound,bound])
    ax1.set_yticks(np.linspace(0.,height,height+1),labels=np.linspace(0.,height,height+1))
    # Title
    plt.title(f'EOF{str(m+1)} forecast', fontsize=12)
    # Tercile information: forecasted probability and reference value (average)
    textstr= '\n'.join((
        'Lower:',
        r'Probability={:.2f}'.format(fpcs_eofx_low),
        r'Ref. value={:.2f}'.format(low_value)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.01, 0.99, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    textstr= '\n'.join((
        'Middle:',
        r'Probability={:.2f}'.format(fpcs_eofx_med),
        r'Ref. value={:.2f}'.format(med_value)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.5, 0.99, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='center', bbox=props)      
    textstr= '\n'.join((
        'Upper:',
        r'Probability={:.2f}'.format(fpcs_eofx_high),
        r'Ref. value={:.2f}'.format(high_value)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.99, 0.99, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)      
    # Skill score information: Correlation and RPSS
    textstr= '\n'.join((
        r'Correlation={:.2f}'.format(corr_eofx),
        r'RPSS={:.2f}'.format(rpss_eofx)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.99, 0.5, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='right', bbox=props)

# Figure title
fig.suptitle(f'{institution} {name}\n Forecast start: {year}-{startmonth:02d}, Valid time: {meses}\n Variability patterns', fontsize=18)
# Save figure
figname = f'./{PLOTSDIR}/{fcst_bname}_patterns-forecast.png'
fig.savefig(figname,dpi=600)  
