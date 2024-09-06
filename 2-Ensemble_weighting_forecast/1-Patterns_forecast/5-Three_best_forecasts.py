# %% [markdown]
# # 5. Three best forecasts

# This script selects the three best forecasts of each EOF (according to the RPSS value),
# and represents the results.
# 
# First we have to decide a valid year and month and an aggregation.
# 
# NOTE: This script is used to select the three best forecasts among different models and initializations,
# so there is no need to introduce a start month and year (because that would "fix" a specific initialization).
# Instead, once the different initializations were computed for a specific valid month or season, we would run this script.
# So here we have to introduce the desired valid year and month/season, not the start year and month. 

#%%
print("5. Three best forecasts")

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
    year_end = int(sys.argv[1])
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
    year_end = input("Resultados para el año: ")

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

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
HINDDIR = os.getenv('HIND_DIR')
FOREDIR = os.getenv('FORE_DIR')
MODES_HINDDIR = HINDDIR + '/modes'
SCORE_HINDDIR = HINDDIR + '/scores'
MODES_FOREDIR = FOREDIR + '/modes'
SCORE_FOREDIR = FOREDIR + '/scores'

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
# ## 5.1 Boxplots

# We calculate forecasted PCs values by comparing forecast and hindcast values,
# and then we represent the forecasted PCs with boxplots.

#%%
print("5.1 Boxplots")


# For each eof (eof1, eof2, eof3, eof4) 
for m in range(4):

    # Read the csv with the forecast
    try:
        forecast_eofx = pd.read_csv(PLOTSDIR+f'/Forecasted_EOF{m+1}.csv',index_col=0)
    except:
        print(f'Aún no se ha realizado ninguna predicción de este patrón de variabilidad: EOF{str(m+1)}')

    # Sort the values
    forecast_eofx = forecast_eofx.sort_values('RPSS',ascending=False).reset_index(drop=True)

    # Create the figure 
    fig = plt.figure(figsize=(16,6))
    # Subdivide the figure (rows x columns)
    gs = fig.add_gridspec(1,3)

    # For the three best forecasts
    for forecast in range(3):

        # Here we save the configuration
        config = dict(
            list_vars = 'geopotential',
            pressure_level = '500',
            fcy = forecast_eofx.loc[forecast]['Initialization'].split('-')[0],
            hcstarty = 1993,
            hcendy = 2016,
            start_month = forecast_eofx.loc[forecast]['Initialization'].split('-')[1],
            origin = models[(models['institution']==forecast_eofx.loc[forecast]['institution']) & (models['name']==forecast_eofx.loc[forecast]['name'])]['short_institution'].values[0],
            system = models[(models['institution']==forecast_eofx.loc[forecast]['institution']) & (models['name']==forecast_eofx.loc[forecast]['name'])]['short_name'].values[0],
        )

        # Base name for hindcast
        hcst_bname = '{origin}_s{system}_stmonth{start_month}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)
        # File name for hindcast PCs
        hpcs_fname = f'{MODES_HINDDIR}/{hcst_bname}.{aggr}.PCs.nc'
        # Base name for forecast
        fcst_bname = '{origin}_s{system}_stmonth{start_month}_forecast{fcy}_monthly'.format(**config)
        # File name for forecast PCs
        fpcs_fname = f'{MODES_FOREDIR}/{fcst_bname}.{aggr}.PCs.nc'
        # File name for hindcast scores
        corr_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.corr.nc'
        rpss_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.rpss.nc'

        # Obtain forecast month
        fcmonth = (endmonth-int(config['start_month']))+1 if (endmonth-int(config['start_month']))>=0 else (endmonth-int(config['start_month']))+13

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
        ax1 = fig.add_subplot(gs[0,forecast])
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
        plt.title('{} Forecast start: {fcy}-{start_month}'.format(forecast_eofx.loc[forecast]['Model'],**config), fontsize=12)
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
    fig.suptitle(f'Variability pattern EOF{str(m+1)} Valid time: {year_end}-{meses}', fontsize=18)
    # Save figure
    figname = f'./{PLOTSDIR}/Three-best_patterns-forecast_EOF{str(m+1)}.png'
    fig.savefig(figname,dpi=600)  
