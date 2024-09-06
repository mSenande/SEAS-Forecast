# %% [markdown]
# # 5. Compute deterministic and probabilistic scores (Multimodel verification)

# This script is used to compute different verification scores 
# for multimodel monthly seasonal forescasts of surface variables, previously applying a ensemble weighting technique.
#
# The ensemble weighting technique is based on the 4 main varability patterns, whose values were predicted in 1-PCs_forecast.py script.
# 
# The computed scores are: Spearman's rank correlation, area under Relative Operating Characteristic (ROC) curve, 
# Relative Operating Characteristic Skill Score (ROCSS), Ranked Probability Score (RPS), Ranked Probability Skill Score (RPSS) and Brier Score (BS).
#
# First we have to decide a start month. 

#%%
print("5. Multimodel verification")

import os
import sys
from dotenv import load_dotenv
import xarray as xr
import pandas as pd
import numpy as np
import xskillscore as xs
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# Read the file with model names
models = pd.read_csv('./models_running.csv')

# If variables are introduced from the command line, read them
if len(sys.argv) > 1:
    startmonth = int(sys.argv[1])
# If no variables were introduced, ask for them
else:
    # Which start month
    startmonth = int(input("Mes de inicialización (en número): "))

# Here we save the configuration
config = dict(
    list_vars = ['2m_temperature', 'mean_sea_level_pressure', 'total_precipitation'],
    hcstarty = 1993,
    hcendy = 2016,
    start_month = startmonth,
)

# Directory selection
load_dotenv() # Data is read from a path defined in file .env
HINDDIR = os.getenv('HIND_DIR')
# Base name for observations
obs_bname = 'era5_monthly_stmonth{start_month:02d}_{hcstarty}-{hcendy}'.format(**config)
# File name for observations
obs_fname = f'{HINDDIR}/{obs_bname}.grib'

# Check if files exist
if not os.path.exists(obs_fname):
    print('No se descargaron aún los datos de ERA5')
    sys.exit()

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
MODES_HINDDIR = os.getenv('MODES_HIND_DIR')  # Directory where hindcast variability patterns files are located
POSTDIR_VER = os.getenv('POST_DIR_VER')
NEW_MODES_HINDDIR = POSTDIR_VER + '/modes' # Directory where variability patterns forecasts will be saved (only for verification)
NEW_SCORES_HINDDIR = POSTDIR_VER + '/scores' # Directory where new skill scores will be saved (only for verification)

# Directory creation
for directory in [NEW_SCORES_HINDDIR]:
    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

# Base name for multimodel hindcast
multi_hcst_bname = 'multimodel_stmonth{start_month:02d}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)
aggr_list = ['1m','3m']

# %% [markdown]
# ## 5.1 Multimodel hindcast anomalies and probabilities

# We calculate the monthly and 3-months anomalies for the hindcast data.

#%%
print("5.1 Multimodel hindcast anomalies and probabilities")

# We define a function to calculate the boundaries of forecast categories defined by quantiles
def get_thresh(icat,quantiles,xrds,dims=['number','start_date']):
    if not all(elem in xrds.dims for elem in dims):           
        raise Exception('Some of the dimensions in {} is not present in the xr.Dataset {}'.format(dims,xrds)) 
    else:
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

l_probs_models_1m=list()
l_anoms_models_1m=list()
l_probs_models_3m=list()
l_anoms_models_3m=list()
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
    # Check if files exist
    if not os.path.exists(hcst_fname):
        print('No se descargaron aún los datos de hindcast de este modelo y sistema')
        sys.exit()

    # File name for hindcast
    hcpcs_fname = f'{MODES_HINDDIR}/{hcst_bname}.1m.PCs.nc'
    hcpcs_3m_fname = f'{MODES_HINDDIR}/{hcst_bname}.3m.PCs.nc'
    # Reading HCST data from file
    hcpcs = xr.open_dataset(hcpcs_fname)
    hcpcs_3m = xr.open_dataset(hcpcs_3m_fname)

    # Reading variability patterns forecast data from file
    forepcs = xr.open_dataset(f'{NEW_MODES_HINDDIR}/PCs_best_forecasts_stmonth{startmonth:02d}.1m.nc')
    forepcs_3m = xr.open_dataset(f'{NEW_MODES_HINDDIR}/PCs_best_forecasts_stmonth{startmonth:02d}.3m.nc')

    # For 1m aggregation
    l_anom=list()
    # For each forecastMonth
    for this_fcmonth in forepcs.forecastMonth.values:
        # Select forecast values
        thisfore = forepcs.sel(forecastMonth=this_fcmonth)

        # If forecastMonth=2 (leadtime=1), we have 2 additional forecasts available that can have better skill for the same valid month
        if this_fcmonth==2:
            # This is the variability pattern forecast of forecastMonth=2 (leadtime=1)
            # We will use this variable to save all the best forecasts
            thisfore_prev0 = thisfore.swap_dims({"start_date": "valid_time"})
            # This is the variability pattern forecast of forecastMonth=3 (leadtime=2)
            # Since the valid month must be the same, we have to change the startmonth
            startmonth_prev1 = startmonth-1 if startmonth-1>0 else startmonth-1+12
            forepcs_prev1 = xr.open_dataset(f'{NEW_MODES_HINDDIR}/PCs_best_forecasts_stmonth{startmonth_prev1:02d}.1m.nc')
            thisfore_prev1 = forepcs_prev1.sel(forecastMonth=this_fcmonth+1).swap_dims({"start_date": "valid_time"})
            # This is the variability pattern forecast of forecastMonth=4 (leadtime=3)
            # Since the valid month must be the same, we have to change the startmonth
            startmonth_prev2 = startmonth-2 if startmonth-2>0 else startmonth-2+12
            forepcs_prev2 = xr.open_dataset(f'{NEW_MODES_HINDDIR}/PCs_best_forecasts_stmonth{startmonth_prev2:02d}.1m.nc')
            thisfore_prev2 = forepcs_prev2.sel(forecastMonth=this_fcmonth+2).swap_dims({"start_date": "valid_time"})
            # For each variability pattern / mode:
            for m in thisfore.mode.values:
                # We check if the variability pattern forecast skill of forecastMonth=4 (leadtime=3) is greater than that of forecastMonth=3 (leadtime=2) and forecastMonth=2 (leadtime=1)
                if (thisfore_prev2.sel(mode=m).rpss>thisfore_prev0.sel(mode=m).rpss) & (thisfore_prev2.sel(mode=m).rpss>thisfore_prev1.sel(mode=m).rpss):
                    # If so, we select the values of this forecast
                    # We have to do this in a loop around all the years and try/except commands, because for the first year we may not have the forecastMonth=4 (leadtime=3) forecast 
                    for t in thisfore.valid_time.values:
                        try:
                            thisfore_prev0.loc[dict(valid_time=t,mode=m)] = thisfore_prev2.sel(valid_time=t,mode=m)
                        except:
                            pass
                    # We change also the corresponding rpss value
                    thisfore_prev0.loc[dict(mode=m)].rpss.values = thisfore_prev2.sel(mode=m).rpss.values
                # We check if the variability pattern forecast skill of forecastMonth=3 (leadtime=2) is greater than that of forecastMonth=4 (leadtime=3) and forecastMonth=2 (leadtime=1)
                elif (thisfore_prev1.sel(mode=m).rpss>thisfore_prev0.sel(mode=m).rpss) & (thisfore_prev1.sel(mode=m).rpss>thisfore_prev2.sel(mode=m).rpss):
                    # If so, we select the values of this forecast
                    # We have to do this in a loop around all the years and try/except commands, because for the first year we may not have the forecastMonth=3 (leadtime=2) forecast 
                    for t in thisfore.valid_time.values:
                        try:
                            thisfore_prev0.loc[dict(valid_time=t,mode=m)] = thisfore_prev1.sel(valid_time=t,mode=m)
                        except:
                            pass
                    # We change also the corresponding rpss value
                    thisfore_prev0.loc[dict(mode=m)].rpss.values = thisfore_prev1.sel(mode=m).rpss.values
            # We save the best-forecasts values in the thisfore variable
            thisfore.pseudo_pcs.values = thisfore_prev0.pseudo_pcs.values
            thisfore.rpss.values = thisfore_prev0.rpss.values

        # If forecastMonth=3 (leadtime=2), we have 1 additional forecasts available that can have better skill for the same valid month
        if this_fcmonth==3:
            # This is the variability pattern forecast of forecastMonth=3 (leadtime=2)
            # We will use this variable to save all the best forecasts
            thisfore_prev0 = thisfore.swap_dims({"start_date": "valid_time"})
            # This is the variability pattern forecast of forecastMonth=4 (leadtime=3)
            # Since the valid month must be the same, we have to change the startmonth
            startmonth_prev1 = startmonth-1 if startmonth-1>0 else startmonth-1+12
            forepcs_prev1 = xr.open_dataset(f'{NEW_MODES_HINDDIR}/PCs_best_forecasts_stmonth{startmonth_prev1:02d}.1m.nc')
            thisfore_prev1 = forepcs_prev1.sel(forecastMonth=this_fcmonth+1).swap_dims({"start_date": "valid_time"})
            # For each variability pattern / mode:
            for m in thisfore.mode.values:
                # We check if the variability pattern forecast skill of forecastMonth=4 (leadtime=3) is greater than that of forecastMonth=3 (leadtime=2)
                if (thisfore_prev1.sel(mode=m).rpss>thisfore_prev0.sel(mode=m).rpss):
                    # If so, we select the values of this forecast
                    # We have to do this in a loop around all the years and try/except commands, because for the first year we may not have the forecastMonth=4 (leadtime=3) forecast 
                    for t in thisfore.valid_time.values:
                        try:
                            thisfore_prev0.loc[dict(valid_time=t,mode=m)] = thisfore_prev1.sel(valid_time=t,mode=m)
                        except:
                            pass
                    # We change also the corresponding rpss value
                    thisfore_prev0.loc[dict(mode=m)].rpss.values = thisfore_prev1.sel(mode=m).rpss.values
            # We save the best-forecasts values in the thisfore variable
            thisfore.pseudo_pcs.values = thisfore_prev0.pseudo_pcs.values
            thisfore.rpss.values = thisfore_prev0.rpss.values

        # Select hindcast values
        thishcst = hcpcs.sel(forecastMonth=this_fcmonth)
        # Compute anomaly
        pcs_anom = (thishcst - thisfore)**2.
        # Append anomalies
        l_anom.append( pcs_anom )
    # Concat along forecastMonth
    hcpcs_anom=xr.concat(l_anom,dim='forecastMonth')

    # For 3m aggregation
    l_anom=list()
    # For each forecastMonth
    for this_fcmonth in forepcs_3m.forecastMonth.values:
        # Select forecast values
        thisfore = forepcs_3m.sel(forecastMonth=this_fcmonth)

        # If forecastMonth=4 (leadtime=1), we have 2 additional forecasts available that can have better skill for the same valid month
        if this_fcmonth==4:
            # This is the variability pattern forecast of forecastMonth=4 (leadtime=1)
            # We will use this variable to save all the best forecasts
            thisfore_prev0 = thisfore.swap_dims({"start_date": "valid_time"})
            # This is the variability pattern forecast of forecastMonth=5 (leadtime=2)
            # Since the valid month must be the same, we have to change the startmonth
            startmonth_prev1 = startmonth-1 if startmonth-1>0 else startmonth-1+12
            forepcs_prev1 = xr.open_dataset(f'{NEW_MODES_HINDDIR}/PCs_best_forecasts_stmonth{startmonth_prev1:02d}.3m.nc')
            thisfore_prev1 = forepcs_prev1.sel(forecastMonth=this_fcmonth+1).swap_dims({"start_date": "valid_time"})
            # This is the variability pattern forecast of forecastMonth=6 (leadtime=3)
            # Since the valid month must be the same, we have to change the startmonth
            startmonth_prev2 = startmonth-2 if startmonth-2>0 else startmonth-2+12
            forepcs_prev2 = xr.open_dataset(f'{NEW_MODES_HINDDIR}/PCs_best_forecasts_stmonth{startmonth_prev2:02d}.3m.nc')
            thisfore_prev2 = forepcs_prev2.sel(forecastMonth=this_fcmonth+2).swap_dims({"start_date": "valid_time"})
            # For each variability pattern / mode:
            for m in thisfore.mode.values:
                # We check if the variability pattern forecast skill of forecastMonth=6 (leadtime=3) is greater than that of forecastMonth=5 (leadtime=2) and forecastMonth=4 (leadtime=1)
                if (thisfore_prev2.sel(mode=m).rpss>thisfore_prev0.sel(mode=m).rpss) & (thisfore_prev2.sel(mode=m).rpss>thisfore_prev1.sel(mode=m).rpss):
                    # If so, we select the values of this forecast
                    # We have to do this in a loop around all the years and try/except commands, because for the first year we may not have the forecastMonth=6 (leadtime=3) forecast 
                    for t in thisfore.valid_time.values:
                        try:
                            thisfore_prev0.pseudo_pcs.loc[dict(valid_time=t,mode=m)] = thisfore_prev2.pseudo_pcs.loc[dict(valid_time=t,mode=m)].values
                        except:
                            pass
                    # We change also the corresponding rpss value
                    thisfore_prev0.rpss.loc[dict(mode=m)].values = thisfore_prev2.rpss.loc[dict(mode=m)].values
                # We check if the variability pattern forecast skill of forecastMonth=5 (leadtime=2) is greater than that of forecastMonth=6 (leadtime=3) and forecastMonth=4 (leadtime=1)
                elif (thisfore_prev1.sel(mode=m).rpss>thisfore_prev0.sel(mode=m).rpss) & (thisfore_prev1.sel(mode=m).rpss>thisfore_prev2.sel(mode=m).rpss):
                    # If so, we select the values of this forecast
                    # We have to do this in a loop around all the years and try/except commands, because for the first year we may not have the forecastMonth=6 (leadtime=3) forecast 
                    for t in thisfore.valid_time.values:
                        try:
                            thisfore_prev0.pseudo_pcs.loc[dict(valid_time=t,mode=m)] = thisfore_prev1.pseudo_pcs.loc[dict(valid_time=t,mode=m)].values
                        except:
                            pass
                    # We change also the corresponding rpss value
                    thisfore_prev0.rpss.loc[dict(mode=m)] = thisfore_prev1.rpss.loc[dict(mode=m)].values
            # We save the best-forecasts values in the thisfore variable
            thisfore.pseudo_pcs.values = thisfore_prev0.pseudo_pcs.values
            thisfore.rpss.values = thisfore_prev0.rpss.values

        # If forecastMonth=5 (leadtime=2), we have 1 additional forecast available that can have better skill for the same valid month
        if this_fcmonth==5:
            # This is the variability pattern forecast of forecastMonth=5 (leadtime=2)
            # We will use this variable to save all the best forecasts
            thisfore_prev0 = thisfore.swap_dims({"start_date": "valid_time"})
            # This is the variability pattern forecast of forecastMonth=6 (leadtime=3)
            # Since the valid month must be the same, we have to change the startmonth
            startmonth_prev1 = startmonth-1 if startmonth-1>0 else startmonth-1+12
            forepcs_prev1 = xr.open_dataset(f'{NEW_MODES_HINDDIR}/PCs_best_forecasts_stmonth{startmonth_prev1:02d}.3m.nc')
            thisfore_prev1 = forepcs_prev1.sel(forecastMonth=this_fcmonth+1).swap_dims({"start_date": "valid_time"})
            # For each variability pattern / mode:
            for m in thisfore.mode.values:
                # We check if the variability pattern forecast skill of forecastMonth=6 (leadtime=3) is greater than that of forecastMonth=5 (leadtime=2)
                if (thisfore_prev1.sel(mode=m).rpss>thisfore_prev0.sel(mode=m).rpss):
                    for t in thisfore.valid_time.values:
                        try:
                            thisfore_prev0.pseudo_pcs.loc[dict(valid_time=t,mode=m)] = thisfore_prev1.pseudo_pcs.loc[dict(valid_time=t,mode=m)].values
                        except:
                            pass
                    # We change also the corresponding rpss value
                    thisfore_prev0.rpss.loc[dict(mode=m)] = thisfore_prev1.rpss.loc[dict(mode=m)].values
            # We save the best-forecasts values in the thisfore variable
            thisfore.pseudo_pcs.values = thisfore_prev0.pseudo_pcs.values
            thisfore.rpss.values = thisfore_prev0.rpss.values

        # Select hindcast values
        thishcst = hcpcs_3m.sel(forecastMonth=this_fcmonth)
        # Compute anomaly
        pcs_anom = (thishcst - thisfore)**2.
        # Append anomalies
        l_anom.append( pcs_anom )
    # Concat along forecastMonth
    hcpcs_3m_anom=xr.concat(l_anom,dim='forecastMonth')

    # Variances for 1m aggregation
    # We create an array (4x12) with all the percentages of explained variance associated with each variability pattern for each month
    eof_variances = xr.DataArray(np.zeros([4,12]),coords={'mode':[0,1,2,3],'month':range(1,13)})
    # For each valid month
    for validmonth in range(1,13):
        # Explained variance percentages are stored in csv files
        variances = pd.read_csv(f'{MODES_HINDDIR}/ERA5_VAR_{validmonth:02d}.csv')
        eof_variances.loc[dict(month=validmonth)]=variances['VAR']
    # We convert the month coordinate into a forecastMonth coordinate
    forecastmonths = eof_variances.month-startmonth+1
    forecastmonths = xr.where(forecastmonths<=0,forecastmonths+12,forecastmonths)
    eof_variances = eof_variances.assign_coords(forecastMonth= forecastmonths)
    eof_variances = eof_variances.swap_dims({"month": "forecastMonth"}).drop('month')
    # We sort the values by the forecastMonth coordinate, and delate the excess values (forecastMonth>6 and forecastMonth<2)
    eof_variances = eof_variances.sortby('forecastMonth')
    eof_variances = eof_variances.where(eof_variances.forecastMonth.isin(hcpcs.forecastMonth.values),drop=True)

    # Variances for 3m aggregation
    # We create an array (4x12) with all the percentages of explained variance associated with each variability pattern for each season
    eof_variances_3m = xr.DataArray(np.zeros([4,12]),coords={'mode':[0,1,2,3],'month':range(1,13)})
    vm = 1
    # For each valid season
    for validmonth in ["NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND"]:
        # Explained variance percentages are stored in csv files
        variances = pd.read_csv(f'{MODES_HINDDIR}/ERA5_VAR_{validmonth}.csv')
        eof_variances_3m.loc[dict(month=vm)]=variances['VAR']
        vm+=1
    # We convert the month coordinate into a forecastMonth coordinate
    forecastmonths = eof_variances_3m.month-startmonth+1
    forecastmonths = xr.where(forecastmonths<=0,forecastmonths+12,forecastmonths)
    eof_variances_3m = eof_variances_3m.assign_coords(forecastMonth= forecastmonths)
    eof_variances_3m = eof_variances_3m.swap_dims({"month": "forecastMonth"}).drop('month')
    # We sort the values by the forecastMonth coordinate, and delate the excess values (forecastMonth>6 and forecastMonth<4)
    eof_variances_3m = eof_variances_3m.sortby('forecastMonth')
    eof_variances_3m = eof_variances_3m.where(eof_variances_3m.forecastMonth.isin(hcpcs_3m.forecastMonth.values),drop=True)

    # We calculate the weighting funtions
    weights = 1./(1.+hcpcs_anom['pseudo_pcs'].weighted(eof_variances).mean(dim='mode'))
    weights_3m = 1./(1.+hcpcs_3m_anom['pseudo_pcs'].weighted(eof_variances_3m).mean(dim='mode'))
    # We normalize the weighting funtions
    weights_norm = (weights/weights.sum(dim='number'))
    weights_3m_norm = (weights_3m/weights_3m.sum(dim='number'))
    # We standarize the weighting funtions
    weights_off = (weights-weights.min(dim='number'))/(weights.max(dim='number')-weights.min(dim='number'))
    weights_3m_off = (weights_3m-weights_3m.min(dim='number'))/(weights_3m.max(dim='number')-weights_3m.min(dim='number'))
    weights_scal = weights_off/weights_off.sum(dim='number')
    weights_3m_scal = weights_3m_off/weights_3m_off.sum(dim='number')
    # Control for non-negative values
    weights_norm = xr.where(weights_norm<0.,0.,weights_norm)
    weights_3m_norm = xr.where(weights_3m_norm<0.,0.,weights_3m_norm)
    weights_scal = xr.where(weights_scal<0.,0.,weights_scal)
    weights_3m_scal = xr.where(weights_3m_scal<0.,0.,weights_3m_scal)
    # Control for non-nan values
    weights_norm = weights_norm.fillna(1./weights_norm.number.size)
    weights_3m_norm = weights_3m_norm.fillna(1./weights_3m_norm.number.size)
    weights_scal = weights_scal.fillna(1./weights_scal.number.size)
    weights_3m_scal = weights_3m_scal.fillna(1./weights_3m_scal.number.size)

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

    # Calculate 3-month aggregations
    hcst_3m = hcst.rolling(forecastMonth=3).mean()
    # rolling() assigns the label to the end of the N month period, so the first N-1 elements have NaN and can be dropped
    hcst_3m = hcst_3m.where(hcst_3m.forecastMonth>=3,drop=True)

    # Calculate 1m anomalies
    hcmean = hcst.mean(['number','start_date'])
    anom = hcst - hcmean
    anom = anom.assign_attrs(reference_period='{hcstarty}-{hcendy}'.format(**config))
    # Calculate 3m anomalies
    hcmean_3m = hcst_3m.mean(['number','start_date'])
    anom_3m = hcst_3m - hcmean_3m
    anom_3m = anom_3m.assign_attrs(reference_period='{hcstarty}-{hcendy}'.format(**config))

    hcst = hcst.where(hcst.forecastMonth.isin(weights_scal.forecastMonth.values),drop=True)
    hcst_3m = hcst_3m.where(hcst_3m.forecastMonth.isin(weights_3m_scal.forecastMonth.values),drop=True)
    anom = anom.where(anom.forecastMonth.isin(weights_scal.forecastMonth.values),drop=True)
    anom_3m = anom_3m.where(anom_3m.forecastMonth.isin(weights_3m_scal.forecastMonth.values),drop=True)

    # Ensemble mean
    anom = anom.weighted(weights_scal).mean('number')
    anom_3m = anom_3m.weighted(weights_3m_scal).mean('number')

    # We will create a new coordinate called 'model' and concatenae the results
    l_anoms_models_1m.append(anom.assign_coords({'model':model}))
    l_anoms_models_3m.append(anom_3m.assign_coords({'model':model}))

    # Calculate probabilities for tercile categories by counting members within each category
    quantiles = [1/3., 2/3.]
    numcategories = len(quantiles)+1
    # For each aggregation
    for aggr in aggr_list:
        if aggr=='1m':
            w = weights_scal
            h = hcst
        elif aggr=='3m':
            w = weights_3m_scal
            h = hcst_3m
        else:
            raise BaseException(f'Unknown aggregation {aggr}')

        l_probs_hcst=list()
        h_ones = xr.full_like(h, 1)
        # For each quantile
        for icat in range(numcategories):
            # Get the lower and higher threshold
            h_lo,h_hi = get_thresh(icat, quantiles, h)
            # Count the number of member between the threshold
            #probh = np.logical_and(h>h_lo, h<=h_hi).weighted(w).sum('number')/w.sum('number')#.sum('number')/float(h.number.size)
            probh = h_ones.where((h>h_lo) & (h<=h_hi)).weighted(w).sum('number')#.sum('number')/float(h.number.size)

            # Instead of using the coordinate 'quantile' coming from the hindcast xr.Dataset
            # we will create a new coordinate called 'category'
            if 'quantile' in probh:
                probh = probh.drop('quantile')
            l_probs_hcst.append(probh.assign_coords({'category':icat}))

        # Concatenating tercile probs categories
        if aggr=='1m':
            probs_1m = xr.concat(l_probs_hcst,dim='category')                    
        elif aggr=='3m':
            probs_3m = xr.concat(l_probs_hcst,dim='category')                    

    probs_1m = xr.where(probs_1m>1., 1., probs_1m)
    probs_3m = xr.where(probs_3m>1., 1., probs_3m)

    probs_1m = probs_1m.where(probs_1m.forecastMonth.isin(weights_scal.forecastMonth.values),drop=True)
    probs_3m = probs_3m.where(probs_3m.forecastMonth.isin(weights_3m_scal.forecastMonth.values),drop=True)

    # We will create a new coordinate called 'model' and concatenae the results
    l_probs_models_1m.append(probs_1m.assign_coords({'model':model}))
    l_probs_models_3m.append(probs_3m.assign_coords({'model':model}))

# We average the results of all models
anoms_1m = xr.concat(l_anoms_models_1m,dim='model').mean(dim='model')
anoms_3m = xr.concat(l_anoms_models_3m,dim='model').mean(dim='model')
probs_1m = xr.concat(l_probs_models_1m,dim='model').mean(dim='model')
probs_3m = xr.concat(l_probs_models_3m,dim='model').mean(dim='model')


# %% [markdown]
# ## 5.2 Read observation data

# We read the monthly ERA5 data and obtain 3-months means.

#%%
print("5.2 Read observation data")  

if 'total_precipitation' in config['list_vars']:
    # Total precipitation in ERA5 grib must be read separately because of time dimension
    era5_1deg_notp = xr.open_dataset(obs_fname, engine='cfgrib', backend_kwargs={'filter_by_keys': {'step': 0}})
    era5_1deg_tp = xr.open_dataset(obs_fname, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})
    # We assign the same time dimension
    era5_1deg_tp = era5_1deg_tp.assign_coords(time=era5_1deg_notp.time.values)
    # We assign the same name as in hindcast
    era5_1deg_tp = era5_1deg_tp.rename({'tp':'tprate'})
    # We merge the two datasets
    era5_1deg = xr.merge([era5_1deg_notp,era5_1deg_tp],compat='override')
    del era5_1deg_notp, era5_1deg_tp
else: 
    era5_1deg = xr.open_dataset(obs_fname, engine='cfgrib')

# Renaming to match hindcast names 
era5_1deg = era5_1deg.rename({'latitude':'lat','longitude':'lon','time':'start_date'}).swap_dims({'start_date':'valid_time'})

# Assign 'forecastMonth' coordinate values
fcmonths = [mm+1 if mm>=0 else mm+13 for mm in [t.month - config['start_month'] for t in pd.to_datetime(era5_1deg.valid_time.values)] ]
era5_1deg = era5_1deg.assign_coords(forecastMonth=('valid_time',fcmonths))
# Drop obs values not needed (earlier than first start date) - this is useful to create well shaped 3-month aggregations from obs.
era5_1deg = era5_1deg.where(era5_1deg.valid_time>=np.datetime64('{hcstarty}-{start_month:02d}-01'.format(**config)),drop=True)

# Calculate 3-month AGGREGATIONS
era5_1deg_3m = era5_1deg.rolling(valid_time=3).mean()
era5_1deg_3m = era5_1deg_3m.where(era5_1deg_3m.forecastMonth>=3)

era5_1deg = era5_1deg.where(era5_1deg.forecastMonth.isin(weights_scal.forecastMonth.values),drop=True)
era5_1deg_3m = era5_1deg_3m.where(era5_1deg_3m.forecastMonth.isin(weights_3m_scal.forecastMonth.values),drop=True)

# As we don't need it anymore at this stage, we can safely remove 'forecastMonth'
era5_1deg = era5_1deg.drop('forecastMonth')
era5_1deg_3m = era5_1deg_3m.drop('forecastMonth')


# %% [markdown]
# ## 5.3 Compute deterministic scores

# Here we calculate the Spearman's rank correlation and their p-values. 
# 
# This score is based on the ensemble mean, not on the probabilities for each tercile.

# %% 
print("5.3 Compute deterministic scores")

# Loop over aggregations
for aggr in aggr_list:
    print(aggr)
    if aggr=='1m':
        o = era5_1deg
        h = anoms_1m.compute()
    elif aggr=='3m':
        o = era5_1deg_3m
        h = anoms_3m.compute()
    else:
        raise BaseException(f'Unknown aggregation {aggr}')

    # Check if hindcast data is ensemble
    is_fullensemble = 'number' in h.dims

    l_corr=list()
    l_corr_pval=list()
    # For each forecast month
    for this_fcmonth in h.forecastMonth.values:
        # Select hindcast values
        thishcst = h.sel(forecastMonth=this_fcmonth).swap_dims({'start_date':'valid_time'})
        # Select observation values for this hindcast
        thisobs = o.where(o.valid_time==thishcst.valid_time,drop=True)
        # Calculate Spearman's rank correlation
        l_corr.append( xs.spearman_r(thishcst, thisobs, dim='valid_time') )
        # Calculate p-value
        l_corr_pval.append ( xs.spearman_r_p_value(thishcst, thisobs, dim='valid_time') )

    # Concatenating (by fcmonth) correlation
    corr=xr.concat(l_corr,dim='forecastMonth')
    corr_pval=xr.concat(l_corr_pval,dim='forecastMonth')

    # Assign attribute indicating models
    corr = corr.assign_attrs({'Models': " | ".join(models['institution']+'-'+models['name'])})
    corr_pval = corr_pval.assign_attrs({'Models': " | ".join(models['institution']+'-'+models['name'])})

    # Saving to netCDF file correlation   
    corr.to_netcdf(f'{NEW_SCORES_HINDDIR}/{multi_hcst_bname}.{aggr}.corr.nc')
    corr_pval.to_netcdf(f'{NEW_SCORES_HINDDIR}/{multi_hcst_bname}.{aggr}.corr_pval.nc')

# %% [markdown]
# ## 5.4 Compute probabilistic scores for tercile categories

# Here we calculate the probabilistic scores: area under Relative Operating Characteristic (ROC) curve, 
# Relative Operating Characteristic Skill Score (ROCSS), Ranked Probability Score (RPS), Ranked Probability Skill Score (RPSS) and Brier Score (BS). 

# %% 
print("5.4 Compute probabilistic scores for tercile categories")

# Loop over aggregations
for aggr in aggr_list:
    print(aggr)
    if aggr=='1m':
        o = era5_1deg
        probs_hcst = probs_1m.compute()
    elif aggr=='3m':
        o = era5_1deg_3m
        probs_hcst = probs_3m.compute()
    else:
        raise BaseException(f'Unknown aggregation {aggr}')
   
    l_roc=list()
    l_rps=list()
    l_rpss=list()
    l_rocss=list()
    l_bs=list()
    # For each forecast month
    for this_fcmonth in probs_hcst.forecastMonth.values:
        # Select hindcast values
        thishcst = probs_hcst.sel(forecastMonth=this_fcmonth).swap_dims({'start_date':'valid_time'})
        # Select observation values for this hindcast
        thiso = o.where(o.valid_time==thishcst.valid_time,drop=True)

        # Calculate probabilities from observations and climatology
        l_probs_obs=list()
        l_probs_clim=list()
        # For each quantile
        for icat in range(numcategories):
            # Get the lower and higher threshold
            o_lo,o_hi = get_thresh(icat, quantiles, thiso, dims=['valid_time'])
            # Count the number of "members" between the threshold (1 or 0)
            probo = 1. * np.logical_and(thiso>o_lo, thiso<=o_hi)
            if 'quantile' in probo:
                probo=probo.drop('quantile')
            l_probs_obs.append(probo.assign_coords({'category':icat}))
            # Count the number of months between the threshold (1 or 0)
            probc = np.logical_and(thiso>o_lo, thiso<=o_hi).sum('valid_time')/float(thiso.dims['valid_time'])        
            if 'quantile' in probc:
                probc=probc.drop('quantile')
            l_probs_clim.append(probc.assign_coords({'category':icat}))
        # Concatenate observations and climatology probabilities
        thisobs = xr.concat(l_probs_obs, dim='category')
        thisclim = xr.concat(l_probs_clim, dim='category')

        # Calculate the probabilistic (tercile categories) scores
        thisroc = xr.Dataset()
        thisrps = xr.Dataset()
        rpsclim = xr.Dataset()
        thisrpss = xr.Dataset()
        thisrocss = xr.Dataset()
        thisbs = xr.Dataset()
        # For each variable
        for var in thishcst.data_vars:
            # Compute Area ROC
            thisroc[var] = xs.roc(thisobs[var],thishcst[var], dim='valid_time', bin_edges=np.linspace(0,1,101))
            # Compute RPS
            thisrps[var] = xs.rps(thisobs[var],thishcst[var], dim='valid_time', category_edges=None, input_distributions='p')
            # Compute climatological RPS
            rpsclim[var] = xs.rps(thisobs[var],thisclim[var], dim='valid_time', category_edges=None, input_distributions='p')
            # Compute RPSS
            thisrpss[var] = 1.-thisrps[var]/rpsclim[var]
            # Compute ROCSS
            thisrocss[var] = (thisroc[var] - 0.5) / (1. - 0.5)
            # Compute Brier Score
            bscat = list()
            for cat in thisobs[var].category:
                thisobscat = thisobs[var].sel(category=cat)
                thishcstcat = thishcst[var].sel(category=cat)
                bscat.append(xs.brier_score(thisobscat, thishcstcat, dim='valid_time'))
            thisbs[var] = xr.concat(bscat,dim='category')
        l_roc.append(thisroc)
        l_rps.append(thisrps)
        l_rpss.append(thisrpss)
        l_rocss.append(thisrocss)
        l_bs.append(thisbs)

    # Concatenate along forecast month
    roc=xr.concat(l_roc,dim='forecastMonth').assign_attrs({'Models': " | ".join(models['institution']+'-'+models['name'])})
    rps=xr.concat(l_rps,dim='forecastMonth').assign_attrs({'Models': " | ".join(models['institution']+'-'+models['name'])})
    rpss=xr.concat(l_rpss,dim='forecastMonth').assign_attrs({'Models': " | ".join(models['institution']+'-'+models['name'])})
    rocss=xr.concat(l_rocss,dim='forecastMonth').assign_attrs({'Models': " | ".join(models['institution']+'-'+models['name'])})
    bs=xr.concat(l_bs,dim='forecastMonth').assign_attrs({'Models': " | ".join(models['institution']+'-'+models['name'])})

    # Save scores to netcdf
    rps.to_netcdf(f'{NEW_SCORES_HINDDIR}/{multi_hcst_bname}.{aggr}.rps.nc')
    rpss.to_netcdf(f'{NEW_SCORES_HINDDIR}/{multi_hcst_bname}.{aggr}.rpss.nc')
    bs.to_netcdf(f'{NEW_SCORES_HINDDIR}/{multi_hcst_bname}.{aggr}.bs.nc')
    roc.to_netcdf(f'{NEW_SCORES_HINDDIR}/{multi_hcst_bname}.{aggr}.roc.nc')
    rocss.to_netcdf(f'{NEW_SCORES_HINDDIR}/{multi_hcst_bname}.{aggr}.rocss.nc')

