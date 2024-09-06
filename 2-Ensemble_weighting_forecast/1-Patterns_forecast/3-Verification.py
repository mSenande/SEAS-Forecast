# %% [markdown]
# # 3. Compute deterministic and probabilistic scores

# This script is used to compute different verification scores 
# for monthly seasonal forescasts of the four main climate variability modes.
# 
# The computed scores are: Spearman's rank correlation, area under Relative Operating Characteristic (ROC) curve, 
# Relative Operating Characteristic Skill Score (ROCSS), Ranked Probability Score (RPS), Ranked Probability Skill Score (RPSS) and Brier Score (BS).
#
# First we have to decide a forecast system (institution and system name) and a start month. 

#%%
print("3. Compute deterministic and probabilistic scores")

import os
import sys
from dotenv import load_dotenv
import xarray as xr
import numpy as np
import pandas as pd
import xskillscore as xs
import warnings
warnings.filterwarnings('ignore')

# Read the file with model names
models = pd.read_csv('../../models.csv')

# If variables are introduced from the command line, read them
if len(sys.argv) > 2:
    institution = str(sys.argv[1]).replace('"', '')
    name = str(sys.argv[2]).replace('"', '')
    startmonth = int(sys.argv[3])
# If no variables were introduced, ask for them
else:
    # Which model institution
    institutions = [inst for inst in models.institution.unique()]
    institution = input(f"Usar modelo del siguiente organismo {institutions}: ")

    # Which model system
    names = [name for name in models[models["institution"]==institution]["name"]]
    name = input(f"Sistema del modelo {names}: ")

    # Which start month
    startmonth = int(input("Mes de inicialización (en número): "))

# Save the simplier model and system name
model = str(models[(models["institution"]==institution) & (models["name"]==name)]["short_institution"].values[0])
system = str(models[(models["institution"]==institution) & (models["name"]==name)]["short_name"].values[0])

# Here we save the configuration
config = dict(
    list_vars = 'geopotential',
    pressure_level = '500',
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

# Base name for hindcast
hcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)
# File name for hindcast
hcpcs_fname = f'{MODES_HINDDIR}/{hcst_bname}.1m.PCs.nc'
hcpcs_3m_fname = f'{MODES_HINDDIR}/{hcst_bname}.3m.PCs.nc'
# Base name for observations
obs_bname = 'era5_monthly_stmonth{start_month:02d}_{hcstarty}-{hcendy}'.format(**config)
# File name for observations
obpcs_fname = f'{MODES_HINDDIR}/{obs_bname}.1m.PCs.nc'
obpcs_3m_fname = f'{MODES_HINDDIR}/{obs_bname}.3m.PCs.nc'

# Check if files exist
if not os.path.exists(obpcs_fname) & os.path.exists(obpcs_3m_fname):
    print('No se calcularon aún las PCs de ERA5')
    sys.exit()
elif not os.path.exists(hcpcs_fname) & os.path.exists(hcpcs_3m_fname):
    print('No se calcularon aún las PCs de este modelo y sistema')
    sys.exit()


aggr_list = []
# Check if verification is already done
for aggr in ['1m','3m']:
    corr_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.corr.nc'
    corr_pval_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.corr_pval.nc'
    rps_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.rps.nc'
    rpss_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.rpss.nc'
    bs_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.bs.nc'
    roc_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.roc.nc'
    rocss_fname = f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.rocss.nc'
    if os.path.exists(corr_fname) & os.path.exists(corr_pval_fname) & os.path.exists(rps_fname) & os.path.exists(rpss_fname) & os.path.exists(bs_fname) & os.path.exists(roc_fname) & os.path.exists(rocss_fname):
        print(f'La verificación con agregacón {aggr} ya está hecha')
    else:
        aggr_list += [aggr] # We write down which aggregation is not done
# If everything is done, we finish
if not aggr_list:
    print('Verificación lista')
    sys.exit()

# %% [markdown]
# ## 3.1 Probabilities for tercile categories

# Here we get the probabilities for tercile categories of the hindcast data, 
# by counting the number of ensemble members found in each tercile.

# %% 
print("3.1 Probabilities for tercile categories")

# Reading HCST data from file
hcpcs = xr.open_dataset(hcpcs_fname)
hcpcs_3m = xr.open_dataset(hcpcs_3m_fname)

# Reading OBS data from file
obpcs = xr.open_dataset(obpcs_fname)
obpcs_3m = xr.open_dataset(obpcs_3m_fname)

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

# Calculate probabilities for tercile categories by counting members within each category
quantiles = [1/3., 2/3.]
numcategories = len(quantiles)+1
# For each aggregation
for aggr in aggr_list:
    if aggr=='1m':
        h = hcpcs
    elif aggr=='3m':
        h = hcpcs_3m
    else:
        raise BaseException(f'Unknown aggregation {aggr}')

    l_probs_hcst=list()
    # For each quantile
    for icat in range(numcategories):
        # Get the lower and higher threshold
        h_lo,h_hi = get_thresh(icat, quantiles, h)
        # Count the number of member between the threshold
        probh = np.logical_and(h>h_lo, h<=h_hi).sum('number')/float(h.number.size)
        # Instead of using the coordinate 'quantile' coming from the hindcast xr.Dataset
        # we will create a new coordinate called 'category'
        if 'quantile' in list(probh.coords):
            probh = probh.drop('quantile')
        l_probs_hcst.append(probh.assign_coords({'category':icat}))

    # Concatenating tercile probs categories
    if aggr=='1m':
        probs_1m = xr.concat(l_probs_hcst,dim='category')                    
    elif aggr=='3m':
        probs_3m = xr.concat(l_probs_hcst,dim='category')                    

# %% [markdown]
# ## 3.2 Compute deterministic scores

# Here we calculate the Spearman's rank correlation and thei p-values. 
# 
# This score is based on the ensemble mean, not on the probabilities for each tercile.

# %% 
print("3.2 Compute deterministic scores")

# Loop over aggregations
for aggr in aggr_list:
    if aggr=='1m':
        o = obpcs
        h = hcpcs
    elif aggr=='3m':
        o = obpcs_3m
        h = hcpcs_3m
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
        # Compute ensemble mean (if data is an ensemble)
        thishcst_em = thishcst if not is_fullensemble else thishcst.mean('number')
        # Calculate Spearman's rank correlation
        l_corr.append( xs.spearman_r(thishcst_em, thisobs, dim='valid_time') )
        # Calculate p-value
        l_corr_pval.append ( xs.spearman_r_p_value(thishcst_em, thisobs, dim='valid_time') )

    # Concatenating (by fcmonth) correlation
    corr = xr.concat(l_corr,dim='forecastMonth')                    
    corr_pval = xr.concat(l_corr_pval,dim='forecastMonth')

    # Saving to netCDF file correlation   
    corr.to_netcdf(f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.corr.nc')
    corr_pval.to_netcdf(f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.corr_pval.nc')

# %% [markdown]
# ## 3.3 Compute probabilistic scores for tercile categories

# Here we calculate the probabilistic scores: area under Relative Operating Characteristic (ROC) curve, 
# Relative Operating Characteristic Skill Score (ROCSS), Ranked Probability Score (RPS), Ranked Probability Skill Score (RPSS) and Brier Score (BS). 

# %% 
print("3.3 Compute probabilistic scores for tercile categories")

# Loop over aggregations
for aggr in aggr_list:
    if aggr=='1m':
        o = obpcs
        probs_hcst = probs_1m
    elif aggr=='3m':
        o = obpcs_3m
        probs_hcst = probs_3m
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
            if 'quantile' in list(probo.coords):
                probo=probo.drop('quantile')
            l_probs_obs.append(probo.assign_coords({'category':icat}))
            # Count the number of months between the threshold (1 or 0)
            probc = np.logical_and(thiso>o_lo, thiso<=o_hi).sum('valid_time')/float(thiso.valid_time.size)        
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
    roc=xr.concat(l_roc,dim='forecastMonth')
    rps=xr.concat(l_rps,dim='forecastMonth')
    rpss=xr.concat(l_rpss,dim='forecastMonth')
    rocss=xr.concat(l_rocss,dim='forecastMonth')
    bs=xr.concat(l_bs,dim='forecastMonth')

    # Save scores to netcdf
    rps.to_netcdf(f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.rps.nc')
    rpss.to_netcdf(f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.rpss.nc')
    bs.to_netcdf(f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.bs.nc')
    roc.to_netcdf(f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.roc.nc')
    rocss.to_netcdf(f'{SCORE_HINDDIR}/{hcst_bname}.{aggr}.rocss.nc')

