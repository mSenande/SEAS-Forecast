# %% [markdown]
# # 2a. Principal Component Forecast (Verification)

# This script uses seasonal forecast systems to predict NAO, EA, EAWR and SCA patterns for the hindcast period.

# Information about the skill of each forecast system (RPSS) is used to select the best performing system for each start-month and forecasted-season pair.
# In this way, considering the 3-month aggregation, each forecasted season would have 3 variability pattern forecasts (one for each initialization / leadtime).

# First we have to decide a start month and a month aggregation. 

#%%
print("2a. Principal Component Forecast (Verification)")

import os
import sys
from dotenv import load_dotenv
import xarray as xr
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Read the file with model names
models = pd.read_csv('../../models.csv')
# Create a new column with full model name
models['full_name']=models['institution']+'-'+models['name']

# If variables are introduced from the command line, read them
if len(sys.argv) > 2:
    aggr = str(sys.argv[1])
    startmonth = int(sys.argv[2])
# If no variables were introduced, ask for them
else:
    # Monthly aggregation used
    aggr = input("Selecciona el tipo de agregación mensual [ 1m , 3m ]: ")

    # Which start month
    startmonth = int(input("Mes de inicialización (en número): "))

# Number of lead-times
lts=3
# List of initializations
if aggr=='1m':
    # Array with month names
    endmonth_name = np.array(['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    # Dictionary to link the name of each valid month name to the endmonth and forecastmonth (in number)
    # Example: {validmonth_name1: [endmonth1, forecastMonth1], validmonth_name2: [endmonth2, forecastMonth2], validmonth_name3: [endmonth3, forecastMonth3]}
    initialization = {endmonth_name[(startmonth+(l+1) if startmonth+(l+1)<12 else startmonth+(l+1)-12)-1]: 
                      [startmonth+(l+1) if startmonth+(l+1)<=12 else startmonth+(l+1)-12, l+2] for l in reversed(range(lts))}
elif aggr=='3m': 
    # Array with 3-month season names
    endmonth_name = np.array(['NDJ', 'DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND'])
    # Dictionary to link the name of each valid season name to the startmonth and forecastmonth (in number)
    # Example: {validseason_name1: [endmonth1, forecastMonth1], validseason_name2: [endmonth2, forecastMonth2], validseason_name3: [endmonth3, forecastMonth3]}
    initialization = {endmonth_name[(startmonth+(l+3) if startmonth+(l+3)<12 else startmonth+(l+3)-12)-1]: 
                      [startmonth+(l+3) if startmonth+(l+3)<=12 else startmonth+(l+3)-12, l+4] for l in reversed(range(lts))}

# Dictionary with some other information
config = dict(
    hcstarty = 1993,
    hcendy = 2016,
    aggr = aggr,
)

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
MODES_HINDDIR = os.getenv('MODES_HIND_DIR')  # Directory where hindcast variability patterns files are located
CSVDIR_VER = os.getenv('CSV_DIR_VER') # Directory where variability patterns verification info (RPSS) is located (only for verification)
POSTDIR_VER = os.getenv('POST_DIR_VER')
NEW_MODES_HINDDIR = POSTDIR_VER + '/modes' # Directory where variability patterns forecasts will be saved (only for verification)
# Check if the directory exists
if not os.path.exists(NEW_MODES_HINDDIR):
    # If it doesn't exist, create it
    try:
        os.makedirs(NEW_MODES_HINDDIR)
    except FileExistsError:
        pass

if os.path.exists(f'{NEW_MODES_HINDDIR}/PCs_best_forecasts_stmonth{startmonth:02d}.'+aggr+'.nc'):
    print(f'La verificación con agregacón {aggr} ya está hecha')
    sys.exit()
else:
    pass

l_hcpcs_eofs_leads_val=list()
# For each eof (eof1, eof2, eof3, eof4) 
for m in range(4):
    l_hcpcs_eofx_leads_val=list()
    # For each leadtime (lead1, lead2, lead3)
    for lead in range(1,4):
        # Read skill scores for this eof
        eofx_rpss = pd.read_csv(CSVDIR_VER+'/Score-card_rpss_'+aggr+'_eof'+str(m+1)+'.csv')
        # Rename model and lead columns
        eofx_rpss = eofx_rpss.rename(columns={eofx_rpss.columns[0]: "Model", eofx_rpss.columns[1]: "lead"})
        # Select forecastMonth and validmonth from initialization list
        validmonth = list(initialization)[-lead]
        fcmonth = initialization[validmonth][1]
        # Select desired month from scorecard table
        eofx_rpss_leadx = eofx_rpss[["Model","lead",validmonth]]
        # Sort models by forecast skill
        eofx_rpss_leadx = eofx_rpss_leadx[eofx_rpss_leadx['lead']=='lead'+str(lead)].sort_values(by=list(initialization)[-lead], ascending=False)
        # Select best model for this eof and leadtime
        origin = models[models['full_name']==eofx_rpss_leadx['Model'].values[0]]['short_institution'].values[0]
        system = models[models['full_name']==eofx_rpss_leadx['Model'].values[0]]['short_name'].values[0]
        # Reading HCST data from file
        hcpcs_fname = f'{MODES_HINDDIR}/{origin}_s{system}_stmonth{startmonth:02d}_hindcast'+str(config['hcstarty'])+'-'+str(config['hcendy'])+'_monthly.'+aggr+'.PCs.nc'
        # Read hindcast PCS data and select the desired eof and forecastMonth
        hcpcs_eofx_leadx = xr.open_dataset(hcpcs_fname).sel(mode=m,forecastMonth=fcmonth)
        # Quantify tercile thresholds
        low = hcpcs_eofx_leadx.quantile(1./3.,dim=['number','start_date'],skipna=True).drop('quantile')
        high = hcpcs_eofx_leadx.quantile(2./3.,dim=['number','start_date'],skipna=True).drop('quantile')
        # Quantify tercile averages
        low_value = hcpcs_eofx_leadx.where(hcpcs_eofx_leadx<low).mean(dim=['number','start_date']).pseudo_pcs
        med_value = hcpcs_eofx_leadx.where((hcpcs_eofx_leadx<=high) & (hcpcs_eofx_leadx>=low)).mean(dim=['number','start_date']).pseudo_pcs
        high_value = hcpcs_eofx_leadx.where(hcpcs_eofx_leadx>high).mean(dim=['number','start_date']).pseudo_pcs
        # Calculate the probability of each tercile (number of ensemble members above the threshold divided by the ensemble size), and assigning a representative value of each tercile
        hcpcs_eofx_leadx_low = (hcpcs_eofx_leadx.where(hcpcs_eofx_leadx<low).count(dim='number')/float(hcpcs_eofx_leadx.number.size)).assign_coords({'category':float(low_value)})
        hcpcs_eofx_leadx_med = (hcpcs_eofx_leadx.where((hcpcs_eofx_leadx<=high) & (hcpcs_eofx_leadx>=low)).count(dim='number')/float(hcpcs_eofx_leadx.number.size)).assign_coords({'category':float(med_value)})
        hcpcs_eofx_leadx_high = (hcpcs_eofx_leadx.where(hcpcs_eofx_leadx>high).count(dim='number')/float(hcpcs_eofx_leadx.number.size)).assign_coords({'category':float(high_value)})
        # Concatenate the three terciles
        hcpcs_eofx_leadx_ter = xr.concat([hcpcs_eofx_leadx_low,hcpcs_eofx_leadx_med,hcpcs_eofx_leadx_high],dim='category')
        # Find the most probable tercile and assign the corresponding value
        hcpcs_eofx_leadx_val = hcpcs_eofx_leadx_ter.idxmax(dim='category').drop('mode')
        # Include also information about the variability patterns forecast skill (RPSS)
        hcpcs_eofx_leadx_val = hcpcs_eofx_leadx_val.assign_coords({'rpss':eofx_rpss_leadx[list(initialization)[-lead]].values[0]})
        # Append results for each leadtime
        l_hcpcs_eofx_leads_val.append(hcpcs_eofx_leadx_val.assign_coords({'forecastMonth':fcmonth}))
    # Concatenate results for all leadtimes         
    hcpcs_eofx_leads_val = xr.concat(l_hcpcs_eofx_leads_val,dim='forecastMonth')        
    # Append results for each eof          
    l_hcpcs_eofs_leads_val.append(hcpcs_eofx_leads_val.assign_coords({'mode':m}))
# Concatenate results for all eofs                 
hcpcs_eofs_leads_val = xr.concat(l_hcpcs_eofs_leads_val,dim='mode')                    

# Save pcs best forecasts
hcpcs_eofs_leads_val.to_netcdf(f'{NEW_MODES_HINDDIR}/PCs_best_forecasts_stmonth{startmonth:02d}.'+aggr+'.nc')
