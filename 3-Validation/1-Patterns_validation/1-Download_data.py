# %% [markdown]
# # 1. Download data

# This script is used to download monthly seasonal observations for variability patterns validation.
# 
# First we have to decide a valid year and month. 

# %%
print("1. Download data")  

import os
import sys
from dotenv import load_dotenv
import cdsapi
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
# If no variables were introduced, ask for them
else:
    # Valid year 
    endyear = input("Resultados para el año: ")

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

# %% [markdown]
# ## 1.1 Request data from the CDS using CDS API

# Choose variables, create directories and define CDS configuration.

# %%
print("1.1 Request data from the CDS using CDS API")  

# Here we save the configuration
config = dict(
    list_vars = 'geopotential',
    pressure_level = '500',
    endyear = endyear,
    hcstarty = 1993,
    hcendy = 2016,
    endmonth = endmonth,
)

# Directory selection
load_dotenv() # Data is saved in a path defined in file .env
HINDDIR = os.getenv('HIND_DIR')
FOREDIR = os.getenv('FORE_DIR')

# Directory creation
for directory in [HINDDIR , FOREDIR]:
    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

# CDS configuration
c = cdsapi.Client()

# %% [markdown]
# ## 1.2 Retrieve validation data

# We will download the selected months

# %%
print("1.2 Retrieve validation data")

# Which months do we have to download?
if aggr=='1m':
    months = [endmonth]
    years = [int(endyear)]
elif aggr=='3m':
    months = [endmonth-m if endmonth-m>=1 else endmonth-m+12 for m in reversed(range(3))]
    years = [int(endyear) if endmonth-m>=1 else int(endyear)-1 for m in reversed(range(3))]
elif aggr=='5m':
    months = [endmonth-m if endmonth-m>=1 else endmonth-m+12 for m in reversed(range(5))]
    years = [int(endyear) if endmonth-m>=1 else int(endyear)-1 for m in reversed(range(5))]

for m in range(len(months)):
    month = months[m]
    year = years[m]

    # Base name for validation
    val_bname = f'era5_valid{year}-{month:02d}_monthly'
    # File name for validation
    val_fname = f'{FOREDIR}/{val_bname}.grib'

    # Check if file exists
    if not os.path.exists(val_fname):
        # If it doesn't exist, download it
        c.retrieve(
             'reanalysis-era5-pressure-levels-monthly-means',
            {
                'format': 'grib',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': config['list_vars'],
                'pressure_level': config['pressure_level'],
                'year': '{}'.format(year),
                'month': '{:02d}'.format(month),
                'time': '00:00',
                'grid': '1/1',
                'area': [80, -90, 20, 60],
            },
            val_fname)
        

# %% [markdown]
# ## 1.3 Retrieve climaological data (ERA5)

# Here we will download all the ERA5 period (1940-2020) to obtain the EOFs.

# %%
print("1.3 Retrieve climaological data (ERA5)")

# File name for climatology
clim_fname = f'{HINDDIR}/era5_monthly.grib'

# Check if file exists
if not os.path.exists(clim_fname):
    # If it doesn't exist, download it
    c.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': config['list_vars'],
            'pressure_level': config['pressure_level'],
            'year': ['{}'.format(yy) for yy in range(1940,2021)],
            'month': ['{}'.format(mm) for mm in range(1,13)],
            'time': '00:00',
            'grid': '1/1',
            'area': [80, -90, 20, 60],
            'format': 'grib',
        },
        clim_fname)

# File name for surface climatology
clim_fname2 = f'{HINDDIR}/era5_monthly_sf.grib'

# Check if file exists
if not os.path.exists(clim_fname2):
    # If it doesn't exist, download it
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': ['2m_temperature', 'total_precipitation', 'mean_sea_level_pressure'],
            'year': ['{}'.format(yy) for yy in range(1940,2021)],
            'month': ['{}'.format(mm) for mm in range(1,13)],
            'time': '00:00',
            'grid': '1/1',
            'area': [80, -90, 20, 60],
            'format': 'grib',
        },
        clim_fname2)
