# %% [markdown]
# # 1. Download data

# This script is used to download monthly seasonal data for variability patterns forescasts.
# 
# We will download data with lead times between 1 and 6 months.
# First we have to decide a forecast system (institution and system name) and a start month. 

# %%
print("1. Download data")  

import os
import sys
from dotenv import load_dotenv
import cdsapi
import pandas as pd
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

# Save the simplier model and system name
model = str(models[(models["institution"]==institution) & (models["name"]==name)]["short_institution"].values[0])
system = str(models[(models["institution"]==institution) & (models["name"]==name)]["short_name"].values[0])

# %% [markdown]
# ## 1.1 Request data from the CDS using CDS API

# Choose variables, create directories and define CDS configuration.

# %%
print("1.1 Request data from the CDS using CDS API")  

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
# ## 1.2 Retrieve forecast data

# We will download the selected start month and all the corresponding forecast months (from 1 to 6)
# for the selected year.

# %%
print("1.2 Retrieve forecast data")

# Base name for forecast
fcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_forecast{fcy}_monthly'.format(**config)
# File name for forecast
fcst_fname = f'{FOREDIR}/{fcst_bname}.grib'

# Check if file exists
if not os.path.exists(fcst_fname):
    # If it doesn't exist, download it
    c.retrieve(
        'seasonal-monthly-pressure-levels',
        {
            'format': 'grib',
            'originating_centre': config['origin'],
            'system': config['system'],
            'variable': config['list_vars'],
            'pressure_level': config['pressure_level'],
            'product_type': 'monthly_mean',
            'year': '{}'.format(config['fcy']),
            'month': '{:02d}'.format(config['start_month']),
            'leadtime_month': ['1', '2', '3','4', '5', '6'],
            'grid': '1/1',
            'area': [80, -90, 20, 60],
        },
        fcst_fname)

# %% [markdown]
# ## 1.3 Retrieve hindcast data

# We will download all the selected start month and the corresponding forecast months (from 1 to 6)
# for all the hindcast period (from 1993 to 2016).

# %%
print("1.3 Retrieve hindcast data")

# Base name for hindcast
hcst_bname = '{origin}_s{system}_stmonth{start_month:02d}_hindcast{hcstarty}-{hcendy}_monthly'.format(**config)
# File name for hindcast
hcst_fname = f'{HINDDIR}/{hcst_bname}.grib'

# Check if file exists
if not os.path.exists(hcst_fname):
    # If it doesn't exist, download it
    c.retrieve(
        'seasonal-monthly-pressure-levels',
        {
            'format': 'grib',
            'originating_centre': config['origin'],
            'system': config['system'],
            'variable': config['list_vars'],
            'pressure_level': config['pressure_level'],
            'product_type': 'monthly_mean',
            'year': ['{}'.format(yy) for yy in range(config['hcstarty'],config['hcendy']+1)],
            'month': '{:02d}'.format(config['start_month']),
            'leadtime_month': ['1', '2', '3','4', '5', '6'],
            'grid': '1/1',
            'area': [80, -90, 20, 60],
        },
        hcst_fname)


# %% [markdown]
# ## 1.4 Retrieve observational data (ERA5)

# Here we will download the same months as for the hindcast data.

# %%
print("1.4 Retrieve observational data (ERA5)")

# File name for observations
obs_bname = 'era5_monthly_stmonth{start_month:02d}_{hcstarty}-{hcendy}'.format(**config)
# File name for hindcast
obs_fname = f'{HINDDIR}/{obs_bname}.grib'

# Check if file exists
if not os.path.exists(obs_fname):
    # If it doesn't exist, download it
    c.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': config['list_vars'],
            'pressure_level': config['pressure_level'],
            # NOTE from observations we need to go one year beyond so we have available all the right valid dates
            # e.g. Nov.2016 start date forecast goes up to April 2017             
            'year': ['{}'.format(yy) for yy in range(config['hcstarty'],config['hcendy']+2)],
            'month': ['{:02d}'.format((config['start_month']+leadm)%12) if config['start_month']+leadm!=12 else '12' for leadm in range(6)],
            'time': '00:00',
            'grid': '1/1',
            'area': [80, -90, 20, 60],
            'format': 'grib',
        },
        obs_fname)

# %% [markdown]
# ## 1.5 Retrieve climaological data (ERA5)

# Here we will download all the ERA5 period (1940-2020) to obtain the EOFs.

# %%
print("1.5 Retrieve climaological data (ERA5)")

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
