# Forecast proposed methodology of C3S seasonal forecasting systems

We present a methodology to improve seasonal forecasts, based on using information on climate variability patterns to weight the ensemble members of different seasonal forecasting systems.

A two-steps approach is used: (1) a first forecast of 4 climate variability patterns and (2) a second forecast of surface variables (temperatura and precipitation) in which the ensemble members of the different seasonal forecasting systems are weighted according to their similarity to the predicted variability patterns.

Different verification scores are computed in order to compare the skill of the proposed methodology with the skill of the non-processed seasonal forecasting systems. The calculation of the verification scores is based on the Copernicus Climate Change Service (C3S) [Seasonal Forecast Verification Tutorial](https://ecmwf-projects.github.io/copernicus-training-c3s/sf-verification.html).

IMPORTANT: These scripts use data that was downloaded/calculated in [Seasonal Verification](https://github.com/mSenande/SEAS-Verification) and  [Seasonal Ensemble Weighting](https://github.com/mSenande/SEAS-Ensemble_weighting).

## Index

* 1-Default_forecast: Set of scripts to produce plots of temperature and precipitation seasonal forecasts using C3S seasonal forecasting systems with no specific weighting postprocessing.
We use for all seasons the same four varability patterns, obtained as the four main Empirical Orthogonal Functions of the December-January-February (DJF) ERA5 500 hPa geopotential height climatology (North Atlantic Oscillation, East Atlantic, East Atlantic / Western Russia and Scandinavian Pattern).
* 2-Ensemble_weighting_forecast: Set of scripts to produce plots of temperature and precipitation seasonal forecasts using C3S seasonal forecasting systems with ensemble weighting postprocessing. A first forecast is needed for each considered variability pattern. We use for all seasons the same four varability patterns, obtained as the four main Empirical Orthogonal Functions of the December-January-February (DJF) ERA5 500 hPa geopotential height climatology (North Atlantic Oscillation, East Atlantic, East Atlantic / Western Russia and Scandinavian Pattern). Then all ensemble members of each C3S seasonal forecasting system are weighted according to the difference between their variability pattern values and the previously forecasted values.
* 3-Validation: We use ERA5 for computeing seasonal observed anomalies and percentiles to comnpare with forecasts.