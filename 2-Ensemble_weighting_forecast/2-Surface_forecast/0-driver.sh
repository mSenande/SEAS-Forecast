#!/bin/bash -l

year=$1
startmonth=$2
aggr=$3
fcmonth=$4

while IFS="," read -r institution name short_institution short_name
do
    echo "Model: $institution $name"
    python -u 1-Download_data.py "$institution" "$name" $year $startmonth
    python -u 2a-Verification_PCs_forecast.py "$aggr" $startmonth
    python -u 2b-Verification_Postprocess.py "$institution" "$name" $startmonth
    python -u 3-Forecast_plots.py "$institution" "$name" $year $startmonth "$aggr" $fcmonth "MedCOF" "t2m"
    python -u 3-Forecast_plots.py "$institution" "$name" $year $startmonth "$aggr" $fcmonth "MedCOF" "tprate"
done < <(tail -n +2 models_running.csv)
python -u 4-Forecast_allmodels_plots.py $year $startmonth "$aggr" $fcmonth "MedCOF" "t2m"
python -u 4-Forecast_allmodels_plots.py $year $startmonth "$aggr" $fcmonth "MedCOF" "tprate"
python -u 5-Multimodel_Verification.py $startmonth
python -u 6-Multimodel_plots.py $year $startmonth "$aggr" $fcmonth "MedCOF" "t2m"
python -u 6-Multimodel_plots.py $year $startmonth "$aggr" $fcmonth "MedCOF" "tprate"

