#!/bin/bash -l

year=$1
startmonth=$2
aggr=$3
fcmonth=$4

while IFS="," read -r institution name short_institution short_name
do
    echo "Model: $institution $name"
    python -u 1-Download_data.py "$institution" "$name" $year $startmonth
    python -u 2-Compute_EOFs.py "$institution" "$name" $year $startmonth
    python -u 3-Verification.py "$institution" "$name" $startmonth
    python -u 4-Forecast_plots.py "$institution" "$name" $year $startmonth "$aggr" $fcmonth
done < <(tail -n +2 models_running.csv)

