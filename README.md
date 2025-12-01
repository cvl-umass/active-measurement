## Overview
We provide data and codes for our "fixed checkpoint" approach. The codes are runnable with basic python libraries (to install, run `pip install numpy pandas click`).

## Repository structure
```
├── data/
│   ├── DSC5214_data.csv  # Annotation and detection on the sky image
│   ├── DSC5295_data2.csv  # Annotation and detection on the reeds image
│   ├── roost_counts/  # Annotation and detection for the radar station data
│
├── src/
│   ├── estimator/  # Implementation of active measurement
│   ├── model/  # Classes for the measurement tasks
│   ├── run/  # Running codes for the measurement tasks
```

## Running commands

Under the directory `src/run/` we provide codes for running the estimation pipeline. 

To perform estimation on the sky image with AIS, run
```commandline
python -m run.run_image --path ../data/DSC5214_data.csv --var_weighting uniform
```
To do the same task, but adding sampling without replacement, run
```commandline
python -m run.run_image --path ../data/DSC5214_data.csv --swor --var_weighting LURE
```

To further use the conditional variances instead of the simple variance, run

```commandline
python -m run.run_image --path ../data/DSC5214_data.csv --swor --cond --var_weighting LURE
```

To estimate on the reeds image, the path should be replaced with `../data/DSC5295_data2.csv`. 

The commands for the radar counting tasks are similar. For active measurement estimation on the KBUF station, run

```commandline
python -m run.run_station --station KBUF --var_weighting LURE
```

The other arguments `--swor` and `--cond` work in the same way as the image counting task.
