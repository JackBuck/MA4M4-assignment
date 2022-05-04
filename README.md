# MA4M4 assignment
This python implementation can be used to reproduce the figures from my essay submitted for MA4M4, titled _"Using asymptotic surprise to discover communities in a sea surface temperature network"_.
The file `main.py` is the entry point, while all 'content' can be found in the `ma4m4` package. The file `environment.yml` can be used to reproduce the conda the virtual environment:
```
conda create -n ma4m4-jb --file environment.yaml
```
Note that `ma4m4` has not been properly packaged as a python package, so is not installed. Instead, `main.py` must be run from the project root directory, so that `ma4m4` appears on the python path.

## Summary of libraries used
The signal processing to generate the anomaly series and correlations is done using `numpy` and `scipy`.
Graphs are represented using `networkx` and `cdlib` is used to perform all community detection.
Plots are generated using `matplotlib` and `cartopy`.

## Data
Data can be downloaded in NetCDF format from https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html . The file should be saved as `HadISST_sst.nc` in `data/01_raw/`.

