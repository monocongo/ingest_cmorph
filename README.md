# ingest_cmorph
Python code to ingest binary CMORPH data to NetCDF

To ingest the data into a NetCDF file containing daily values use the `ingest_cmorph_daily.py` script.

Example usage:

`$ python -u ingest_cmorph_daily.py --cmorph_dir /data/cmorph/adjusted --out_file /data/cmorph/cmorph_adjusted_conus.nc --obs_type adjusted --conus_only`

To ingest the data into a NetCDF file containing monthly values use the `ingest_cmorph_monthly.py` script.

Example usage:

`$ python -u ingest_cmorph_monthly.py --cmorph_dir /data/cmorph/adjusted --out_file /data/cmorph/cmorph_adjusted_conus.nc --obs_type adjusted --conus_only --download_file`