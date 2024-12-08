# prepare the netcdf files starting from the Sentinel-2 raw data and the auxiliary one (e.g. dh/dt, DEMs)
export S2_ALPS_YEAR="inv"
python main_build_rasters.py

# process the data for training (i.e. patchify, cross-validation splits and training patches statistics)
python main_prep_data_train.py

# prepare the netcdf files for 2023
export S2_ALPS_YEAR="2023"
python main_build_rasters.py
