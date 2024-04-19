import os
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc
import numpy as np
import cv2
from tqdm import tqdm
import json
import pandas as pd

from astropy.convolution import Box2DKernel, convolve

isPlot = False # Change to save map plots of upwelling

# Loading in a single netcdf4 file
def loadingInDataSingle(file, var):
    data = xr.open_dataset(file, engine = 'netcdf4')
    if var == "sst":
        return data.sst
    elif data == "chlor_a":
        return data.chlor_a
    else:
        raise NameError

# Producing a data map
def plotImage(data, cMin, cMax):
  quadmesh = xr.plot.pcolormesh(data, extend='both')
  quadmesh.set_clim(vmin=cMin, vmax=cMax)


filenameList = os.listdir("Data/SST/")

data_for_each_file_list = []

for filename in tqdm(filenameList):

    filepath_sst = "Data/SST/" + filename
    filepath_chla = "Data/Chla/" + filename[:-14] + "CHL.chlor_a.4km.nc"

    time_range = filename[11:28]

    sst_data_uncropped = loadingInDataSingle(filepath_sst, "sst")
    chla_data_uncropped = loadingInDataSingle(filepath_chla, "chlor_a")

    # Crop
    cropped_sst_data = sst_data_uncropped.where((sst_data_uncropped.lat > 33) & (sst_data_uncropped.lat < 42) & (sst_data_uncropped.lon < -117) & (sst_data_uncropped.lon > -127), drop = True)
    cropped_chla_data = chla_data_uncropped.where((chla_data_uncropped.lat > 33) & (chla_data_uncropped.lat < 42) & (chla_data_uncropped.lon < -117) & (chla_data_uncropped.lon > -127), drop = True)

    coastal_cropped_sst_data = cropped_sst_data.where(
      ((cropped_sst_data.lat > 38) & (cropped_sst_data.lat < 42) & (cropped_sst_data.lon > -126) & (cropped_sst_data.lon < -123))
    | ((cropped_sst_data.lat > 36) & (cropped_sst_data.lat < 38) & (cropped_sst_data.lon > -125) & (cropped_sst_data.lon < -121))
    | ((cropped_sst_data.lat > 35) & (cropped_sst_data.lat < 36) & (cropped_sst_data.lon > -124) & (cropped_sst_data.lon < -120))
    | ((cropped_sst_data.lat > 33) & (cropped_sst_data.lat < 35) & (cropped_sst_data.lon > -123) & (cropped_sst_data.lon < -116))
    )

    coastal_cropped_chla_data = cropped_chla_data.where(
      ((cropped_chla_data.lat > 38) & (cropped_chla_data.lat < 42) & (cropped_chla_data.lon > -126) & (cropped_chla_data.lon < -123))
    | ((cropped_chla_data.lat > 36) & (cropped_chla_data.lat < 38) & (cropped_chla_data.lon > -125) & (cropped_chla_data.lon < -121))
    | ((cropped_chla_data.lat > 35) & (cropped_chla_data.lat < 36) & (cropped_chla_data.lon > -124) & (cropped_chla_data.lon < -120))
    | ((cropped_chla_data.lat > 33) & (cropped_chla_data.lat < 35) & (cropped_chla_data.lon > -123) & (cropped_chla_data.lon < -116))
    )

    #Naive threshold
    sst_percentile = 0.25
    threshold = (coastal_cropped_sst_data.max() - coastal_cropped_sst_data.min())*sst_percentile + coastal_cropped_sst_data.min()

    thresholded_sst_data = coastal_cropped_sst_data.where(coastal_cropped_sst_data < 13) #Using Absolute threshold

    #TPI SST
    sst_data_tpi = xr.DataArray(
        data=cropped_sst_data.values - convolve(cropped_sst_data.values, Box2DKernel(25), boundary = 'extend', preserve_nan = True),
        dims=cropped_sst_data.dims,
        coords=cropped_sst_data.coords,
        attrs=cropped_sst_data.attrs
    )

    thresholded_sst_data_tpi = coastal_cropped_sst_data.where((sst_data_tpi < -1) & (sst_data_tpi > -5))

    #Chla filtering
    thresholded_chla_data_naive = cropped_chla_data.where(~np.isnan(thresholded_sst_data))
    thresholded_chla_data_tpi = cropped_chla_data.where(~np.isnan(thresholded_sst_data_tpi))
    
    #Plotting
    if isPlot:
    
      thresholded_sst_data_for_plot = thresholded_sst_data
      
      # SST Map
      plotImage(cropped_sst_data, 10, 22)
      plt.savefig('plots/sst/sst_{}.png'.format(time_range), bbox_inches='tight')
      plt.close()

      # Naive Threshold Map    
      fig = plt.figure()
      ax = fig.add_subplot(111)
      xr.plot.pcolormesh(thresholded_sst_data.where(np.isnan(thresholded_sst_data),100, 0), zorder=2, add_colorbar=False, cmap="inferno")
      quadmesh = xr.plot.pcolormesh(cropped_sst_data, zorder=1, extend='both')
      quadmesh.set_clim(vmin=10, vmax=22)
      plt.savefig('plots/thresh_sst/thresholded_sst_{}.png'.format(time_range), bbox_inches='tight')
      plt.close()

      # TPI Map
      plotImage(sst_data_tpi, -5, 5)
      plt.savefig('plots/sst_tpi/sst_tpi_{}.png'.format(time_range), bbox_inches='tight')
      plt.close()

      # Thresholded TPI Map
      fig = plt.figure()
      ax = fig.add_subplot(111)
      xr.plot.pcolormesh(thresholded_sst_data_tpi.where(np.isnan(thresholded_sst_data_tpi),100, 0), zorder=2, add_colorbar=False, cmap="inferno")
      quadmesh = xr.plot.pcolormesh(cropped_sst_data, zorder=1, extend='both')
      quadmesh.set_clim(vmin=10, vmax=22)
      plt.savefig('plots/thresh_sst_tpi/thresholded_sst_tpi_{}.png'.format(time_range), bbox_inches='tight')
      plt.close()



    ## Data Computation

    num_pixels_upwelling_naive = np.count_nonzero(~np.isnan(thresholded_sst_data)) 
    area_one_pixel = 4.64**2
    area_upwelling_naive = num_pixels_upwelling_naive * area_one_pixel
    num_pixels_upwelling_tpi = np.count_nonzero(~np.isnan(thresholded_sst_data_tpi))
    area_upwelling_tpi = num_pixels_upwelling_tpi * area_one_pixel

    weights = np.cos(np.deg2rad(cropped_sst_data.lat)) # Correction for averaging for latitude area difference
    weights.name = "weights"

    sst_mean_overall = cropped_sst_data.weighted(weights).mean(("lon","lat")).item() 
    sst_mean_coast = coastal_cropped_sst_data.weighted(weights).mean(("lon","lat")).item() 
    sst_mean_upwelling_naive = thresholded_sst_data.weighted(weights).mean(("lon","lat")).item()  
    sst_mean_upwelling_tpi = thresholded_sst_data_tpi.weighted(weights).mean(("lon","lat")).item() 

    chl_mean_overall = cropped_chla_data.weighted(weights).mean(("lon","lat")).item()  
    chl_mean_coast = coastal_cropped_chla_data.weighted(weights).mean(("lon","lat")).item() 
    chl_mean_upwelling_naive = thresholded_chla_data_naive.weighted(weights).mean(("lon","lat")).item()  
    chl_mean_upwelling_tpi = thresholded_chla_data_tpi.weighted(weights).mean(("lon","lat")).item() 
    

    ## Data Compilation

    file_data_dict ={}
    
    file_data_dict["time_range"] = time_range
    file_data_dict["year"] = time_range[0:4]
    file_data_dict["area_naive"] = area_upwelling_naive
    file_data_dict["area_tpi"] = area_upwelling_tpi
    file_data_dict["sst_mean_overall"] = sst_mean_overall
    file_data_dict["sst_mean_coast"] = sst_mean_coast
    file_data_dict["sst_mean_upwelling_naive"] = sst_mean_upwelling_naive
    file_data_dict["sst_mean_upwelling_tpi"] = sst_mean_upwelling_tpi
    file_data_dict["chl_mean_overall"] = chl_mean_overall
    file_data_dict["chl_mean_coast"] = chl_mean_coast
    file_data_dict["chl_mean_upwelling_naive"] = chl_mean_upwelling_naive
    file_data_dict["chl_mean_upwelling_tpi"] = chl_mean_upwelling_tpi
    


    data_for_each_file_list.append(file_data_dict)

json_object = json.dumps(data_for_each_file_list)

with open("results.json", "w") as outfile:
    outfile.write(json_object)

results = pd.read_json("results.json")

results.to_csv('results.csv')
