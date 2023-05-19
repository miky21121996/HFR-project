import argparse
from datetime import date
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import xarray
from netCDF4 import Dataset
from core import Get_List_Of_Files, Get_String_Time_Resolution, getSourceAntennas, Get_Closest_Hfr_Time_Range_Index, find_nearest, Get_Max_Min_Interpolated_Model, Get_Max_Min_Bias, Get_Max_Min_Rmsd, seaoverland, interp_mod_to_obs, interp_hfr_mask_to_mod_mask, interp_obs_to_mod, wind_direction, append_value
from core_plot import plot_hfr_wind_field, plot_model_wind_field, plot_bias, plot_rmsd, plot_mod_obs_ts_comparison, scatterPlot, plot_mod_obs_ts_comparison, plot_mod_obs_ts_comparison_1, plot_interpolated_hfr_wind_field, plot_windrose, TaylorDiagram, srl
import numpy as np
import numpy.ma as ma
import csv
from windrose import WindroseAxes
import zapata as zint
import skill_metrics as sm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot Statistics')
    parser.add_argument('plot_date_in', type=str,
                        help='start date from which to extract')
    parser.add_argument('plot_date_fin', type=str,
                        help='end date until which to extract')
    parser.add_argument('path_to_hfr_folder', type=str,
                        help='path to high-frequency radar files')
    parser.add_argument('time_res_to_average', type=str,
                        help='Time resolution to average data')
    parser.add_argument('grid_arr', type=str,
                        help='Type of model grid: regular or irregulat')
    parser.add_argument('interpolation', type=str,
                        help='Type of interpolation: MTO (from model grid to observation grid) or OTM (viceversa)')
    parser.add_argument('mesh_mask_arr', type=str,
                        help='Path to model mesh mask')
    parser.add_argument('u_combined_arr', type=str,
                        help='Path to the destaggered temporally concatenated U model file')
    parser.add_argument('v_combined_arr', type=str,
                        help='Path to the destaggered temporally concatenated V model file')
    parser.add_argument('name_exp_arr', type=str,
                        help='Name of the model experiment in the destaggered files')
    parser.add_argument('label_plot_arr', type=str,
                        help='Name of the model experiment in the plots')
    parser.add_argument('path_to_out_plot_folder_arr', type=str,
                        help='Path to output plot folder for each experiment')
    parser.add_argument('path_to_out_plot_folder_comparison', type=str,
                        help='Path to the output plot folder for comparison')
    args = parser.parse_args()
    return args

def main(args):
    date_in = args.plot_date_in
    date_fin = args.plot_date_fin
    path_to_hfr_folder = args.path_to_hfr_folder
    time_res_to_average = args.time_res_to_average
    grid_arr = args.grid_arr.split()
    interpolation = args.interpolation
    mesh_mask_arr = args.mesh_mask_arr.split()
    u_combined_arr = args.u_combined_arr.split()
    v_combined_arr = args.v_combined_arr.split()
    name_exp_arr = args.name_exp_arr.split()
    label_plot_arr = args.label_plot_arr.split()
    path_to_out_plot_folder_arr = args.path_to_out_plot_folder_arr.split()
    path_to_out_plot_folder_comparison = args.path_to_out_plot_folder_comparison
    
    
    start_date = date(int(date_in[0:4]),int(date_in[4:6]) , int(date_in[6:8]))
    end_date = date(int(date_fin[0:4]),int(date_fin[4:6]) , int(date_fin[6:8]))
    
    listOfFiles = Get_List_Of_Files(path_to_hfr_folder)

    if time_res_to_average[1]=='M':
        timerange = pd.date_range(start_date, end_date, freq=time_res_to_average[1]) - pd.DateOffset(days=15)
    if time_res_to_average[1]=='D':
        timerange = pd.date_range(start_date, end_date, freq=time_res_to_average[1]) - pd.DateOffset(hours=12)

    string_time_res = Get_String_Time_Resolution(start_date, end_date, time_res_to_average)

    if interpolation=="MTO":
        skip = (slice(None, None, 3), slice(None, None, 3))
        skip_coords = (slice(None,None,3))
        statistics={ }
        spatial_mean_hfr_ts={ }
        spatial_mean_model_ts={ }
    elif interpolation=="OTM":
        skip_model = (slice(None,None, 1), slice(None,None, 1))
        skip_coords_model = (slice(None,None,1))
        statistics_rev_interp={ }
        spatial_not_interp_mean_model_ts={ }
        spatial_mean_interp_hfr_ts={ }
    else:
        sys.exit(1)
        
    sdev = {}
    crmsd = {}
    ccoef = {}

    y_model_min = {}
    y_model_max = {}
    y_obs_min = {}
    y_obs_max = {}
    y_obs_min_averaged = {}
    y_obs_max_averaged = {}
    
    masked_u_interpolated_arr = {}
    masked_v_interpolated_arr = {}
    c_model_dir = {}

    possible_colors=['red', 'blue', 'black','green','purple','orange','brown','pink','grey','olive']
    possible_markers=np.array(["o","^","s","P","*","D"])

    for exp in range(len(name_exp_arr)):
        
        y_model_min[exp] = {}
        y_model_max[exp] = {}
        masked_u_interpolated_arr[exp] = {}
        masked_v_interpolated_arr[exp] = {}
        c_model_dir[exp] = {}
        
        
        if interpolation=="MTO":
            hfr_names=[]
            len_not_nan_values=[]
            spatial_mean_model_ts[exp]={}
            spatial_mean_hfr_ts={}
            statistics[exp]={}
            mod_array=np.array([])
            obs_array=np.array([])
        elif interpolation=="OTM":
            hfr_names_rev=[]
            len_not_nan_values_rev=[]
            spatial_not_interp_mean_model_ts[exp]={}
            spatial_mean_interp_hfr_ts[exp]={}
            statistics_rev_interp[exp]={}
            mod_array_rev_interp=np.array([])
            obs_array_rev_interp=np.array([])
        else:
            sys.exit(1)
        
        os.makedirs(path_to_out_plot_folder_arr[exp], exist_ok=True)

        mesh_mask_ds = Dataset(mesh_mask_arr[exp])
        if 'x' in list(mesh_mask_ds.variables) or 'y' in list(mesh_mask_ds.variables):
            mesh_mask=xarray.Dataset(data_vars=dict(tmask=(['time_counter','z','y','x'], mesh_mask_ds['tmask']), nav_lon=(['y','x'], mesh_mask_ds['x']), nav_lat=(['y','x'], mesh_mask_ds['y']),nav_lev=(['z'], mesh_mask_ds['nav_lev']), umask=(['time_counter','nav_lev','y','x'], mesh_mask_ds['umask']),vmask=(['time_counter','nav_lev','y','x'], mesh_mask_ds['vmask']),glamt=(['time_counter','y','x'], mesh_mask_ds['glamt']),gphit=(['time_counter','y','x'], mesh_mask_ds['gphit']),glamu=(['time_counter','y','x'], mesh_mask_ds['glamu']),gphiu=(['time_counter','y','x'], mesh_mask_ds['gphiu']),glamv=(['time_counter','y','x'], mesh_mask_ds['glamv']),gphiv=(['time_counter','y','x'], mesh_mask_ds['gphiv']),glamf=(['time_counter','y','x'], mesh_mask_ds['glamf']),gphif=(['time_counter','y','x'], mesh_mask_ds['gphif'])))
        else:
            mesh_mask = xarray.open_dataset(xarray.backends.NetCDF4DataStore(mesh_mask_ds))

        t_mask = mesh_mask.tmask.values

        for hfr_counter,hfr_file in enumerate(listOfFiles):
            print('loading ' + hfr_file + ' ...')
            ds = xarray.open_dataset(hfr_file)
            info = getSourceAntennas(ds)
            if 'id' not in list(ds.attrs.keys()):
                head, tail = os.path.split(hfr_file)
                splitted_hf_name=tail.split(".")
                ds.attrs['id']=splitted_hf_name[0]

            y_obs_min[ds.id] = {}
            y_obs_max[ds.id] = {}
            y_obs_min_averaged[ds.id] = {}
            y_obs_max_averaged[ds.id] = {}
                
            if interpolation=="MTO":
                spatial_mean_model_ts[exp][ds.id]=[]
                spatial_mean_hfr_ts[ds.id]=[]
            elif interpolation=="OTM":
                spatial_not_interp_mean_model_ts[exp][ds.id]=[]
                spatial_mean_interp_hfr_ts[exp][ds.id]=[]           

            ds_1=ds[['QCflag','EWCT','NSCT','LATITUDE','LONGITUDE']]
            ds_restricted=ds_1[['EWCT','NSCT','LATITUDE','LONGITUDE']].where((ds.QCflag==0) | (ds.QCflag == 1) | (ds.QCflag==2),drop=True)
            idx1_restricted,idx2_restricted,closerval1_restricted,closerval2_restricted = Get_Closest_Hfr_Time_Range_Index('1D',date_in,date_fin,ds_restricted)
            max_hfr_original = np.nanmax((ds_restricted['EWCT'][idx1_restricted:idx2_restricted+1,0].data ** 2 + ds_restricted['NSCT'][idx1_restricted:idx2_restricted+1,0].data ** 2) ** 0.5)
            min_hfr_original = np.nanmin((ds_restricted['EWCT'][idx1_restricted:idx2_restricted+1,0].data ** 2 + ds_restricted['NSCT'][idx1_restricted:idx2_restricted+1,0].data ** 2) ** 0.5)
            c_dir_cardinal=wind_direction(ds_restricted['EWCT'].data, ds_restricted['NSCT'].data)
            ds_restricted=ds_restricted.assign(CURRENT_DIRECTION=(['TIME', 'DEPTH', 'LATITUDE', 'LONGITUDE'],c_dir_cardinal))
            
            averaged_ds=ds_restricted.resample(TIME=time_res_to_average).mean(skipna=True)
            c_dir_cardinal=wind_direction(averaged_ds['EWCT'].data, averaged_ds['NSCT'].data)
            averaged_ds=averaged_ds.assign(CURRENT_DIRECTION=(['TIME', 'DEPTH', 'LATITUDE', 'LONGITUDE'],c_dir_cardinal))

            lat_hfr=averaged_ds.variables['LATITUDE'][:]
            lon_hfr=averaged_ds.variables['LONGITUDE'][:]

            x = averaged_ds['LONGITUDE'].data
            y = averaged_ds['LATITUDE'].data

            idx1,idx2,closerval1,closerval2 = Get_Closest_Hfr_Time_Range_Index(time_res_to_average,date_in,date_fin,averaged_ds)

            extent = [info['bbox'][0], info['bbox'][1]+0.2,info['bbox'][2], info['bbox'][3]+0.1]

            max_hfr = np.nanmax((averaged_ds['EWCT'][idx1:idx2+1,0].data ** 2 + averaged_ds['NSCT'][idx1:idx2+1,0].data ** 2) ** 0.5)
            min_hfr = np.nanmin((averaged_ds['EWCT'][idx1:idx2+1,0].data ** 2 + averaged_ds['NSCT'][idx1:idx2+1,0].data ** 2) ** 0.5)

            # read the model input
            ds_model_u = xarray.open_dataset(u_combined_arr[exp])
            if grid_arr[exp]=='irregular':
                ds_model_u = ds_model_u.vozocrtx[:,0,:,:]
                print(ds_model_u)
            ds_model_u.close()
            ds_model_v = xarray.open_dataset(v_combined_arr[exp])
            if grid_arr[exp]=='irregular':
                ds_model_v = ds_model_v.vomecrty[:,0,:,:]
            ds_model_v.close()

            # average in time the model input
            averaged_model_u = ds_model_u.resample(time_counter=time_res_to_average).mean(skipna=True)
            averaged_model_u.close()
            averaged_model_v = ds_model_v.resample(time_counter=time_res_to_average).mean(skipna=True)
            averaged_model_v.close()

            if grid_arr[exp]=='regular':
                x_model = averaged_model_u['nav_lon'].data[0,:]
                y_model = averaged_model_u['nav_lat'].data[:,0]

                u_model = averaged_model_u.variables['destaggered_u'][:,:,:].data
                v_model = averaged_model_v.variables['destaggered_v'][:,:,:].data
                speed_model = np.sqrt(u_model*u_model + v_model*v_model)
            elif grid_arr[exp]=='irregular':
                u_model = averaged_model_u.data[:,:,:]
                v_model = averaged_model_v.data[:,:,:]
                speed_model = np.sqrt(u_model*u_model + v_model*v_model)
            else:
                sys.exit(1)

            if grid_arr[exp]=='regular':
                # repeat the mask for all the days
                T_mask = np.repeat(t_mask[:, :, :, :], speed_model.shape[0], axis=0)
                t_mask_1 = T_mask[:, 0, :, :]
                u_model = ma.masked_array(u_model, mask=np.logical_not(t_mask_1))
                v_model = ma.masked_array(v_model, mask=np.logical_not(t_mask_1))
                speed_model = ma.masked_array(speed_model, mask=np.logical_not(t_mask_1))
            
                lon_model = averaged_model_u['nav_lon'].data[0,:]
                lat_model = averaged_model_u['nav_lat'].data[:,0]

                closer_min_mod_lon = find_nearest(lon_model, np.nanmin(ds['LONGITUDE'].data))
                closer_min_mod_lat = find_nearest(lat_model, np.nanmin(ds['LATITUDE'].data))
                closer_max_mod_lon = find_nearest(lon_model, np.nanmax(ds['LONGITUDE'].data))
                closer_max_mod_lat = find_nearest(lat_model, np.nanmax(ds['LATITUDE'].data))

                coord_min_lon_min_lat = [closer_min_mod_lon, closer_min_mod_lat]
                coord_idx_min_lon_min_lat = np.argwhere((averaged_model_u['nav_lon'].data==coord_min_lon_min_lat[0]) & 
                                        (averaged_model_u['nav_lat'].data==coord_min_lon_min_lat[1]))[0]
                coord_min_lon_max_lat = [closer_min_mod_lon, closer_max_mod_lat]
                coord_idx_min_lon_max_lat = np.argwhere((averaged_model_u['nav_lon'].data==coord_min_lon_max_lat[0]) & 
                                        (averaged_model_u['nav_lat'].data==coord_min_lon_max_lat[1]))[0]
                coord_max_lon_min_lat = [closer_max_mod_lon, closer_min_mod_lat]
                coord_idx_max_lon_min_lat = np.argwhere((averaged_model_u['nav_lon'].data==coord_max_lon_min_lat[0]) & 
                                        (averaged_model_u['nav_lat'].data==coord_max_lon_min_lat[1]))[0]
                coord_max_lon_max_lat = [closer_max_mod_lon, closer_max_mod_lat]
                coord_idx_max_lon_max_lat = np.argwhere((averaged_model_u['nav_lon'].data==coord_max_lon_max_lat[0]) & 
                                        (averaged_model_u['nav_lat'].data==coord_max_lon_max_lat[1]))[0]

                subset_averaged_model_u = averaged_model_u.isel(x=slice(coord_idx_min_lon_min_lat[1]-1,
								coord_idx_max_lon_min_lat[1]+1),
							y=slice(coord_idx_min_lon_min_lat[0]-1,
								coord_idx_min_lon_max_lat[0]+1))
                subset_averaged_model_v = averaged_model_v.isel(x=slice(coord_idx_min_lon_min_lat[1]-1,
								coord_idx_max_lon_min_lat[1]+1),
							y=slice(coord_idx_min_lon_min_lat[0]-1,
								coord_idx_min_lon_max_lat[0]+1))

                x_subset_model = subset_averaged_model_u['nav_lon'].data[0,:]
                y_subset_model = subset_averaged_model_u['nav_lat'].data[:,0]

                subset_u_model = subset_averaged_model_u.variables['destaggered_u'][:,:,:].data
                subset_v_model = subset_averaged_model_v.variables['destaggered_v'][:,:,:].data
                subset_speed_model = np.sqrt(subset_u_model * subset_u_model + subset_v_model * subset_v_model)

                subset_t_mask = t_mask_1[:, slice(coord_idx_min_lon_min_lat[0]-1, coord_idx_min_lon_max_lat[0]+1),slice(coord_idx_min_lon_min_lat[1]-1, coord_idx_max_lon_min_lat[1]+1)]
                subset_t_mask = np.logical_not(subset_t_mask)
                masked_subset_u_model = ma.masked_array(subset_u_model, mask=subset_t_mask)
                masked_subset_v_model = ma.masked_array(subset_v_model, mask=subset_t_mask)
                masked_subset_speed_model = ma.masked_array(subset_speed_model, mask=subset_t_mask)
            
                min_model_value,max_model_value = Get_Max_Min_Interpolated_Model(idx1,idx2,averaged_ds,masked_subset_speed_model,x_subset_model,y_subset_model,lon_hfr,lat_hfr)
                min_bias,max_bias = Get_Max_Min_Bias(idx1,idx2,averaged_ds,masked_subset_speed_model,x_subset_model,y_subset_model,lon_hfr,lat_hfr)
                min_rmsd,max_rmsd = Get_Max_Min_Rmsd(idx1,idx2,averaged_ds,masked_subset_speed_model,x_subset_model,y_subset_model,lon_hfr,lat_hfr)

            elif grid_arr[exp]=='irregular':
                update_u = averaged_model_u
                update_v = averaged_model_v
                variables3DU = {"vozocrtx":update_u}
                variables3DV = {"vomecrty":update_v}

                for time_counter,index in enumerate(range(idx1,idx2+1)):
                    U = averaged_ds['EWCT'][index,0].data
                    V = averaged_ds['NSCT'][index,0].data
                    speed_hfr = (U ** 2 + V ** 2) ** 0.5
                    mask_hfr=np.ma.masked_invalid(speed_hfr).mask

                    nav_lon, nav_lat = np.meshgrid(lon_hfr, lat_hfr)

                    grid=xarray.Dataset(data_vars=dict(nav_lon=(['y','x'], nav_lon),nav_lat=(['y','x'], nav_lat),mask=(['y','x'], mask_hfr)))                
                    w = zint.Ocean_Interpolator(name_exp_arr[exp],ds.id,grid,nloops=3)                
                    in_u = variables3DU["vozocrtx"][time_counter,:,:]
                    in_v = variables3DV["vomecrty"][time_counter,:,:]
                    targetU, targetV, destagU, destagV = w.interp_UV(in_u, in_v, method='linear')

                    masked_u_interpolated = ma.masked_array(targetU, mask=mask_hfr)
                    masked_v_interpolated = ma.masked_array(targetV, mask=mask_hfr)
                    speed_interpolated=np.sqrt(targetU * targetU + targetV * targetV)
                    masked_speed_interpolated= ma.masked_array(speed_interpolated, mask=mask_hfr)             

                    if time_counter == 0:
                        min_model_value=np.nanmin(masked_speed_interpolated.data)
                        max_model_value=np.nanmax(masked_speed_interpolated.data)
                        min_model_bias=np.nanmin(masked_speed_interpolated.data-speed_hfr.data)
                        max_model_bias=np.nanmax(masked_speed_interpolated.data-speed_hfr.data)
                        min_model_rmsd=np.nanmin(np.sqrt((masked_speed_interpolated.data-speed_hfr.data)**2))
                        max_model_rmsd=np.nanmax(np.sqrt((masked_speed_interpolated.data-speed_hfr.data)**2))
                    else:
                        min_interpolated_model = np.nanmin(masked_speed_interpolated.data)
                        min_model_value = min(min_model_value,min_interpolated_model)
                        max_interpolated_model = np.nanmax(masked_speed_interpolated.data)
                        max_model_value = max(max_model_value,max_interpolated_model)          
                        min_bias = np.nanmin(masked_speed_interpolated.data-speed_hfr.data)
                        min_model_bias = min(min_model_bias,min_bias)
                        max_bias = np.nanmax(masked_speed_interpolated.data-speed_hfr.data)
                        max_model_bias = max(max_model_bias,max_bias) 
                        min_rmsd = np.nanmin(np.sqrt((masked_speed_interpolated.data-speed_hfr.data)**2))
                        min_model_rmsd = min(min_model_rmsd,min_rmsd)
                        max_rmsd = np.nanmax(np.sqrt((masked_speed_interpolated.data-speed_hfr.data)**2))
                        max_model_rmsd = max(max_model_rmsd,max_rmsd)
            else:
                sys.exit(1)
                
            a=ds_restricted['CURRENT_DIRECTION'][idx1_restricted:idx2_restricted+1,0].data.ravel()
            b=((ds_restricted['EWCT'][idx1_restricted:idx2_restricted+1,0].data ** 2 + ds_restricted['NSCT'][idx1_restricted:idx2_restricted+1,0].data ** 2) ** 0.5).ravel()
            a = a[~np.isnan(b)]
            b = b[~np.isnan(b)]
            ax = WindroseAxes.from_ax()
            turbo = plt.get_cmap('turbo')
            ax.bar(a[~np.isnan(a)], b[~np.isnan(a)], normed=True, bins=np.linspace(
            min(min_hfr_original,min_model_value), max(max_hfr_original,max_model_value), 5), opening=0.8, edgecolor='white', cmap=turbo)
            y_obs_min[ds.id], y_obs_max[ds.id] = ax.get_ylim()
            
            a=averaged_ds['CURRENT_DIRECTION'][idx1_restricted:idx2_restricted+1,0].data.ravel()
            b=((averaged_ds['EWCT'][idx1_restricted:idx2_restricted+1,0].data ** 2 + averaged_ds['NSCT'][idx1_restricted:idx2_restricted+1,0].data ** 2) ** 0.5).ravel()
            a = a[~np.isnan(b)]
            b = b[~np.isnan(b)]
            ax = WindroseAxes.from_ax()
            turbo = plt.get_cmap('turbo')
            ax.bar(a[~np.isnan(a)], b[~np.isnan(a)], normed=True, bins=np.linspace(
            min(min_hfr,min_model_value), max(max_hfr,max_model_value), 5), opening=0.8, edgecolor='white', cmap=turbo)
            y_obs_min_averaged[ds.id], y_obs_max_averaged[ds.id] = ax.get_ylim()

            masked_u_interpolated_arr[exp][ds.id]=np.empty([len(range(idx1,idx2+1)),ds_restricted['EWCT'].data.shape[2],ds_restricted['EWCT'].data.shape[3]])
            masked_v_interpolated_arr[exp][ds.id]=np.empty([len(range(idx1,idx2+1)),ds_restricted['EWCT'].data.shape[2],ds_restricted['EWCT'].data.shape[3]])
            c_model_dir[exp][ds.id]=np.empty([len(range(idx1,idx2+1)),ds_restricted['EWCT'].data.shape[2],ds_restricted['EWCT'].data.shape[3]])

            for time_counter,index in enumerate(range(idx1,idx2+1)):
                date_str = string_time_res[time_counter]

                U = averaged_ds['EWCT'][index,0].data
                V = averaged_ds['NSCT'][index,0].data
                speed_hfr = (U ** 2 + V ** 2) ** 0.5

                spatial_mean_hfr_ts[ds.id].append(np.nanmean(speed_hfr))
            
                mask_hfr=np.ma.masked_invalid(speed_hfr).mask

                if grid_arr[exp]=='regular':
                    if interpolation=="MTO":
                        plot_hfr_wind_field(info, extent, min_hfr, min_model_value, max_hfr, max_model_value, x, y, speed_hfr, U, V, skip, skip_coords, date_str, ds, path_to_out_plot_folder_arr[exp])
                        subset_speed_model_instant = seaoverland(masked_subset_speed_model[time_counter],3)
                        subset_u_model_instant=seaoverland(masked_subset_u_model[time_counter],3)
                        subset_v_model_instant=seaoverland(masked_subset_v_model[time_counter],3)
                        masked_speed_interpolated, masked_u_interpolated, masked_v_interpolated, spatial_mean_model_ts_instant=interp_mod_to_obs(x_subset_model,y_subset_model,subset_speed_model_instant,subset_u_model_instant,subset_v_model_instant,lon_hfr,lat_hfr,mask_hfr)
                        c_model_dir[exp][ds.id][time_counter,:,:]=wind_direction(masked_u_interpolated.data,masked_v_interpolated.data)
                        masked_u_interpolated_arr[exp][ds.id][time_counter,:,:]=masked_u_interpolated.data
                        masked_v_interpolated_arr[exp][ds.id][time_counter,:,:]=masked_v_interpolated.data
                        spatial_mean_model_ts[exp][ds.id].append(spatial_mean_model_ts_instant)
                        print("spatial_mean_model_ts: ",spatial_mean_model_ts[exp][ds.id])
                        title_substring='interpolated model surface current'
                        name_file_substring='model_surface_current_velocity_'
                        plot_model_wind_field(info, extent, min_hfr, min_model_value, max_hfr, max_model_value, x, y, skip, skip_coords, masked_speed_interpolated, masked_u_interpolated, masked_v_interpolated, date_str, path_to_out_plot_folder_arr[exp],label_plot_arr[exp],title_substring,name_file_substring,ds,spatial_mean_model_ts_instant)

                        title_substring='surface current bias'
                        name_file_substring='surface_current_velocity_bias'
                        plot_bias(info, extent, x, y, min_bias, max_bias, masked_speed_interpolated, speed_hfr, date_str, path_to_out_plot_folder_arr[exp],label_plot_arr[exp],title_substring,name_file_substring,ds)

                        title_substring='surface current rmsd'
                        name_file_substring='surface_current_velocity_rmsd'
                        plot_rmsd(info, extent, x, y, min_rmsd, max_rmsd, masked_speed_interpolated, speed_hfr, date_str, path_to_out_plot_folder_arr[exp],label_plot_arr[exp],title_substring,name_file_substring,ds)
                    elif interpolation=="OTM":
                        masked_speed_hfr=ma.masked_array(speed_hfr,mask=mask_hfr)
                        masked_U=ma.masked_array(U,mask=mask_hfr)
                        masked_V=ma.masked_array(V,mask=mask_hfr)
                        sol_speed_hfr = seaoverland(masked_speed_hfr,3)
                        sol_u_hfr=seaoverland(masked_U,3)
                        sol_v_hfr=seaoverland(masked_V,3)
                        threshold=0.7
                        step_lon=lon_hfr[1]-lon_hfr[0]
                        step_lat=lat_hfr[1]-lat_hfr[0]
                        X=np.concatenate(([lon_hfr[0]-step_lon], lon_hfr, [lon_hfr[-1]+step_lon]))
                        Y=np.concatenate(([lat_hfr[0]-step_lat], lat_hfr, [lat_hfr[-1]+step_lat]))
                        mask_hfr_prova=np.pad(np.logical_not(mask_hfr), 1)
                        hfr_mask_interpolated=interp_hfr_mask_to_mod_mask(X,Y,np.logical_not(mask_hfr_prova),x_subset_model,y_subset_model,threshold)
                        masked_hfr_speed_interpolated, masked_hfr_u_interpolated, masked_hfr_v_interpolated, spatial_mean_hfr_ts_instant=interp_obs_to_mod(lon_hfr,lat_hfr,sol_speed_hfr,sol_u_hfr,sol_v_hfr,x_subset_model,y_subset_model,hfr_mask_interpolated)
                        spatial_mean_interp_hfr_ts[exp][ds.id].append(spatial_mean_hfr_ts_instant)
                        title_substring='interpolated hfr surface current'
                        name_file_substring='_interp_hfr_surface_current_velocity_'
                        plot_interpolated_hfr_wind_field(info, extent, min_hfr, min_model_value, max_hfr, max_model_value, x_subset_model, y_subset_model, skip_model, skip_coords_model, masked_hfr_speed_interpolated, masked_hfr_u_interpolated, masked_hfr_v_interpolated, date_str, path_to_out_plot_folder_arr[exp],title_substring,name_file_substring,ds)

                        title_substring='model surface current'
                        name_file_substring='_model_surface_current_velocity_'
                        plot_model_wind_field(info, extent, min_hfr, min_model_value, max_hfr, max_model_value, x_subset_model, y_subset_model, skip_model, skip_coords_model, masked_subset_speed_model[time_counter], masked_subset_u_model[time_counter], masked_subset_v_model[time_counter], date_str, path_to_out_plot_folder_arr[exp],label_plot_arr[exp],title_substring,name_file_substring,ds,masked_subset_speed_model[time_counter].mean())

                        title_substring='surface current bias (rev_interp)'
                        name_file_substring='_surface_current_velocity_bias_rev_interp'
                        plot_bias(info, extent, x_subset_model, y_subset_model, min_bias, max_bias, masked_subset_speed_model[time_counter], masked_hfr_speed_interpolated, date_str, path_to_out_plot_folder_arr[exp],label_plot_arr[exp],title_substring,name_file_substring,ds)

                        title_substring='surface current rmsd (rev_interp)'
                        name_file_substring='_surface_current_velocity_rmsd_rev_interp'
                        plot_rmsd(info, extent, x_subset_model, y_subset_model, min_rmsd, max_rmsd, masked_subset_speed_model[time_counter], masked_hfr_speed_interpolated, date_str, path_to_out_plot_folder_arr[exp],label_plot_arr[exp],title_substring,name_file_substring,ds)
                    else:
                        sys.exit(1)
                elif grid_arr[exp]=='irregular':
                    plot_hfr_wind_field(info, extent, min_hfr, min_model_value, max_hfr, max_model_value, x, y, speed_hfr, U, V, skip, skip_coords, date_str, ds, path_to_out_plot_folder_arr[exp])
                    in_u = variables3DU["vozocrtx"][time_counter,:,:]
                    in_v = variables3DV["vomecrty"][time_counter,:,:]
                    targetU, targetV, destagU, destagV = w.interp_UV(in_u, in_v, method='linear')
                    masked_u_interpolated = ma.masked_array(targetU, mask=mask_hfr)
                    masked_v_interpolated = ma.masked_array(targetV, mask=mask_hfr)
                    c_model_dir[exp][ds.id][time_counter,:,:]=wind_direction(masked_u_interpolated.data,masked_v_interpolated.data)
                    masked_u_interpolated_arr[exp][ds.id][time_counter,:,:]=masked_u_interpolated.data
                    masked_v_interpolated_arr[exp][ds.id][time_counter,:,:]=masked_v_interpolated.data
                    speed_interpolated=np.sqrt(targetU * targetU + targetV * targetV)
                    masked_speed_interpolated= ma.masked_array(speed_interpolated, mask=mask_hfr)
                    spatial_mean_model_ts_instant = np.nanmean(speed_interpolated)
                    spatial_mean_model_ts[exp][ds.id].append(spatial_mean_model_ts_instant)

                    title_substring='interpolated model surface current'
                    name_file_substring='model_surface_current_velocity_'
                    plot_model_wind_field(info, extent, min_hfr, min_model_value, max_hfr, max_model_value, x, y, skip, skip_coords, masked_speed_interpolated, masked_u_interpolated, masked_v_interpolated, date_str, path_to_out_plot_folder_arr[exp],label_plot_arr[exp],title_substring,name_file_substring,ds,spatial_mean_model_ts_instant)

                    title_substring='surface current bias'
                    name_file_substring='surface_current_velocity_bias'
                    plot_bias(info, extent, x, y, min_model_bias, max_model_bias, masked_speed_interpolated, speed_hfr, date_str, path_to_out_plot_folder_arr[exp],label_plot_arr[exp], title_substring,name_file_substring,ds)

                    title_substring='surface current rmsd'
                    name_file_substring='surface_current_velocity_rmsd'
                    plot_rmsd(info, extent, x, y, min_model_rmsd, max_model_rmsd, masked_speed_interpolated, speed_hfr, date_str, path_to_out_plot_folder_arr[exp],label_plot_arr[exp], title_substring,name_file_substring,ds)
                else:
                    sys.exit(1)

            if interpolation=="MTO":
                a=c_model_dir[exp][ds.id].ravel()
                b=((masked_u_interpolated_arr[exp][ds.id] ** 2 + masked_v_interpolated_arr[exp][ds.id] ** 2) ** 0.5).ravel()
                ax = WindroseAxes.from_ax()
                turbo = plt.get_cmap('turbo')
                ax.bar(a[~np.isnan(a)], b[~np.isnan(a)], normed=True, bins=np.linspace(
                min(min_model_value,min_hfr), max(max_model_value,max_hfr), 5), opening=0.8, edgecolor='white', cmap=turbo)
                y_model_min[exp][ds.id], y_model_max[exp][ds.id] = ax.get_ylim()                

            if grid_arr[exp]=="regular":
                if interpolation=="MTO":
                    title_substring='Spatial Surface Current Velocity Mean Comparison'
                    name_file_substring='_mod_obs_ts_comparison'
                    mean_vel_mod,mean_vel_obs=plot_mod_obs_ts_comparison(spatial_mean_hfr_ts[ds.id][:], spatial_mean_model_ts[exp][ds.id][:], time_res_to_average, ds, date_in, date_fin, path_to_out_plot_folder_arr[exp],timerange,label_plot_arr[exp],title_substring,name_file_substring)
                    tot_mean_stat=[mean_vel_mod,mean_vel_obs]

                    plotname = ds.id + '_' + date_in + '_' + date_fin + '_' + time_res_to_average +  '_qqPlot.png'
                    title = 'Spatial Mean Surface Current Velocity ' + ds.id + '\n Period: ' + date_in + ' - ' + date_fin
                    xlabel = 'Observation Current Velocity [m/s]'
                    ylabel = 'Model Current Velocity [m/s]'
                    splitted_name=ds.id.split("-")
                    hfr_names.append(splitted_name[1])
                    if timerange.shape[0] > 2:
                        statistics_array=scatterPlot(np.array(spatial_mean_model_ts[exp][ds.id]),np.array(spatial_mean_hfr_ts[ds.id]),path_to_out_plot_folder_arr[exp] + plotname,label_plot_arr[exp],1,len(spatial_mean_model_ts[exp][ds.id]),possible_markers[hfr_counter],splitted_name[1],possible_colors,string_time_res,title=title,xlabel=xlabel,ylabel=ylabel)
                        row_stat = tot_mean_stat + statistics_array
                        statistics[exp][ds.id] = row_stat
                    ciao=np.array(spatial_mean_hfr_ts[ds.id])
                    len_not_nan_values.append(len(ciao[~np.isnan(ciao)]))
                    mod_array = np.concatenate([mod_array, np.array(spatial_mean_model_ts[exp][ds.id])])
                    obs_array = np.concatenate([obs_array, np.array(spatial_mean_hfr_ts[ds.id])])
                if interpolation=="OTM":
                    
                    title_substring='Spatial Surface Current Velocity Mean Comparison\n (Rev Interp)'
                    name_file_substring='_mod_obs_ts_comparison_rev_interp'
                    mean_vel_no_interp_mod,mean_vel_interp_obs=plot_mod_obs_ts_comparison(spatial_mean_interp_hfr_ts[exp][ds.id], spatial_not_interp_mean_model_ts[exp][ds.id], time_res_to_average, ds, date_in, date_fin, path_to_out_plot_folder_arr[exp],timerange,label_plot_arr[exp],title_substring,name_file_substring)
                    tot_mean_stat_rev_interp=[mean_vel_no_interp_mod,mean_vel_interp_obs]
                    plotname = ds.id + '_' + date_in + '_' + date_fin + '_' + time_res_to_average +  '_qqPlot_rev_interp.png'
                    title = 'Spatial Mean Surface Current Velocity (Rev Interp)' + ds.id + '\n Period: ' + date_in + ' - ' + date_fin
                    xlabel = 'Observation Current Velocity [m/s]'
                    ylabel = 'Model Current Velocity [m/s]'
                    splitted_name=ds.id.split("-")
                    hfr_names_rev.append(splitted_name[1])
                    if timerange.shape[0] > 2:
                        statistics_array_rev_interp=scatterPlot(np.array(spatial_not_interp_mean_model_ts[exp][ds.id]),np.array(spatial_mean_interp_hfr_ts[exp][ds.id]),path_to_out_plot_folder_arr[exp] + plotname,label_plot_arr[exp],1,len(spatial_not_interp_mean_model_ts[exp][ds.id]),possible_markers[hfr_counter],splitted_name[1],possible_colors,string_time_res,title=title,xlabel=xlabel,ylabel=ylabel)
                        row_stat_rev_interp = tot_mean_stat_rev_interp + statistics_array_rev_interp
                        statistics_rev_interp[exp][ds.id] = row_stat_rev_interp
                    ciao=np.array(spatial_mean_interp_hfr_ts[exp][ds.id])
                    len_not_nan_values_rev.append(len(ciao[~np.isnan(ciao)]))

                    mod_array_rev_interp = np.concatenate([mod_array_rev_interp, np.array(spatial_not_interp_mean_model_ts[exp][ds.id])])
                    obs_array_rev_interp = np.concatenate([obs_array_rev_interp, np.array(spatial_mean_interp_hfr_ts[exp][ds.id])])
            
            elif grid_arr[exp]=="irregular":
                title_substring='Spatial Surface Current Velocity Mean Comparison'
                name_file_substring='_mod_obs_ts_comparison'
                mean_vel_mod,mean_vel_obs=plot_mod_obs_ts_comparison(spatial_mean_hfr_ts[ds.id], spatial_mean_model_ts[exp][ds.id], time_res_to_average, ds, date_in, date_fin, path_to_out_plot_folder_arr[exp],timerange,label_plot_arr[exp],title_substring,name_file_substring)
                tot_mean_stat=[mean_vel_mod,mean_vel_obs]

                plotname = ds.id + '_' + date_in + '_' + date_fin + '_' + time_res_to_average +  '_qqPlot.png'
                title = 'Spatial Mean Surface Current Velocity ' + ds.id + '\n Period: ' + date_in + ' - ' + date_fin
                xlabel = 'Observation Current Velocity [m/s]'
                ylabel = 'Model Current Velocity [m/s]'
                splitted_name=ds.id.split("-")
                hfr_names.append(splitted_name[1])
                if timerange.shape[0] > 2:
                    statistics_array=scatterPlot(np.array(spatial_mean_model_ts[exp][ds.id]),np.array(spatial_mean_hfr_ts[ds.id]),path_to_out_plot_folder_arr[exp] + plotname,label_plot_arr[exp],1,len(spatial_mean_model_ts[exp][ds.id]),possible_markers[hfr_counter],splitted_name[1],possible_colors,string_time_res,title=title,xlabel=xlabel,ylabel=ylabel)
                    row_stat = tot_mean_stat + statistics_array
                    statistics[exp][ds.id] = row_stat
                ciao=np.array(spatial_mean_hfr_ts[ds.id])
                len_not_nan_values.append(len(ciao[~np.isnan(ciao)]))
                mod_array = np.concatenate([mod_array, np.array(spatial_mean_model_ts[exp][ds.id])])
                obs_array = np.concatenate([obs_array, np.array(spatial_mean_hfr_ts[ds.id])])

            else:
                sys.exit(1)

        if interpolation=="MTO":
            
            tot_mean_mod=round(np.nanmean(mod_array),2)
            tot_mean_obs=round(np.nanmean(obs_array),2)
            mean_all=[tot_mean_mod,tot_mean_obs]

            plotname = date_in + '_' + date_fin + '_' + time_res_to_average +  '_qqPlot.png'
            title = 'Surface Current Velocity -ALL \n Period: ' + date_in + '-' + date_fin
            xlabel = 'Observation Current Velocity [m/s]'
            ylabel = 'Model Current Velocity [m/s]'
            if timerange.shape[0] > 2:
                statistics_array=scatterPlot(mod_array,obs_array,path_to_out_plot_folder_arr[exp] + plotname,label_plot_arr[exp],len(listOfFiles),timerange.shape[0],possible_markers,hfr_names,possible_colors,string_time_res,len_not_nan_values=len_not_nan_values,title=title,xlabel=xlabel,ylabel=ylabel)
            row_all = mean_all + statistics_array
            statistics[exp]["ALL HFR STATIONS"] = row_all

            a_file = open(path_to_out_plot_folder_arr[exp]+"statistics_" + name_exp_arr[exp] + "_" + date_in + "_" + date_fin + ".csv", "w")
            writer = csv.writer(a_file)
            writer.writerow(["name_hfr", "mean_mod", "mean_obs", "bias","rmse","si","corr","stderr","number_of_points"])
            for key, value in statistics[exp].items():
                array = [key] + value
                print(array)
                writer.writerow(array)
            a_file.close()

        elif interpolation=='OTM':
            tot_mean_mod_rev_interp=round(np.nanmean(mod_array_rev_interp),2)
            tot_mean_obs_rev_interp=round(np.nanmean(obs_array_rev_interp),2)
            mean_all_rev_interp=[tot_mean_mod_rev_interp,tot_mean_obs_rev_interp]

            plotname = date_in + '_' + date_fin + '_' + time_res_to_average +  '_qqPlot_rev_interp.png'
            title = 'Surface Current Velocity -ALL (Rev Interp)\n Period: ' + date_in + '-' + date_fin
            xlabel = 'Observation Current Velocity [m/s]'
            ylabel = 'Model Current Velocity [m/s]'
            if timerange.shape[0] > 2:
                statistics_array_rev_interp=scatterPlot(mod_array_rev_interp,obs_array_rev_interp,path_to_out_plot_folder_arr[exp] + plotname,label_plot_arr[exp],len(listOfFiles),timerange.shape[0],possible_markers,hfr_names_rev,possible_colors,string_time_res,len_not_nan_values=len_not_nan_values_rev,title=title,xlabel=xlabel,ylabel=ylabel)
            row_all_rev_interp = mean_all_rev_interp + statistics_array_rev_interp
            statistics_rev_interp[exp]["ALL HFR STATIONS"] = row_all_rev_interp

            a_file = open(path_to_out_plot_folder_arr[exp]+"statistics_rev_interp_" + name_exp_arr[exp] + "_" + date_in + "_" + date_fin + ".csv", "w")
            writer = csv.writer(a_file)
            writer.writerow(["name_hfr", "mean_mod", "mean_obs", "bias","rmse","si","corr","stderr","number_of_points"])
            for key, value in statistics_rev_interp[exp].items():
                array = [key] + value
                print(array)
                writer.writerow(array)
            a_file.close()
        else:
            sys.exit(1)
            

    y_min_obs_value = {}
    y_max_obs_value = {}
    y_min_mod_value = {}
    y_max_mod_value = {}
    os.makedirs(path_to_out_plot_folder_comparison, exist_ok=True)
    for hfr_counter,hfr_file in enumerate(listOfFiles):
        print('loading ' + hfr_file + ' ...')
        ds = xarray.open_dataset(hfr_file)
        y_min_obs_value[ds.id] = min(y_obs_min[ds.id],y_obs_min_averaged[ds.id])
        y_max_obs_value[ds.id] = max(y_obs_max[ds.id],y_obs_max_averaged[ds.id])
        y_min_mod_value[ds.id] = np.inf
        y_max_mod_value[ds.id] = 0
        for exp in range(len(name_exp_arr)):
            y_min_mod_value[ds.id] = min(y_min_mod_value[ds.id], y_model_min[exp][ds.id])
            y_max_mod_value[ds.id] = max(y_max_mod_value[ds.id], y_model_max[exp][ds.id])
    
    label_for_taylor = list(np.append('Non-Dimensional Observation', label_plot_arr))
    markers = ['o', 's', '^', '+', 'x', 'D']
    
    for hfr_counter,hfr_file in enumerate(listOfFiles):
        print('loading ' + hfr_file + ' ...')
        ds = xarray.open_dataset(hfr_file)
        
        info = getSourceAntennas(ds)
        if 'id' not in list(ds.attrs.keys()):
            head, tail = os.path.split(hfr_file)
            splitted_hf_name=tail.split(".")
            ds.attrs['id']=splitted_hf_name[0]        

        ds_1=ds[['QCflag','EWCT','NSCT','LATITUDE','LONGITUDE']]
        ds_restricted=ds_1[['EWCT','NSCT','LATITUDE','LONGITUDE']].where((ds.QCflag==0) | (ds.QCflag == 1) | (ds.QCflag==2),drop=True)
        idx1_restricted,idx2_restricted,closerval1_restricted,closerval2_restricted = Get_Closest_Hfr_Time_Range_Index('1D',date_in,date_fin,ds_restricted)
        max_hfr_original = np.nanmax((ds_restricted['EWCT'][idx1_restricted:idx2_restricted+1,0].data ** 2 + ds_restricted['NSCT'][idx1_restricted:idx2_restricted+1,0].data ** 2) ** 0.5)
        min_hfr_original = np.nanmin((ds_restricted['EWCT'][idx1_restricted:idx2_restricted+1,0].data ** 2 + ds_restricted['NSCT'][idx1_restricted:idx2_restricted+1,0].data ** 2) ** 0.5)
        c_dir_cardinal=wind_direction(ds_restricted['EWCT'].data, ds_restricted['NSCT'].data)
        ds_restricted=ds_restricted.assign(CURRENT_DIRECTION=(['TIME', 'DEPTH', 'LATITUDE', 'LONGITUDE'],c_dir_cardinal))
        
        averaged_ds=ds_restricted.resample(TIME=time_res_to_average).mean(skipna=True)
        c_dir_cardinal=wind_direction(averaged_ds['EWCT'].data, averaged_ds['NSCT'].data)
        averaged_ds=averaged_ds.assign(CURRENT_DIRECTION=(['TIME', 'DEPTH', 'LATITUDE', 'LONGITUDE'],c_dir_cardinal))

        lat_hfr=averaged_ds.variables['LATITUDE'][:]
        lon_hfr=averaged_ds.variables['LONGITUDE'][:]

        x = averaged_ds['LONGITUDE'].data
        y = averaged_ds['LATITUDE'].data

        idx1,idx2,closerval1,closerval2 = Get_Closest_Hfr_Time_Range_Index(time_res_to_average,date_in,date_fin,averaged_ds)

        extent = [info['bbox'][0], info['bbox'][1]+0.2,info['bbox'][2], info['bbox'][3]+0.1]

        max_hfr = np.nanmax((averaged_ds['EWCT'][idx1:idx2+1,0].data ** 2 + averaged_ds['NSCT'][idx1:idx2+1,0].data ** 2) ** 0.5)
        min_hfr = np.nanmin((averaged_ds['EWCT'][idx1:idx2+1,0].data ** 2 + averaged_ds['NSCT'][idx1:idx2+1,0].data ** 2) ** 0.5)

            
        name_file_substring="windrose"
        title_substring= ds.id + " Windrose"
        a=ds_restricted['CURRENT_DIRECTION'][idx1_restricted:idx2_restricted+1,0].data.ravel()
        #a=ds_restricted['CURRENT_DIRECTION'][idx1_restricted:idx2_restricted+1].data.ravel()
        b=((ds_restricted['EWCT'][idx1_restricted:idx2_restricted+1,0].data ** 2 + ds_restricted['NSCT'][idx1_restricted:idx2_restricted+1,0].data ** 2) ** 0.5).ravel()
        #b=((ds_restricted['EWCT'][idx1_restricted:idx2_restricted+1].data ** 2 + ds_restricted['NSCT'][idx1_restricted:idx2_restricted+1].data ** 2) ** 0.5).ravel()
        plot_windrose(a[~np.isnan(a)],b[~np.isnan(a)],min(min_hfr_original,min_model_value), max(max_hfr_original,max_model_value),ds,date_in,date_fin,name_file_substring,title_substring,path_to_out_plot_folder_comparison,min(y_min_obs_value[ds.id],y_min_mod_value[ds.id]),max(y_max_obs_value[ds.id],y_max_mod_value[ds.id]))
        
        name_file_substring="averaged_windrose"
        title_substring= ds.id + " Averaged Windrose "
        a=averaged_ds["CURRENT_DIRECTION"][idx1:idx2+1,0].data.ravel()
        b=((averaged_ds['EWCT'][idx1:idx2+1,0].data ** 2 + averaged_ds['NSCT'][idx1:idx2+1,0].data ** 2) ** 0.5).ravel()
        #a=averaged_ds["CURRENT_DIRECTION"][idx1:idx2+1].data.ravel()
        #b=((averaged_ds['EWCT'][idx1:idx2+1].data ** 2 + averaged_ds['NSCT'][idx1:idx2+1].data ** 2) ** 0.5).ravel()
        plot_windrose(a[~np.isnan(a)],b[~np.isnan(a)],min(min_hfr,min_model_value), max(max_hfr,max_model_value),ds,date_in,date_fin,name_file_substring,title_substring,path_to_out_plot_folder_comparison,min(y_min_obs_value[ds.id],y_min_mod_value[ds.id]),max(y_max_obs_value[ds.id],y_max_mod_value[ds.id]))

        for exp in range(len(name_exp_arr)):

            name_file_substring="averaged_windrose_" + name_exp_arr[exp]
            title_substring= name_exp_arr[exp] + " Averaged Windrose for " + ds.id
            a=c_model_dir[exp][ds.id].ravel()
            b=((masked_u_interpolated_arr[exp][ds.id] ** 2 + masked_v_interpolated_arr[exp][ds.id] ** 2) ** 0.5).ravel()
            plot_windrose(a[~np.isnan(a)],b[~np.isnan(a)],min(min_model_value,min_hfr), max(max_model_value,max_hfr),ds,date_in,date_fin,name_file_substring,title_substring,path_to_out_plot_folder_arr[exp],min(y_min_obs_value[ds.id],y_min_mod_value[ds.id]),max(y_max_obs_value[ds.id],y_max_mod_value[ds.id]))
            
            a = ((masked_u_interpolated_arr[exp][ds.id] ** 2 + masked_v_interpolated_arr[exp][ds.id] ** 2) ** 0.5).ravel()
            b = ((averaged_ds['EWCT'][idx1:idx2+1,0].data ** 2 + averaged_ds['NSCT'][idx1:idx2+1,0].data ** 2) ** 0.5).ravel()
            a = a[~np.isnan(b)]
            b = b[~np.isnan(b)]
            taylor_stats = sm.taylor_statistics(a[~np.isnan(a)], b[~np.isnan(a)])

            if exp == 0:
                sdev[ds.id] = list(
                    np.around(np.array([taylor_stats['sdev'][0], taylor_stats['sdev'][1]]), 4))
                crmsd[ds.id] = list(
                    np.around(np.array([taylor_stats['crmsd'][0], taylor_stats['crmsd'][1]]), 4))
                ccoef[ds.id] = list(
                    np.around(np.array([taylor_stats['ccoef'][0], taylor_stats['ccoef'][1]]), 4))
            else:
                append_value(sdev, ds.id, round(
                    taylor_stats['sdev'][1], 4))
                append_value(crmsd, ds.id, round(
                    taylor_stats['crmsd'][1], 4))
                append_value(ccoef, ds.id, round(
                    taylor_stats['ccoef'][1], 4))

            obsSTD = [sdev[ds.id][0]]
            s = sdev[ds.id][1:]
            r = ccoef[ds.id][1:]

            l = label_for_taylor[1:]

            fname = ds.id + '_TaylorDiagram.png'
            srl(obsSTD, s, r, l, fname, markers, path_to_out_plot_folder_comparison)
        

    if len(name_exp_arr)>1:
        os.makedirs(path_to_out_plot_folder_comparison, exist_ok=True)
        for hfr_counter,hfr_file in enumerate(listOfFiles):

            print('loading ' + hfr_file + ' ...')
            ds = xarray.open_dataset(hfr_file)

            info = getSourceAntennas(ds)
            if 'id' not in list(ds.attrs.keys()):
                head, tail = os.path.split(hfr_file)
                splitted_hf_name=tail.split(".")
                ds.attrs['id']=splitted_hf_name[0]
            title_substring='Spatial Surface Current Velocity Mean Comparison'
            name_file_substring='_mod_obs_ts_comparison_all'
            plot_mod_obs_ts_comparison_1(spatial_mean_hfr_ts, spatial_mean_model_ts, time_res_to_average, ds, date_in, date_fin, path_to_out_plot_folder_comparison,timerange,label_plot_arr,title_substring,name_file_substring,len(name_exp_arr))
            
if __name__ == "__main__":

    args = parse_args()
    main(args)