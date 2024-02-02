import argparse
from datetime import date, timedelta
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import xarray
from netCDF4 import Dataset
from core import Get_List_Of_Files, Get_String_Time_Resolution, getSourceAntennas, find_nearest, Get_Max_Min_Interpolated_Model, Get_Max_Min_Bias, Get_Max_Min_Rmsd, seaoverland, interp_mod_to_obs, wind_direction, append_value, unlist, find_date_indices
from core_plot import plot_hfr_wind_field, plot_model_wind_field, plot_bias, plot_rmsd, scatterPlot, plot_mod_obs_ts_comparison, plot_mod_obs_ts_comparison_1, plot_windrose, TaylorDiagram, srl, QQPlot
import numpy as np
import numpy.ma as ma
import csv
from windrose import WindroseAxes
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
    mesh_mask_arr = args.mesh_mask_arr.split()
    u_combined_arr = args.u_combined_arr.split()
    v_combined_arr = args.v_combined_arr.split()
    name_exp_arr = args.name_exp_arr.split()
    label_plot_arr = args.label_plot_arr.split()
    path_to_out_plot_folder_arr = args.path_to_out_plot_folder_arr.split()
    path_to_out_plot_folder_comparison = args.path_to_out_plot_folder_comparison

    start_date = date(int(date_in[0:4]), int(date_in[4:6]), int(date_in[6:8]))
    end_date = date(int(date_fin[0:4]), int(date_fin[4:6]), int(date_fin[6:8]))

    listOfFiles = Get_List_Of_Files(path_to_hfr_folder)

    if time_res_to_average[1] == 'M':
        if end_date.day != pd.DatetimeIndex([end_date])[0].days_in_month:
            end_date += pd.offsets.MonthEnd()
        timerange = pd.date_range(
            start_date, end_date, freq=time_res_to_average[1]) - pd.DateOffset(days=15)
    if time_res_to_average[1] == 'D':
        timerange = pd.date_range(
            start_date, end_date, freq=time_res_to_average[1]) + timedelta(hours=12)

    string_time_res = Get_String_Time_Resolution(
        start_date, end_date, time_res_to_average)

    skip = (slice(None, None, 3), slice(None, None, 3))
    skip_coords = (slice(None, None, 3))
    
    statistics_areal_info = {}
    statistics_qq = {}
    spatial_mean_hfr_ts = {}
    len_hfr = {}
    spatial_mean_model_ts = {}
    len_model = {}

    sdev = {}
    crmsd = {}
    ccoef = {}

    label_for_taylor = list(
        np.append('Non-Dimensional Observation', label_plot_arr))
    markers = ['o', 's', '^', '+', 'x', 'D']

    y_model_min = {}
    y_model_max = {}
    y_obs_min = {}
    y_obs_max = {}
    y_obs_min_averaged = {}
    y_obs_max_averaged = {}

    masked_subset_u_model = {}
    masked_subset_v_model = {}
    masked_subset_speed_model = {}
 
    masked_u_interpolated_arr = {}
    masked_v_interpolated_arr = {}
    c_model_dir = {}

    min_model_value = {}
    max_model_value = {}
    min_model_bias = {}
    max_model_bias = {}
    min_model_rmsd = {}
    max_model_rmsd = {}

    possible_colors = ['red', 'blue', 'black', 'green',
                       'purple', 'orange', 'brown', 'pink', 'grey', 'olive']
    possible_markers = np.array(["o", "^", "s", "P", "*", "D"])

    os.makedirs(path_to_out_plot_folder_comparison, exist_ok=True)

    mod_array = {}
    mod_array_qq = {}
    obs_array = np.array([])
    obs_array_qq = np.array([])

    hfr_names = []

    len_not_nan_values = []
    len_not_nan_values_qq = []
    rem_count = 0

    for hfr_counter, hfr_file in enumerate(listOfFiles):
        
        print('loading ' + hfr_file + ' ...')
        ds = xarray.open_dataset(hfr_file)
        info = getSourceAntennas(ds)
        if 'id' not in list(ds.attrs.keys()):
            head, tail = os.path.split(hfr_file)
            splitted_hf_name = tail.split(".")
            ds.attrs['id'] = splitted_hf_name[0]

        ds_1 = ds[['QCflag', 'EWCT', 'NSCT', 'LATITUDE', 'LONGITUDE']]
        ds_restricted = ds_1[['EWCT', 'NSCT', 'LATITUDE', 'LONGITUDE']].where(
            (ds.QCflag == 0) | (ds.QCflag == 1) | (ds.QCflag == 2), drop=True)

        idx1_original, idx2_original, start_bool_value, final_bool_value = find_date_indices(
            ds, date_in, date_fin, 'D')
        
        if not start_bool_value or not final_bool_value:
            print("radar skipped: ", ds.id)
            rem_count = rem_count + 1
            continue
        
        c_dir_cardinal = wind_direction(
            ds_restricted['EWCT'].data, ds_restricted['NSCT'].data)
        
        ds_restricted = ds_restricted.assign(CURRENT_DIRECTION=(
            ['TIME', 'DEPTH', 'LATITUDE', 'LONGITUDE'], c_dir_cardinal))

        averaged_ds = ds_restricted.resample(
            TIME=time_res_to_average).mean(skipna=True)
        
        c_dir_cardinal = wind_direction(
            averaged_ds['EWCT'].data, averaged_ds['NSCT'].data)
        
        averaged_ds = averaged_ds.assign(CURRENT_DIRECTION=(
            ['TIME', 'DEPTH', 'LATITUDE', 'LONGITUDE'], c_dir_cardinal))

        lat_hfr = averaged_ds.variables['LATITUDE'][:]
        lon_hfr = averaged_ds.variables['LONGITUDE'][:]

        x = averaged_ds['LONGITUDE'].data
        y = averaged_ds['LATITUDE'].data

        idx1, idx2, start_bool_value, final_bool_value = find_date_indices(
            averaged_ds, date_in, date_fin, time_res_to_average[-1])
        
        if not start_bool_value or not final_bool_value:
            print("radar skipped: ", ds.id)
            rem_count = rem_count + 1
            continue

        y_obs_min[ds.id] = {}
        y_obs_max[ds.id] = {}
        y_obs_min_averaged[ds.id] = {}
        y_obs_max_averaged[ds.id] = {}
        y_model_min[ds.id] = {}
        y_model_max[ds.id] = {}

        spatial_mean_model_ts[ds.id] = {}
        len_model[ds.id] = {}
        statistics_qq[ds.id] = {}
        statistics_qq["ALL HFR STATIONS"] = {}
        statistics_areal_info[ds.id] = {}
        statistics_areal_info["ALL HFR STATIONS"] = {}

        extent = [info['bbox'][0], info['bbox'][1] +
                  0.2, info['bbox'][2], info['bbox'][3]+0.1]

        max_hfr = np.nanmax((averaged_ds['EWCT'][idx1: idx2+1, 0].data **
                            2 + averaged_ds['NSCT'][idx1: idx2+1, 0].data ** 2) ** 0.5)
        min_hfr = np.nanmin((averaged_ds['EWCT'][idx1: idx2+1, 0].data **
                            2 + averaged_ds['NSCT'][idx1: idx2+1, 0].data ** 2) ** 0.5)

        min_model_value[ds.id] = {}
        max_model_value[ds.id] = {}
        min_model_bias[ds.id] = {}
        max_model_bias[ds.id] = {}
        min_model_rmsd[ds.id] = {}
        max_model_rmsd[ds.id] = {}

        masked_subset_u_model[ds.id] = {}
        masked_subset_v_model[ds.id] = {}
        masked_subset_speed_model[ds.id] = {}

        masked_u_interpolated_arr[ds.id] = {}
        masked_v_interpolated_arr[ds.id] = {}
        c_model_dir[ds.id] = {}

        for exp in range(len(name_exp_arr)):
            y_model_min[ds.id][exp] = {}
            y_model_max[ds.id][exp] = {}

            spatial_mean_model_ts[ds.id][exp] = []
            len_model[ds.id][exp] = []
            statistics_areal_info[ds.id][exp] = {}
            statistics_qq[ds.id][exp] = {}

            os.makedirs(path_to_out_plot_folder_arr[exp], exist_ok=True)

            mesh_mask_ds = Dataset(mesh_mask_arr[exp])
            if 'x' in list(mesh_mask_ds.variables) or 'y' in list(mesh_mask_ds.variables):
                mesh_mask = xarray.Dataset(data_vars=dict(tmask=(['time_counter', 'z', 'y', 'x'], mesh_mask_ds['tmask']), nav_lon=(['y', 'x'], mesh_mask_ds['x']), nav_lat=(['y', 'x'], mesh_mask_ds['y']), nav_lev=(['z'], mesh_mask_ds['nav_lev']), umask=(['time_counter', 'nav_lev', 'y', 'x'], mesh_mask_ds['umask']), vmask=(['time_counter', 'nav_lev', 'y', 'x'], mesh_mask_ds['vmask']), glamt=(['time_counter', 'y', 'x'], mesh_mask_ds['glamt']), gphit=(
                    ['time_counter', 'y', 'x'], mesh_mask_ds['gphit']), glamu=(['time_counter', 'y', 'x'], mesh_mask_ds['glamu']), gphiu=(['time_counter', 'y', 'x'], mesh_mask_ds['gphiu']), glamv=(['time_counter', 'y', 'x'], mesh_mask_ds['glamv']), gphiv=(['time_counter', 'y', 'x'], mesh_mask_ds['gphiv']), glamf=(['time_counter', 'y', 'x'], mesh_mask_ds['glamf']), gphif=(['time_counter', 'y', 'x'], mesh_mask_ds['gphif'])))
            else:
                mesh_mask = xarray.open_dataset(
                    xarray.backends.NetCDF4DataStore(mesh_mask_ds))

            t_mask = mesh_mask.tmask.values

            # read the model input
            ds_model_u = xarray.open_dataset(
                u_combined_arr[exp])
            ds_model_u.close()
            ds_model_v = xarray.open_dataset(
                v_combined_arr[exp])
            ds_model_v.close()

            ds_model_u_1 = ds_model_u[['nav_lon', 'nav_lat', 'destaggered_u']]
            ds_model_v_1 = ds_model_v[['nav_lon', 'nav_lat', 'destaggered_v']]

            # average in time the model input
            averaged_model_u = ds_model_u_1.resample(
                time_counter=time_res_to_average).mean(skipna=True)
            averaged_model_u.close()
            averaged_model_v = ds_model_v_1.resample(
                time_counter=time_res_to_average).mean(skipna=True)
            averaged_model_v.close()

            u_model = averaged_model_u.variables['destaggered_u'][:, :, :].data
            v_model = averaged_model_v.variables['destaggered_v'][:, :, :].data
            speed_model = np.sqrt(u_model*u_model + v_model*v_model)

            # repeat the mask for all the days
            T_mask = np.repeat(
                t_mask[:, :, :, :], speed_model.shape[0], axis=0)
            t_mask_1 = T_mask[:, 0, :, :]
            
            u_model = ma.masked_array(
                u_model, mask=np.logical_not(t_mask_1))
            v_model = ma.masked_array(
                v_model, mask=np.logical_not(t_mask_1))
            speed_model = ma.masked_array(
                speed_model, mask=np.logical_not(t_mask_1))

            lon_model = averaged_model_u['nav_lon'].data[0, :]
            lat_model = averaged_model_u['nav_lat'].data[:, 0]

            closer_min_mod_lon = find_nearest(
                lon_model, np.nanmin(ds['LONGITUDE'].data))
            closer_min_mod_lat = find_nearest(
                lat_model, np.nanmin(ds['LATITUDE'].data))
            closer_max_mod_lon = find_nearest(
                lon_model, np.nanmax(ds['LONGITUDE'].data))
            closer_max_mod_lat = find_nearest(
                lat_model, np.nanmax(ds['LATITUDE'].data))

            coord_min_lon_min_lat = [
                closer_min_mod_lon, closer_min_mod_lat]
            coord_idx_min_lon_min_lat = np.argwhere((averaged_model_u['nav_lon'].data == coord_min_lon_min_lat[0]) &
                                                    (averaged_model_u['nav_lat'].data == coord_min_lon_min_lat[1]))[0]
            coord_min_lon_max_lat = [
                closer_min_mod_lon, closer_max_mod_lat]
            coord_idx_min_lon_max_lat = np.argwhere((averaged_model_u['nav_lon'].data == coord_min_lon_max_lat[0]) &
                                                    (averaged_model_u['nav_lat'].data == coord_min_lon_max_lat[1]))[0]
            coord_max_lon_min_lat = [
                closer_max_mod_lon, closer_min_mod_lat]
            coord_idx_max_lon_min_lat = np.argwhere((averaged_model_u['nav_lon'].data == coord_max_lon_min_lat[0]) &
                                                    (averaged_model_u['nav_lat'].data == coord_max_lon_min_lat[1]))[0]
            coord_max_lon_max_lat = [
                closer_max_mod_lon, closer_max_mod_lat]
            coord_idx_max_lon_max_lat = np.argwhere((averaged_model_u['nav_lon'].data == coord_max_lon_max_lat[0]) &
                                                    (averaged_model_u['nav_lat'].data == coord_max_lon_max_lat[1]))[0]

            subset_averaged_model_u = averaged_model_u.isel(x=slice(coord_idx_min_lon_min_lat[1]-1,
                                                            coord_idx_max_lon_min_lat[1]+1),
                                                            y=slice(coord_idx_min_lon_min_lat[0]-1,
                                                            coord_idx_min_lon_max_lat[0]+1))
            subset_averaged_model_v = averaged_model_v.isel(x=slice(coord_idx_min_lon_min_lat[1]-1,
                                                            coord_idx_max_lon_min_lat[1]+1),
                                                            y=slice(coord_idx_min_lon_min_lat[0]-1,
                                                            coord_idx_min_lon_max_lat[0]+1))

            x_subset_model = subset_averaged_model_u['nav_lon'].data[0, :]
            y_subset_model = subset_averaged_model_u['nav_lat'].data[:, 0]

            subset_u_model = subset_averaged_model_u.variables['destaggered_u'][:, :, :].data
            subset_v_model = subset_averaged_model_v.variables['destaggered_v'][:, :, :].data
            subset_speed_model = np.sqrt(
                subset_u_model * subset_u_model + subset_v_model * subset_v_model)

            subset_t_mask = t_mask_1[:, slice(coord_idx_min_lon_min_lat[0]-1, coord_idx_min_lon_max_lat[0]+1), slice(
                coord_idx_min_lon_min_lat[1]-1, coord_idx_max_lon_min_lat[1]+1)]
            subset_t_mask = np.logical_not(subset_t_mask)
            masked_subset_u_model[ds.id][exp] = ma.masked_array(
                subset_u_model, mask=subset_t_mask)
            masked_subset_v_model[ds.id][exp] = ma.masked_array(
                subset_v_model, mask=subset_t_mask)
            masked_subset_speed_model[ds.id][exp] = ma.masked_array(
                subset_speed_model, mask=subset_t_mask)

            min_model_value[ds.id][exp], max_model_value[ds.id][exp], *_ = Get_Max_Min_Interpolated_Model(
                idx1, idx2, averaged_ds, masked_subset_speed_model[ds.id][exp], x_subset_model, y_subset_model, lon_hfr, lat_hfr)
            min_model_bias[ds.id][exp], max_model_bias[ds.id][exp], *_ = Get_Max_Min_Bias(
                idx1, idx2, averaged_ds, masked_subset_speed_model[ds.id][exp], x_subset_model, y_subset_model, lon_hfr, lat_hfr)
            min_model_rmsd[ds.id][exp], max_model_rmsd[ds.id][exp], *_ = Get_Max_Min_Rmsd(
                idx1, idx2, averaged_ds, masked_subset_speed_model[ds.id][exp], x_subset_model, y_subset_model, lon_hfr, lat_hfr)

        values = min_model_value[ds.id].values()
        minimum_value = min(min(values), min_hfr)
        values = max_model_value[ds.id].values()
        maximum_value = max(max(values), max_hfr)
        values = min_model_bias[ds.id].values()
        minimum_bias_value = min(values)
        values = max_model_bias[ds.id].values()
        maximum_bias_value = max(values)
        values = min_model_rmsd[ds.id].values()
        minimum_rmsd_value = min(values)
        values = max_model_rmsd[ds.id].values()
        maximum_rmsd_value = max(values)

        a = ds_restricted['CURRENT_DIRECTION'][idx1_original:idx2_original+1, 0].data.ravel()
        b = ((ds_restricted['EWCT'][idx1_original:idx2_original+1, 0].data ** 2 +
              ds_restricted['NSCT'][idx1_original:idx2_original+1, 0].data ** 2) ** 0.5).ravel()
        a = a[~np.isnan(b)]
        b = b[~np.isnan(b)]
        ax = WindroseAxes.from_ax()
        turbo = plt.get_cmap('turbo')
        ax.bar(a[~np.isnan(a)], b[~np.isnan(a)], normed=True, bins=np.linspace(
            minimum_value, maximum_value, 5), opening=0.8, edgecolor='white', cmap=turbo)
        y_obs_min[ds.id], y_obs_max[ds.id] = ax.get_ylim()

        a = averaged_ds['CURRENT_DIRECTION'][idx1:idx2+1, 0].data.ravel()
        b = ((averaged_ds['EWCT'][idx1:idx2+1, 0].data ** 2 +
              averaged_ds['NSCT'][idx1:idx2+1, 0].data ** 2) ** 0.5).ravel()
        a = a[~np.isnan(b)]
        b = b[~np.isnan(b)]
        ax = WindroseAxes.from_ax()
        turbo = plt.get_cmap('turbo')
        ax.bar(a[~np.isnan(a)], b[~np.isnan(a)], normed=True, bins=np.linspace(
            minimum_value, maximum_value, 5), opening=0.8, edgecolor='white', cmap=turbo)
        y_obs_min_averaged[ds.id], y_obs_max_averaged[ds.id] = ax.get_ylim()

        splitted_name = ds.id.split("-")
        hfr_names.append(splitted_name[1])

        for exp in range(len(name_exp_arr)):
            spatial_mean_hfr_ts[ds.id] = []
            len_hfr[ds.id] = []

            masked_u_interpolated_arr[ds.id][exp] = np.empty([len(range(
                idx1, idx2+1)), ds_restricted['EWCT'].data.shape[2], ds_restricted['EWCT'].data.shape[3]])
            masked_v_interpolated_arr[ds.id][exp] = np.empty([len(range(
                idx1, idx2+1)), ds_restricted['EWCT'].data.shape[2], ds_restricted['EWCT'].data.shape[3]])
            c_model_dir[ds.id][exp] = np.empty([len(range(
                idx1, idx2+1)), ds_restricted['EWCT'].data.shape[2], ds_restricted['EWCT'].data.shape[3]])

            for time_counter, index in enumerate(range(idx1, idx2+1)):
                date_str = string_time_res[time_counter]

                U = averaged_ds['EWCT'][index, 0].data
                V = averaged_ds['NSCT'][index, 0].data
                speed_hfr = (U ** 2 + V ** 2) ** 0.5
                len_hfr[ds.id].append(speed_hfr.size)
                spatial_mean_hfr_ts[ds.id].append(np.nanmean(speed_hfr))
                mask_hfr = np.ma.masked_invalid(speed_hfr).mask

                plot_hfr_wind_field(info, extent, minimum_value, maximum_value, x, y, speed_hfr,
                                    U, V, skip, skip_coords, date_str, ds, path_to_out_plot_folder_arr[exp])
                subset_speed_model_instant = seaoverland(
                    masked_subset_speed_model[ds.id][exp][time_counter], 3)
                subset_u_model_instant = seaoverland(
                    masked_subset_u_model[ds.id][exp][time_counter], 3)
                subset_v_model_instant = seaoverland(
                    masked_subset_v_model[ds.id][exp][time_counter], 3)
                masked_speed_interpolated, masked_u_interpolated, masked_v_interpolated, spatial_mean_model_ts_instant = interp_mod_to_obs(
                    x_subset_model, y_subset_model, subset_speed_model_instant, subset_u_model_instant, subset_v_model_instant, lon_hfr, lat_hfr, mask_hfr)
                c_model_dir[ds.id][exp][time_counter, :, :] = wind_direction(
                    masked_u_interpolated.data, masked_v_interpolated.data)
                masked_u_interpolated_arr[ds.id][exp][time_counter,
                                                      :, :] = masked_u_interpolated.data
                masked_v_interpolated_arr[ds.id][exp][time_counter,
                                                      :, :] = masked_v_interpolated.data
                spatial_mean_model_ts[ds.id] = append_value(
                    spatial_mean_model_ts[ds.id], exp, spatial_mean_model_ts_instant)
                len_model[ds.id] = append_value(
                    len_model[ds.id], exp, masked_speed_interpolated.size)

                title_substring = 'interpolated model surface current'
                name_file_substring = 'model_surface_current_velocity_'
                plot_model_wind_field(info, extent, minimum_value, maximum_value, x, y, skip, skip_coords, masked_speed_interpolated, masked_u_interpolated, masked_v_interpolated,
                                      date_str, path_to_out_plot_folder_arr[exp], label_plot_arr[exp], title_substring, name_file_substring, ds, spatial_mean_model_ts_instant)

                title_substring = 'surface current bias'
                name_file_substring = 'surface_current_velocity_bias'
                plot_bias(info, extent, x, y, minimum_bias_value, maximum_bias_value, masked_speed_interpolated, speed_hfr,
                          date_str, path_to_out_plot_folder_arr[exp], label_plot_arr[exp], title_substring, name_file_substring, ds)

                title_substring = 'surface current rmsd'
                name_file_substring = 'surface_current_velocity_rmsd'
                plot_rmsd(info, extent, x, y, minimum_rmsd_value, maximum_rmsd_value, masked_speed_interpolated, speed_hfr,
                          date_str, path_to_out_plot_folder_arr[exp], label_plot_arr[exp], title_substring, name_file_substring, ds)

            a = c_model_dir[ds.id][exp].ravel()
            b = ((masked_u_interpolated_arr[ds.id][exp] ** 2 +
                  masked_v_interpolated_arr[ds.id][exp] ** 2) ** 0.5).ravel()
            ax = WindroseAxes.from_ax()
            turbo = plt.get_cmap('turbo')
            ax.bar(a[~np.isnan(a)], b[~np.isnan(a)], normed=True, bins=np.linspace(
                minimum_value, maximum_value, 5), opening=0.8, edgecolor='white', cmap=turbo)
            y_model_min[ds.id][exp], y_model_max[ds.id][exp] = ax.get_ylim()

            title_substring = 'Spatial Surface Current Velocity Mean Comparison'
            name_file_substring = '_mod_obs_ts_comparison'
            mean_vel_mod, mean_vel_obs = plot_mod_obs_ts_comparison(spatial_mean_hfr_ts[ds.id][:], spatial_mean_model_ts[ds.id][exp][
                                                                    :], len_hfr[ds.id][:], len_model[ds.id][exp][:], time_res_to_average, ds, date_in, date_fin, path_to_out_plot_folder_arr[exp], timerange, label_plot_arr[exp], title_substring, name_file_substring)
            tot_mean_stat = [mean_vel_mod, mean_vel_obs]

            plotname = ds.id + '_' + label_plot_arr[exp] + '_' + date_in + '_' + date_fin + \
                '_' + time_res_to_average + '_scatter_areal_info.png'
            title = 'Spatial Mean Surface Current Velocity ' + \
                ds.id + '\n Period: ' + date_in + ' - ' + date_fin
            xlabel = 'Observation Current Velocity [m/s]'
            ylabel = 'Model Current Velocity [m/s]'
            if timerange.shape[0] > 2:
                statistics_array = scatterPlot(np.array(spatial_mean_model_ts[ds.id][exp]), np.array(spatial_mean_hfr_ts[ds.id]), path_to_out_plot_folder_arr[exp] + plotname, label_plot_arr[exp], 1, len(
                    spatial_mean_model_ts[ds.id][exp]), possible_markers[hfr_counter], splitted_name[1], possible_colors, string_time_res, title=title, xlabel=xlabel, ylabel=ylabel)
                row_stat = tot_mean_stat + statistics_array
                statistics_areal_info[ds.id][exp] = row_stat

            mod_array = append_value(
                mod_array, exp, spatial_mean_model_ts[ds.id][exp])

            mod_array_qq = append_value(
                mod_array_qq, exp, list(b))

            if exp == 0:
                obs_array = np.concatenate(
                    [obs_array, np.array(spatial_mean_hfr_ts[ds.id])])
                a = averaged_ds['CURRENT_DIRECTION'][idx1:idx2 +
                                                     1, 0].data.ravel()
                b = ((averaged_ds['EWCT'][idx1:idx2+1, 0].data ** 2 +
                      averaged_ds['NSCT'][idx1:idx2+1, 0].data ** 2) ** 0.5).ravel()
                obs_array_qq = np.concatenate(
                    [obs_array_qq, b])
                len_not_nan_values_qq.append(len(b[~np.isnan(a)]))
                ciao = np.array(spatial_mean_hfr_ts[ds.id])
                len_not_nan_values.append(len(ciao[~np.isnan(ciao)]))

        name_file_substring = "windrose"
        title_substring = ds.id + " Windrose"
        a = ds_restricted['CURRENT_DIRECTION'][idx1_original:
                                               idx2_original + 1, 0].data.ravel()
        b = ((ds_restricted['EWCT'][idx1_original:idx2_original+1, 0].data ** 2 +
              ds_restricted['NSCT'][idx1_original:idx2_original+1, 0].data ** 2) ** 0.5).ravel()
        a = a[~np.isnan(b)]
        b = b[~np.isnan(b)]

        plot_windrose(a[~np.isnan(a)], b[~np.isnan(a)], minimum_value, maximum_value, ds, date_in, date_fin, name_file_substring, title_substring, path_to_out_plot_folder_comparison, min(
            y_obs_min_averaged[ds.id], min(y_model_min[ds.id].values())), max(y_obs_max_averaged[ds.id], max(y_model_max[ds.id].values())))

        name_file_substring = "averaged_windrose"
        title_substring = ds.id + " Averaged Windrose "
        a = averaged_ds["CURRENT_DIRECTION"][idx1: idx2+1, 0].data.ravel()
        b = ((averaged_ds['EWCT'][idx1: idx2+1, 0].data ** 2 +
              averaged_ds['NSCT'][idx1: idx2+1, 0].data ** 2) ** 0.5).ravel()
        a = a[~np.isnan(b)]
        b = b[~np.isnan(b)]

        plot_windrose(a[~np.isnan(a)], b[~np.isnan(a)], minimum_value, maximum_value, ds, date_in, date_fin, name_file_substring, title_substring, path_to_out_plot_folder_comparison, min(
            y_obs_min_averaged[ds.id], min(y_model_min[ds.id].values())), max(y_obs_max_averaged[ds.id], max(y_model_max[ds.id].values())))

        for exp in range(len(name_exp_arr)):
            name_file_substring = "averaged_windrose_" + name_exp_arr[exp]
            title_substring = name_exp_arr[exp] + \
                " Averaged Windrose for " + ds.id
            a = c_model_dir[ds.id][exp].ravel()
            b = ((masked_u_interpolated_arr[ds.id][exp] ** 2 +
                  masked_v_interpolated_arr[ds.id][exp] ** 2) ** 0.5).ravel()
            plot_windrose(a[~np.isnan(a)], b[~np.isnan(a)], minimum_value, maximum_value, ds, date_in, date_fin, name_file_substring, title_substring, path_to_out_plot_folder_arr[exp], min(
                y_obs_min_averaged[ds.id], min(y_model_min[ds.id].values())), max(y_obs_max_averaged[ds.id], max(y_model_max[ds.id].values())))

            a = ((masked_u_interpolated_arr[ds.id][exp] ** 2 +
                  masked_v_interpolated_arr[ds.id][exp] ** 2) ** 0.5).ravel()
            b = ((averaged_ds['EWCT'][idx1: idx2+1, 0].data ** 2 +
                  averaged_ds['NSCT'][idx1: idx2+1, 0].data ** 2) ** 0.5).ravel()
            a = a[~np.isnan(b)]
            b = b[~np.isnan(b)]
            plotname = ds.id + '_' + label_plot_arr[exp] + '_' + date_in + '_' + date_fin + \
                '_' + time_res_to_average + '_qqPlot.png'
            title = 'Surface Current Velocity ' + \
                ds.id + '\n Period: ' + date_in + ' - ' + date_fin
            xlabel = 'Observation Current Velocity [m/s]'
            ylabel = 'Model Current Velocity [m/s]'
            statistics_array_qq = QQPlot(a[~np.isnan(a)], b[~np.isnan(
                a)], path_to_out_plot_folder_arr[exp] + plotname, label_plot_arr[exp], title=title, xlabel=xlabel, ylabel=ylabel)
            mean_vel_mod = np.mean(a)
            mean_vel_obs = np.mean(b)
            tot_mean_stat = [mean_vel_mod, mean_vel_obs]
            row_stat = tot_mean_stat + statistics_array_qq
            statistics_qq[ds.id][exp] = row_stat

            a = ((masked_u_interpolated_arr[ds.id][exp] ** 2 +
                 masked_v_interpolated_arr[ds.id][exp] ** 2) ** 0.5).ravel()
            b = ((averaged_ds['EWCT'][idx1: idx2+1, 0].data ** 2 +
                 averaged_ds['NSCT'][idx1: idx2+1, 0].data ** 2) ** 0.5).ravel()
            a = a[~np.isnan(b)]
            b = b[~np.isnan(b)]
            taylor_stats = sm.taylor_statistics(
                a[~np.isnan(a)], b[~np.isnan(a)])

            if exp == 0:
                sdev[ds.id] = list(
                    np.around(np.array([taylor_stats['sdev'][0], taylor_stats['sdev'][1]]), 4))
                crmsd[ds.id] = list(
                    np.around(np.array([taylor_stats['crmsd'][0], taylor_stats['crmsd'][1]]), 4))
                ccoef[ds.id] = list(
                    np.around(np.array([taylor_stats['ccoef'][0], taylor_stats['ccoef'][1]]), 4))
            else:
                sdev = append_value(sdev, ds.id, round(
                    taylor_stats['sdev'][1], 4))
                crmsd = append_value(crmsd, ds.id, round(
                    taylor_stats['crmsd'][1], 4))
                ccoef = append_value(ccoef, ds.id, round(
                    taylor_stats['ccoef'][1], 4))

            obsSTD = [sdev[ds.id][0]]
            s = sdev[ds.id][1:]
            r = ccoef[ds.id][1:]

            l = label_for_taylor[1:]

            fname = ds.id + '_TaylorDiagram.png'
            srl(obsSTD, s, r, l, fname, markers,
                path_to_out_plot_folder_comparison)

    for exp in range(len(label_plot_arr)):

        unlisted_array = np.array(unlist(mod_array[exp]))
        tot_mean_mod = round(np.nanmean(unlisted_array), 2)
        tot_mean_obs = round(np.nanmean(obs_array), 2)
        mean_all = [tot_mean_mod, tot_mean_obs]

        plotname = date_in + '_' + date_fin + '_' + \
            time_res_to_average + '_scatter_areal_info.png'
        title = 'Surface Current Velocity -ALL \n Period: ' + date_in + '-' + date_fin
        xlabel = 'Observation Current Velocity [m/s]'
        ylabel = 'Model Current Velocity [m/s]'
        if timerange.shape[0] > 2:
            statistics_array = scatterPlot(unlisted_array, obs_array, path_to_out_plot_folder_arr[exp] + plotname, label_plot_arr[exp], len(
                listOfFiles)-rem_count, timerange.shape[0], possible_markers, hfr_names, possible_colors, string_time_res, len_not_nan_values=len_not_nan_values, title=title, xlabel=xlabel, ylabel=ylabel)
        row_all = mean_all + statistics_array
        statistics_areal_info["ALL HFR STATIONS"][exp] = row_all

        a_file = open(path_to_out_plot_folder_arr[exp]+"statistics_areal_info" +
                      name_exp_arr[exp] + "_" + date_in + "_" + date_fin + ".csv", "w")
        writer = csv.writer(a_file)
        writer.writerow(["name_hfr", "mean_mod", "mean_obs", "bias",
                        "rmse", "si", "corr", "stderr", "number_of_points"])

        for key, value in statistics_areal_info.items():
            array = [key] + value[exp]
            writer.writerow(array)
        a_file.close()

        unlisted_array = np.array(unlist(mod_array_qq[exp]))

        tot_mean_mod = round(np.nanmean(unlisted_array), 2)
        tot_mean_obs = round(np.nanmean(obs_array_qq), 2)
        mean_all = [tot_mean_mod, tot_mean_obs]

        plotname = date_in + '_' + date_fin + '_' + \
            time_res_to_average + '_qq_info.png'
        title = 'Surface Current Velocity -ALL \n Period: ' + date_in + '-' + date_fin
        xlabel = 'Observation Current Velocity [m/s]'
        ylabel = 'Model Current Velocity [m/s]'
        if timerange.shape[0] > 2:
            statistics_array_qq = QQPlot(unlisted_array, obs_array_qq, path_to_out_plot_folder_arr[exp] + plotname,
                                         label_plot_arr[exp], len_not_nan_values=len_not_nan_values_qq, title=title, xlabel=xlabel, ylabel=ylabel)
        row_all = mean_all + statistics_array_qq
        statistics_qq["ALL HFR STATIONS"][exp] = row_all

        a_file = open(path_to_out_plot_folder_arr[exp]+"statistics_qq_plots" +
                      name_exp_arr[exp] + "_" + date_in + "_" + date_fin + ".csv", "w")
        writer = csv.writer(a_file)
        writer.writerow(["name_hfr", "mean_mod", "mean_obs", "bias",
                        "rmse", "si", "corr", "stderr", "number_of_points"])

        for key, value in statistics_qq.items():
            array = [key] + value[exp]
            writer.writerow(array)
        a_file.close()

    if len(name_exp_arr) > 1:
        os.makedirs(path_to_out_plot_folder_comparison, exist_ok=True)
        for hfr_counter, hfr_file in enumerate(listOfFiles):

            print('loading ' + hfr_file + ' ...')
            ds = xarray.open_dataset(hfr_file)

            info = getSourceAntennas(ds)
            if 'id' not in list(ds.attrs.keys()):
                head, tail = os.path.split(hfr_file)
                splitted_hf_name = tail.split(".")
                ds.attrs['id'] = splitted_hf_name[0]
            idx1_original, idx2_original, start_bool_value, final_bool_value = find_date_indices(
                ds, date_in, date_fin, 'D')
            if not start_bool_value or not final_bool_value:
                print("radar skipped: ", ds.id)
                rem_count = rem_count + 1
                continue
            title_substring = 'Spatial Surface Current Velocity Mean Comparison'
            name_file_substring = '_mod_obs_ts_comparison_all'

            plot_mod_obs_ts_comparison_1(spatial_mean_hfr_ts, spatial_mean_model_ts, time_res_to_average, ds, date_in, date_fin,
                                         path_to_out_plot_folder_comparison, timerange, label_plot_arr, title_substring, name_file_substring, len(name_exp_arr))


if __name__ == "__main__":

    args = parse_args()
    main(args)
