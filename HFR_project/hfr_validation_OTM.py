import argparse
from datetime import date, timedelta
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import xarray
from netCDF4 import Dataset
from core import Get_List_Of_Files, Get_String_Time_Resolution, getSourceAntennas, Get_Closest_Hfr_Time_Range_Index, find_nearest, Get_Max_Min_Interpolated_Model, Get_Max_Min_Bias, Get_Max_Min_Rmsd, seaoverland, interp_hfr_mask_to_mod_mask, interp_obs_to_mod, wind_direction, append_value, unlist, find_date_indices
from core_plot import plot_model_wind_field, plot_bias, plot_rmsd, plot_mod_obs_ts_comparison, scatterPlot, plot_mod_obs_ts_comparison, plot_interpolated_hfr_wind_field, plot_windrose, TaylorDiagram, srl
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
    grid_arr = args.grid_arr.split()
    interpolation = args.interpolation
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

    skip_model = (slice(None, None, 1), slice(None, None, 1))
    skip_coords_model = (slice(None, None, 1))
    statistics_rev_interp = {}
    spatial_not_interp_mean_model_ts = {}
    spatial_mean_interp_hfr_ts = {}

    sdev_rev = {}
    crmsd_rev = {}
    ccoef_rev = {}

    label_for_taylor = list(
        np.append('Non-Dimensional Observation', label_plot_arr))
    markers = ['o', 's', '^', '+', 'x', 'D']

    y_mod_hfr_interp_min = {}
    y_mod_hfr_interp_max = {}
    y_mod_original_min = {}
    y_mod_original_max = {}

    masked_subset_u_model = {}
    masked_subset_v_model = {}
    masked_subset_speed_model = {}
    c_mod_original = {}

    masked_hfr_u_interpolated_arr = {}
    masked_hfr_v_interpolated_arr = {}
    c_mod_hfr_interpolated = {}

    min_model_rev_value = {}
    max_model_rev_value = {}
    min_model_bias_rev = {}
    max_model_bias_rev = {}
    min_model_rmsd_rev = {}
    max_model_rmsd_rev = {}

    min_obs_rev = {}
    max_obs_rev = {}

    possible_colors = ['red', 'blue', 'black', 'green',
                       'purple', 'orange', 'brown', 'pink', 'grey', 'olive']
    possible_markers = np.array(["o", "^", "s", "P", "*", "D"])

    os.makedirs(path_to_out_plot_folder_comparison, exist_ok=True)

    mod_array_rev_interp = {}
    obs_array_rev_interp = {}
    hfr_names_rev = []
    len_not_nan_values_rev = {}

    for hfr_counter, hfr_file in enumerate(listOfFiles):
        print('loading ' + hfr_file + ' ...')
        ds = xarray.open_dataset(hfr_file)
        info = getSourceAntennas(ds)
        if 'id' not in list(ds.attrs.keys()):
            head, tail = os.path.split(hfr_file)
            splitted_hf_name = tail.split(".")
            ds.attrs['id'] = splitted_hf_name[0]

        y_mod_hfr_interp_min[ds.id] = {}
        y_mod_hfr_interp_max[ds.id] = {}
        y_mod_original_min[ds.id] = {}
        y_mod_original_max[ds.id] = {}

        spatial_not_interp_mean_model_ts[ds.id] = {}
        statistics_rev_interp[ds.id] = {}
        statistics_rev_interp["ALL HFR STATIONS"] = {}
        spatial_mean_interp_hfr_ts[ds.id] = {}

        ds_1 = ds[['QCflag', 'EWCT', 'NSCT', 'LATITUDE', 'LONGITUDE']]
        ds_restricted = ds_1[['EWCT', 'NSCT', 'LATITUDE', 'LONGITUDE']].where(
            (ds.QCflag == 0) | (ds.QCflag == 1) | (ds.QCflag == 2), drop=True)

#        idx1_restricted, idx2_restricted, closerval1_restricted, closerval2_restricted = Get_Closest_Hfr_Time_Range_Index(
#            '1M', date_in, date_fin, ds_restricted)

        averaged_ds = ds_restricted.resample(
            TIME=time_res_to_average).mean(skipna=True)

        lat_hfr = averaged_ds.variables['LATITUDE'][:]
        lon_hfr = averaged_ds.variables['LONGITUDE'][:]

        x = averaged_ds['LONGITUDE'].data
        y = averaged_ds['LATITUDE'].data

#        idx1, idx2, closerval1, closerval2 = Get_Closest_Hfr_Time_Range_Index(
#            time_res_to_average, date_in, date_fin, averaged_ds)
        idx1, idx2 = find_date_indices(averaged_ds, date_in, date_fin,time_res_to_average[-1])

        extent = [info['bbox'][0], info['bbox'][1] +
                  0.2, info['bbox'][2], info['bbox'][3]+0.1]

        min_model_rev_value[ds.id] = {}
        max_model_rev_value[ds.id] = {}
        min_model_bias_rev[ds.id] = {}
        max_model_bias_rev[ds.id] = {}
        min_model_rmsd_rev[ds.id] = {}
        max_model_rmsd_rev[ds.id] = {}

        min_obs_rev[ds.id] = {}
        max_obs_rev[ds.id] = {}

        masked_subset_u_model[ds.id] = {}
        masked_subset_v_model[ds.id] = {}
        masked_subset_speed_model[ds.id] = {}
        c_mod_original[ds.id] = {}

        masked_hfr_u_interpolated_arr[ds.id] = {}
        masked_hfr_v_interpolated_arr[ds.id] = {}
        c_mod_hfr_interpolated[ds.id] = {}

        for exp in range(len(name_exp_arr)):
            spatial_not_interp_mean_model_ts[ds.id][exp] = []
            spatial_mean_interp_hfr_ts[ds.id][exp] = []
            statistics_rev_interp[ds.id][exp] = {}

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
            ds_model_u = xarray.open_dataset(u_combined_arr[exp])

            ds_model_u.close()
            ds_model_v = xarray.open_dataset(v_combined_arr[exp])

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

            x_model = averaged_model_u['nav_lon'].data[0, :]
            y_model = averaged_model_u['nav_lat'].data[:, 0]

            u_model = averaged_model_u.variables['destaggered_u'][:, :, :].data
            v_model = averaged_model_v.variables['destaggered_v'][:, :, :].data
            speed_model = np.sqrt(u_model*u_model + v_model*v_model)

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

            _, _, min_model_rev_value[ds.id][exp], max_model_rev_value[ds.id][exp], min_obs_rev[ds.id][exp], max_obs_rev[ds.id][exp] = Get_Max_Min_Interpolated_Model(
                idx1, idx2, averaged_ds, masked_subset_speed_model[ds.id][exp], x_subset_model, y_subset_model, lon_hfr, lat_hfr)
            _, _, min_model_bias_rev[ds.id][exp], max_model_bias_rev[ds.id][exp] = Get_Max_Min_Bias(
                idx1, idx2, averaged_ds, masked_subset_speed_model[ds.id][exp], x_subset_model, y_subset_model, lon_hfr, lat_hfr)
            _, _, min_model_rmsd_rev[ds.id][exp], max_model_rmsd_rev[ds.id][exp] = Get_Max_Min_Rmsd(
                idx1, idx2, averaged_ds, masked_subset_speed_model[ds.id][exp], x_subset_model, y_subset_model, lon_hfr, lat_hfr)

        mod_values = min_model_rev_value[ds.id].values()
        obs_values = min_obs_rev[ds.id].values()
        minimum_value = min(min(mod_values), min(obs_values))
        mod_values = max_model_rev_value[ds.id].values()
        obs_values = max_obs_rev[ds.id].values()
        maximum_value = max(max(mod_values), max(obs_values))
        values = min_model_bias_rev[ds.id].values()
        minimum_bias_value = min(values)
        values = max_model_bias_rev[ds.id].values()
        maximum_bias_value = max(values)
        values = min_model_rmsd_rev[ds.id].values()
        minimum_rmsd_value = min(values)
        values = max_model_rmsd_rev[ds.id].values()
        maximum_rmsd_value = max(values)
        splitted_name = ds.id.split("-")
        hfr_names_rev.append(splitted_name[1])

        for exp in range(len(name_exp_arr)):
            masked_hfr_u_interpolated_arr[ds.id][exp] = np.empty([len(range(
                idx1, idx2+1)), masked_subset_u_model[ds.id][exp].data.shape[1], masked_subset_u_model[ds.id][exp].data.shape[2]])
            masked_hfr_v_interpolated_arr[ds.id][exp] = np.empty([len(range(
                idx1, idx2+1)), masked_subset_u_model[ds.id][exp].data.shape[1], masked_subset_u_model[ds.id][exp].data.shape[2]])
            c_mod_hfr_interpolated[ds.id][exp] = np.empty([len(range(
                idx1, idx2+1)), masked_subset_u_model[ds.id][exp].data.shape[1], masked_subset_u_model[ds.id][exp].data.shape[2]])

            for time_counter, index in enumerate(range(idx1, idx2+1)):
                date_str = string_time_res[time_counter]

                U = averaged_ds['EWCT'][index, 0].data
                V = averaged_ds['NSCT'][index, 0].data
                speed_hfr = (U ** 2 + V ** 2) ** 0.5
                mask_hfr = np.ma.masked_invalid(speed_hfr).mask

                masked_speed_hfr = ma.masked_array(
                    speed_hfr, mask=mask_hfr)
                masked_U = ma.masked_array(U, mask=mask_hfr)
                masked_V = ma.masked_array(V, mask=mask_hfr)
                sol_speed_hfr = seaoverland(masked_speed_hfr, 3)
                sol_u_hfr = seaoverland(masked_U, 3)
                sol_v_hfr = seaoverland(masked_V, 3)
                threshold = 0.7
                step_lon = lon_hfr[1]-lon_hfr[0]
                step_lat = lat_hfr[1]-lat_hfr[0]
                X = np.concatenate(
                    ([lon_hfr[0]-step_lon], lon_hfr, [lon_hfr[-1]+step_lon]))
                Y = np.concatenate(
                    ([lat_hfr[0]-step_lat], lat_hfr, [lat_hfr[-1]+step_lat]))
                mask_hfr_prova = np.pad(np.logical_not(mask_hfr), 1)
                hfr_mask_interpolated = interp_hfr_mask_to_mod_mask(X, Y, np.logical_not(
                    mask_hfr_prova), x_subset_model, y_subset_model, threshold)
                masked_hfr_speed_interpolated, masked_hfr_u_interpolated, masked_hfr_v_interpolated, spatial_mean_hfr_ts_instant = interp_obs_to_mod(
                    lon_hfr, lat_hfr, sol_speed_hfr, sol_u_hfr, sol_v_hfr, x_subset_model, y_subset_model, hfr_mask_interpolated)
                masked_hfr_u_interpolated_arr[ds.id][exp][time_counter,
                                                          :, :] = masked_hfr_u_interpolated.data
                masked_hfr_v_interpolated_arr[ds.id][exp][time_counter,
                                                          :, :] = masked_hfr_v_interpolated.data
                spatial_mean_interp_hfr_ts[ds.id] = append_value(
                    spatial_mean_interp_hfr_ts[ds.id], exp, spatial_mean_hfr_ts_instant)
                # spatial_mean_interp_hfr_ts[ds.id][exp].append(
                #    spatial_mean_hfr_ts_instant)
                c_mod_hfr_interpolated[ds.id][exp][time_counter, :, :] = wind_direction(
                    masked_hfr_u_interpolated.data, masked_hfr_v_interpolated.data)
                title_substring = 'interpolated hfr surface current'
                name_file_substring = '_interp_hfr_surface_current_velocity_'
                plot_interpolated_hfr_wind_field(info, extent, minimum_value, maximum_value, x_subset_model, y_subset_model, skip_model, skip_coords_model, masked_hfr_speed_interpolated,
                                                 masked_hfr_u_interpolated, masked_hfr_v_interpolated, date_str, path_to_out_plot_folder_arr[exp], title_substring, name_file_substring, ds)
                spatial_not_interp_mean_model_ts[ds.id] = append_value(
                    spatial_not_interp_mean_model_ts[ds.id], exp, masked_subset_speed_model[ds.id][exp][time_counter].mean())
                print(
                    "check: ", spatial_not_interp_mean_model_ts[ds.id][exp])
                c_mod_original[ds.id][exp] = wind_direction(
                    masked_subset_u_model[ds.id][exp].data, masked_subset_v_model[ds.id][exp].data)
                title_substring = 'model surface current'
                name_file_substring = '_model_surface_current_velocity_'
                plot_model_wind_field(info, extent, minimum_value, maximum_value, x_subset_model, y_subset_model, skip_model, skip_coords_model, masked_subset_speed_model[ds.id][exp][time_counter], masked_subset_u_model[ds.id][exp][
                    time_counter], masked_subset_v_model[ds.id][exp][time_counter], date_str, path_to_out_plot_folder_arr[exp], label_plot_arr[exp], title_substring, name_file_substring, ds, masked_subset_speed_model[ds.id][exp][time_counter].mean())

                title_substring = 'surface current bias (rev_interp)'
                name_file_substring = '_surface_current_velocity_bias_rev_interp'
                plot_bias(info, extent, x_subset_model, y_subset_model, minimum_bias_value, maximum_bias_value,
                          masked_subset_speed_model[ds.id][exp][time_counter], masked_hfr_speed_interpolated, date_str, path_to_out_plot_folder_arr[exp], label_plot_arr[exp], title_substring, name_file_substring, ds)

                title_substring = 'surface current rmsd (rev_interp)'
                name_file_substring = '_surface_current_velocity_rmsd_rev_interp'
                plot_rmsd(info, extent, x_subset_model, y_subset_model, minimum_rmsd_value, maximum_rmsd_value,
                          masked_subset_speed_model[ds.id][exp][time_counter], masked_hfr_speed_interpolated, date_str, path_to_out_plot_folder_arr[exp], label_plot_arr[exp], title_substring, name_file_substring, ds)

            a = c_mod_hfr_interpolated[ds.id][exp].ravel()
            b = ((masked_hfr_u_interpolated_arr[ds.id][exp] ** 2 +
                  masked_hfr_v_interpolated_arr[ds.id][exp] ** 2) ** 0.5).ravel()
            ax = WindroseAxes.from_ax()
            turbo = plt.get_cmap('turbo')
            ax.bar(a[~np.isnan(a)], b[~np.isnan(a)], normed=True, bins=np.linspace(
                minimum_value, maximum_value, 5), opening=0.8, edgecolor='white', cmap=turbo)
            y_mod_hfr_interp_min[ds.id][exp], y_mod_hfr_interp_max[ds.id][exp] = ax.get_ylim(
            )

            a = c_mod_original[ds.id][exp].ravel()
            b = ((masked_subset_u_model[ds.id][exp] ** 2 +
                  masked_subset_v_model[ds.id][exp] ** 2) ** 0.5).ravel()
            ax = WindroseAxes.from_ax()
            turbo = plt.get_cmap('turbo')
            ax.bar(a[~np.isnan(a)], b[~np.isnan(a)], normed=True, bins=np.linspace(
                minimum_value, maximum_value, 5), opening=0.8, edgecolor='white', cmap=turbo)
            y_mod_original_min[ds.id][exp], y_mod_original_max[ds.id][exp] = ax.get_ylim(
            )

            title_substring = 'Spatial Surface Current Velocity Mean Comparison\n (Rev Interp)'
            name_file_substring = '_mod_obs_ts_comparison_rev_interp'
            mean_vel_no_interp_mod, mean_vel_interp_obs = plot_mod_obs_ts_comparison(spatial_mean_interp_hfr_ts[ds.id][exp], spatial_not_interp_mean_model_ts[ds.id][
                exp], time_res_to_average, ds, date_in, date_fin, path_to_out_plot_folder_arr[exp], timerange, label_plot_arr[exp], title_substring, name_file_substring)
            tot_mean_stat_rev_interp = [
                mean_vel_no_interp_mod, mean_vel_interp_obs]
            plotname = ds.id + '_' + date_in + '_' + date_fin + \
                '_' + time_res_to_average + '_qqPlot_rev_interp.png'
            title = 'Spatial Mean Surface Current Velocity (Rev Interp)' + \
                ds.id + '\n Period: ' + date_in + ' - ' + date_fin
            xlabel = 'Observation Current Velocity [m/s]'
            ylabel = 'Model Current Velocity [m/s]'
            if timerange.shape[0] > 2:
                statistics_array_rev_interp = scatterPlot(np.array(spatial_not_interp_mean_model_ts[ds.id][exp]), np.array(spatial_mean_interp_hfr_ts[ds.id][exp]), path_to_out_plot_folder_arr[exp] + plotname, label_plot_arr[exp], 1, len(
                    spatial_not_interp_mean_model_ts[ds.id][exp]), possible_markers[hfr_counter], splitted_name[1], possible_colors, string_time_res, title=title, xlabel=xlabel, ylabel=ylabel)
                row_stat_rev_interp = tot_mean_stat_rev_interp + statistics_array_rev_interp
                statistics_rev_interp[ds.id][exp] = row_stat_rev_interp
            ciao = np.array(spatial_mean_interp_hfr_ts[ds.id][exp])
            len_not_nan_values_rev = append_value(
                len_not_nan_values_rev, exp, len(ciao[~np.isnan(ciao)]))
            # .append(len(ciao[~np.isnan(ciao)]))

            mod_array_rev_interp = append_value(
                mod_array_rev_interp, exp, spatial_not_interp_mean_model_ts[ds.id][exp])
            obs_array_rev_interp = append_value(
                obs_array_rev_interp, exp, spatial_mean_interp_hfr_ts[ds.id][exp])

        for exp in range(len(name_exp_arr)):
            name_file_substring = "averaged_windrose_rev_" + \
                name_exp_arr[exp]
            title_substring = name_exp_arr[exp] + \
                " Averaged Windrose (rev) for " + ds.id
            a = c_mod_hfr_interpolated[ds.id][exp].ravel()
            b = ((masked_hfr_u_interpolated_arr[ds.id][exp] ** 2 +
                  masked_hfr_v_interpolated_arr[ds.id][exp] ** 2) ** 0.5).ravel()
            plot_windrose(a[~np.isnan(a)], b[~np.isnan(a)], minimum_value, maximum_value, ds, date_in, date_fin, name_file_substring, title_substring, path_to_out_plot_folder_arr[exp], min(
                min(y_mod_hfr_interp_min[ds.id].values()), min(y_mod_original_min[ds.id].values())), max(max(y_mod_hfr_interp_max[ds.id].values()), max(y_mod_original_max[ds.id].values())))

            a = ((masked_hfr_u_interpolated_arr[ds.id][exp] ** 2 +
                  masked_hfr_v_interpolated_arr[ds.id][exp] ** 2) ** 0.5).ravel()
            b = ((masked_subset_u_model[ds.id][exp] ** 2 +
                  masked_subset_v_model[ds.id][exp] ** 2) ** 0.5).ravel()
            a = a[~np.isnan(b)]
            b = b[~np.isnan(b)]
            taylor_stats = sm.taylor_statistics(
                a[~np.isnan(a)], b[~np.isnan(a)])

            if exp == 0:
                sdev_rev[ds.id] = list(
                    np.around(np.array([taylor_stats['sdev'][0], taylor_stats['sdev'][1]]), 4))
                crmsd_rev[ds.id] = list(
                    np.around(np.array([taylor_stats['crmsd'][0], taylor_stats['crmsd'][1]]), 4))
                ccoef_rev[ds.id] = list(
                    np.around(np.array([taylor_stats['ccoef'][0], taylor_stats['ccoef'][1]]), 4))
            else:
                sdev_rev = append_value(sdev_rev, ds.id, round(
                    taylor_stats['sdev'][1], 4))
                crmsd_rev = append_value(crmsd_rev, ds.id, round(
                    taylor_stats['crmsd'][1], 4))
                ccoef_rev = append_value(ccoef_rev, ds.id, round(
                    taylor_stats['ccoef'][1], 4))

            obsSTD = [sdev_rev[ds.id][0]]
            s = sdev_rev[ds.id][1:]
            r = ccoef_rev[ds.id][1:]

            l = label_for_taylor[1:]

            fname = ds.id + '_TaylorDiagram_rev.png'
            srl(obsSTD, s, r, l, fname, markers,
                path_to_out_plot_folder_comparison)

    for exp in range(len(label_plot_arr)):
        unlisted_array_mod = np.array(unlist(mod_array_rev_interp[exp]))
        unlisted_array_obs = np.array(unlist(obs_array_rev_interp[exp]))
        tot_mean_mod_rev_interp = round(np.nanmean(unlisted_array_mod), 2)
        tot_mean_obs_rev_interp = round(np.nanmean(unlisted_array_obs), 2)

        mean_all_rev_interp = [
            tot_mean_mod_rev_interp, tot_mean_obs_rev_interp]

        plotname = date_in + '_' + date_fin + '_' + \
            time_res_to_average + '_qqPlot_rev_interp.png'
        title = 'Surface Current Velocity -ALL (Rev Interp)\n Period: ' + \
            date_in + '-' + date_fin
        xlabel = 'Observation Current Velocity [m/s]'
        ylabel = 'Model Current Velocity [m/s]'
        if timerange.shape[0] > 2:
            statistics_array_rev_interp = scatterPlot(unlisted_array_mod, unlisted_array_obs, path_to_out_plot_folder_arr[exp] + plotname, label_plot_arr[exp], len(
                listOfFiles), timerange.shape[0], possible_markers, hfr_names_rev, possible_colors, string_time_res, len_not_nan_values=len_not_nan_values_rev[exp], title=title, xlabel=xlabel, ylabel=ylabel)
        row_all_rev_interp = mean_all_rev_interp + statistics_array_rev_interp
        statistics_rev_interp["ALL HFR STATIONS"][exp] = row_all_rev_interp

        a_file = open(path_to_out_plot_folder_arr[exp]+"statistics_rev_interp_" +
                      name_exp_arr[exp] + "_" + date_in + "_" + date_fin + ".csv", "w")
        writer = csv.writer(a_file)
        writer.writerow(["name_hfr", "mean_mod", "mean_obs", "bias",
                        "rmse", "si", "corr", "stderr", "number_of_points"])
        for key, value in statistics_rev_interp.items():
            array = [key] + value[exp]
            print(array)
            writer.writerow(array)
        a_file.close()


if __name__ == "__main__":

    args = parse_args()
    main(args)
