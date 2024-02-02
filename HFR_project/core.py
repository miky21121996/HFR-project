import os
import xarray as xr
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta, date
from netCDF4 import Dataset
from collections import OrderedDict
import numpy.ma as ma
from scipy import interpolate
import copy


def Link_Files(paths, old_names, new_names, date_in, date_fin, time_res, out_paths):
    
    current_date = date_in
    for path in out_paths:
        os.makedirs(path, exist_ok=True)
        
    # Remove all files in the directory if it already exists
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    while current_date <= date_fin:
        print("date: ", current_date)
        # Loop through paths
        for i, path in enumerate(paths):
            # Find files
            for root, dirs, files in os.walk(path):
                for file in files:
                    u_file = None
                    v_file = None
                    if old_names[i] in file and time_res[i] in file and current_date.strftime('%Y%m%d') in file:
                        print(file)
                        if 'U' in file:
                            u_file = os.path.join(root, file)
                        elif 'V' in file:
                            v_file = os.path.join(root, file)

                    # Link files to output folder
                        if u_file is not None and v_file is not None:
                            os.symlink(u_file, os.path.join(
                                out_paths[i], f"{new_names[i]}_{time_res[i]}_{current_date.strftime('%Y%m%d')}_grid_U.nc"))
                            os.symlink(v_file, os.path.join(
                                out_paths[i], f"{new_names[i]}_{time_res[i]}_{current_date.strftime('%Y%m%d')}_grid_V.nc"))
                        elif u_file is not None and 'U' in u_file:
                            os.symlink(u_file, os.path.join(
                                out_paths[i], f"{new_names[i]}_{time_res[i]}_{current_date.strftime('%Y%m%d')}_grid_U.nc"))
                        elif v_file is not None and 'V' in v_file:
                            os.symlink(v_file, os.path.join(
                                out_paths[i], f"{new_names[i]}_{time_res[i]}_{current_date.strftime('%Y%m%d')}_grid_V.nc"))

        # Increment date
        current_date += timedelta(days=1)


def daterange(start_date, end_date, resolution):
    if resolution == 'd':
        for n in range(int((end_date - start_date).days)+1):
            yield start_date + timedelta(n)
    elif resolution == 'h':
        for n in range(int((end_date - start_date).total_seconds()//3600)+1):
            yield start_date + timedelta(hours=n)
    elif resolution == 'm':
        for n in range(int((end_date - start_date).total_seconds()//60)+1):
            yield start_date + timedelta(minutes=n)
    elif resolution == 's':
        for n in range(int((end_date - start_date).total_seconds())+1):
            yield start_date + timedelta(seconds=n)
    else:
        raise ValueError("Invalid resolution")


def Destaggering(date_in, date_fin, path_to_mod_output_arr, path_to_destag_output_folder_arr, name_exp, time_res, path_to_mask_arr):

    start_date = date(int(date_in[0:4]), int(date_in[4:6]), int(date_in[6:8]))
    end_date = date(int(date_fin[0:4]), int(date_fin[4:6]), int(date_fin[6:8]))

    for i, (path_to_mod_output, path_to_destag_output_folder, path_to_mask) in enumerate(zip(path_to_mod_output_arr, path_to_destag_output_folder_arr, path_to_mask_arr)):

        os.makedirs(path_to_destag_output_folder, exist_ok=True)

        listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(path_to_mod_output, followlinks=True):
            print("dirpath: ", dirpath)
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        mesh_mask_ds = Dataset(path_to_mask)
        mesh_mask = xr.open_dataset(xr.backends.NetCDF4DataStore(mesh_mask_ds))

        u_mask = mesh_mask.umask.values
        v_mask = mesh_mask.vmask.values
        t_mask = mesh_mask.tmask.values
        
        u_mask = np.squeeze(u_mask[:, 0, :, :])
        v_mask = np.squeeze(v_mask[:, 0, :, :])
        t_mask = np.squeeze(t_mask[:, 0, :, :])

        for single_date in daterange(start_date, end_date, time_res[i][-1]):
            
            print("date: ", single_date)
            timetag = single_date.strftime("%Y%m%d")

            u_filename = name_exp[i] + "_" + \
                time_res[i] + "_" + timetag + "_grid_U.nc"
            v_filename = name_exp[i] + "_" + \
                time_res[i] + "_" + timetag + "_grid_V.nc"

            if any(u_filename in s for s in listOfFiles) and any(v_filename in r for r in listOfFiles):
                matching_u = [
                    u_match for u_match in listOfFiles if u_filename in u_match]
                matching_v = [
                    v_match for v_match in listOfFiles if v_filename in v_match]

                U_current = xr.open_dataset(
                    listOfFiles[listOfFiles.index(matching_u[0])])
                V_current = xr.open_dataset(
                    listOfFiles[listOfFiles.index(matching_v[0])])
            else:
                print("model current file (U or V or both) not found.\n Go to next time")
                continue

            if time_res[i] == "1d":
                [dim_t, _, dim_lat, dim_lon] = U_current.vozocrtx.shape

                u_int = U_current.vozocrtx.values
                u = u_int[:, 0, :, :]

                v_int = V_current.vomecrty.values
                v = v_int[:, 0, :, :]

##TO DO: function definition
                # destaggering of u
                sum_u_mask = u_mask[:, 1:]+u_mask[:, :(dim_lon-1)]
                sum_u_mask = np.repeat(sum_u_mask[np.newaxis, :, :], dim_t, axis=0)
                sum_u = u[:, :, 1:]+u[:, :, :(dim_lon-1)]
                denominator_u_mask = np.maximum(sum_u_mask, 1)
                destaggered_u = np.zeros(u.shape)
                destaggered_u[:, :, 1:] = sum_u / denominator_u_mask
                destaggered_u = destaggered_u * t_mask

                # destaggering of v
                sum_v_mask = v_mask[1:, :]+v_mask[:(dim_lat-1), :]
                sum_v_mask = np.repeat(sum_v_mask[np.newaxis, :, :], dim_t, axis=0)
                sum_v = v[:, 1:, :]+v[:, :(dim_lat-1), :]
                denominator_v_mask = np.maximum(sum_v_mask, 1)
                destaggered_v = np.zeros(v.shape)
                destaggered_v[:, 1:, :] = sum_v / denominator_v_mask
                destaggered_v = destaggered_v*t_mask

##TO DO: def function for saving destaggered files
                # save destaggered u in nc file
                destaggered_U_current = U_current
                if 'nav_lat' in list(destaggered_U_current.keys()):
                    destaggered_U_current = destaggered_U_current.drop(
                        ("nav_lat"))
                if 'nav_lon' in list(destaggered_U_current.keys()):
                    destaggered_U_current = destaggered_U_current.drop(
                        ("nav_lon"))

                destaggered_U_current = destaggered_U_current.assign(
                    destaggered_u=(('time_counter', 'y', 'x'), destaggered_u))
                destaggered_U_current = destaggered_U_current.assign(
                    nav_lon=(('y', 'x'), mesh_mask.glamt.values[0, :, :]))
                destaggered_U_current = destaggered_U_current.assign(
                    nav_lat=(('y', 'x'), mesh_mask.gphit.values[0, :, :]))
                destaggered_U_current.destaggered_u.attrs = U_current.vozocrtx.attrs

                destaggered_U_current = destaggered_U_current.drop(
                                        ("vozocrtx"))

                destaggered_U_current.to_netcdf(
                    path_to_destag_output_folder + name_exp[i] + "_" + time_res[i] + "_" + timetag + "_grid_U2T.nc")

                # save destaggered v in nc file
                destaggered_V_current = V_current
                if 'nav_lat' in list(destaggered_V_current.keys()):
                    destaggered_V_current = destaggered_V_current.drop(
                        ("nav_lat"))
                if 'nav_lon' in list(destaggered_V_current.keys()):
                    destaggered_U_current = destaggered_V_current.drop(
                        ("nav_lon"))

                destaggered_V_current = destaggered_V_current.assign(
                    destaggered_v=(('time_counter', 'y', 'x'), destaggered_v))
                destaggered_V_current = destaggered_V_current.assign(
                    nav_lon=(('y', 'x'), mesh_mask.glamt.values[0, :, :]))
                destaggered_V_current = destaggered_V_current.assign(
                    nav_lat=(('y', 'x'), mesh_mask.gphit.values[0, :, :]))

                destaggered_V_current.destaggered_v.attrs = V_current.vomecrty.attrs
                
                destaggered_V_current = destaggered_V_current.drop(
                                        ("vomecrty"))

                destaggered_V_current.to_netcdf(
                    path_to_destag_output_folder + name_exp[i] + "_" + time_res[i] + "_" + timetag + "_grid_V2T.nc")

            elif time_res == "1h":
                [dim_t, dim_lat, dim_lon] = U_current.ssu.shape

                u_int = U_current.ssu.values
                v_int = V_current.ssv.values

##TO DO: functions to be defined
                # destaggering of u
                sum_u_mask = u_mask[:, 1:]+u_mask[:, :(dim_lon-1)]
                sum_u_mask = np.repeat(
                    sum_u_mask[np.newaxis, :, :], dim_t, axis=0)
                sum_u = u[:, :, 1:]+u[:, :, :(dim_lon-1)]
                denominator_u_mask = np.maximum(sum_u_mask, 1)
                destaggered_u = np.zeros(u.shape)
                destaggered_u[:, :, 1:] = sum_u / denominator_u_mask
                destaggered_u = destaggered_u * t_mask

                # destaggering of v
                sum_v_mask = v_mask[1:, :]+v_mask[:(dim_lat-1), :]
                sum_v_mask = np.repeat(
                    sum_v_mask[np.newaxis, :, :], dim_t, axis=0)
                sum_v = v[:, 1:, :]+v[:, :(dim_lat-1), :]
                denominator_v_mask = np.maximum(sum_v_mask, 1)
                destaggered_v = np.zeros(v.shape)
                destaggered_v[:, 1:, :] = sum_v / denominator_v_mask
                destaggered_v = destaggered_v*t_mask

                destaggered_U_current = U_current
                if 'nav_lat' in list(destaggered_U_current.keys()):
                    destaggered_U_current = destaggered_U_current.drop(
                        ("nav_lat"))
                if 'nav_lon' in list(destaggered_U_current.keys()):
                    destaggered_U_current = destaggered_U_current.drop(
                        ("nav_lon"))

                destaggered_U_current = destaggered_U_current.assign(
                    destaggered_u=(('time_counter', 'y', 'x'), destaggered_u))
                destaggered_U_current = destaggered_U_current.assign(
                    nav_lon=(('y', 'x'), mesh_mask.glamt.values[0, :, :]))
                destaggered_U_current = destaggered_U_current.assign(
                    nav_lat=(('y', 'x'), mesh_mask.gphit.values[0, :, :]))
                destaggered_U_current.destaggered_u.attrs = U_current.ssu.attrs
                
                destaggered_U_current = destaggered_U_current.drop(
                        ("ssu"))

                destaggered_U_current.to_netcdf(
                    path_to_destag_output_folder + name_exp[i] + "_" + time_res[i] + "_" + timetag + "_grid_U2T.nc")

                # save destaggered v in nc file
                destaggered_V_current = V_current
                if 'nav_lat' in list(destaggered_V_current.keys()):
                    destaggered_V_current = destaggered_V_current.drop(
                        ("nav_lat"))
                if 'nav_lon' in list(destaggered_V_current.keys()):
                    destaggered_U_current = destaggered_V_current.drop(
                        ("nav_lon"))

                destaggered_V_current = destaggered_V_current.assign(
                    destaggered_v=(('time_counter', 'y', 'x'), destaggered_v))
                destaggered_V_current = destaggered_V_current.assign(
                    nav_lon=(('y', 'x'), mesh_mask.glamt.values[0, :, :]))
                destaggered_V_current = destaggered_V_current.assign(
                    nav_lat=(('y', 'x'), mesh_mask.gphit.values[0, :, :]))

                destaggered_V_current.destaggered_v.attrs = V_current.ssv.attrs
                destaggered_V_current = destaggered_V_current.drop(
                        ("ssv"))
                destaggered_V_current.to_netcdf(
                    path_to_destag_output_folder + name_exp[i] + "_" + time_res[i] + "_" + timetag + "_grid_V2T.nc")


def Get_List_Of_Files(path_to_hfr_files):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(path_to_hfr_files, followlinks=True):
        print("dirpath: ", dirpath)
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    print(listOfFiles)
    return listOfFiles


def Get_String_Time_Resolution(start_date, end_date, time_res_to_average):
    dates = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
    start, end = [datetime.datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    if time_res_to_average[-1] == 'D':
        string_time_res = list(OrderedDict(((start + timedelta(_)).strftime(
            r"%d-%b-%y"), None) for _ in range((end - start).days+1)).keys())
    if time_res_to_average[-1] == 'M':
        string_time_res = list(OrderedDict(
            ((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days+1)).keys())

    return string_time_res


def getSourceAntennas(ds):
    
    data = {'antennas': [], 'bbox': []}
    try:
        for name, lat, lon in zip(ds['SCDT'][0, :].astype('str').data, ds['SLTT'][0, :].data, ds['SLNT'][0, :].data):
            if np.isnan(lat) == False:
                name, lat, lon = name.strip(), round(lat, 4), round(lon, 4)
                data['antennas'].append({'name': name, 'lat': lat, 'lon': lon})
    except Exception as e:
        print('An error ocurred when checking antennas')
    data['bbox'] = [float(ds.geospatial_lon_min), float(ds.geospatial_lon_max), float(
        ds.geospatial_lat_min), float(ds.geospatial_lat_max)]
    return data


def Get_Closest_Hfr_Time_Range_Index(time_res_to_average, ini_date, fin_date, averaged_ds):
    if time_res_to_average[-1] == 'D':
        timestamp_start = ini_date[0:4]+'-'+ini_date[4:6]+'-'+ini_date[6:8]
        timestamp_end = fin_date[0:4]+'-'+fin_date[4:6]+'-'+fin_date[6:8]
        datetime_obj1 = datetime.datetime.strptime(timestamp_start, '%Y-%m-%d')
        datetime_obj2 = datetime.datetime.strptime(timestamp_end, '%Y-%m-%d')

    if time_res_to_average[-1] == 'M':
        timestamp_start = ini_date[0:4]+'-'+ini_date[4:6]
        timestamp_end = fin_date[0:4]+'-'+fin_date[4:6]
        datetime_obj1 = datetime.datetime.strptime(timestamp_start, '%Y-%m')
        datetime_obj2 = datetime.datetime.strptime(timestamp_end, '%Y-%m')

    print(f"HF time instants: {averaged_ds['TIME']}")
    closerval1 = averaged_ds['TIME'].sel(TIME=datetime_obj1, method="backfill")
    idx1 = averaged_ds['TIME'].astype(
        str).values.tolist().index(str(closerval1.data))
    print(f"nearest start time instant: {averaged_ds['TIME'][idx1]}")
    closerval2 = averaged_ds['TIME'].sel(TIME=datetime_obj2, method="backfill")
    idx2 = averaged_ds['TIME'].astype(
        str).values.tolist().index(str(closerval2.data))
    print(f"nearest end time instant: {averaged_ds['TIME'][idx2]}")
    return idx1, idx2, closerval1, closerval2

def find_date_indices(dataset, start_date, final_date, time_resolution):
    start_bool_value = True
    final_bool_value = True
    if time_resolution == 'D':
        start_date = pd.to_datetime(start_date, format='%Y%m%d').date()
        final_date = pd.to_datetime(final_date, format='%Y%m%d').date()
        time_values = dataset['TIME'].values.astype('datetime64[D]')
    elif time_resolution == 'M':
        start_date = pd.to_datetime(start_date, format='%Y%m%d').to_period('M').start_time.strftime('%Y-%m')
        final_date = pd.to_datetime(final_date, format='%Y%m%d').to_period('M').end_time.strftime('%Y-%m')
        time_values = dataset['TIME'].values.astype('datetime64[M]')
    else:
        raise ValueError("Unsupported time resolution: {}".format(time_resolution))

    # Find the index of the first time instant of the start_date
    start_idx = np.where(time_values == np.datetime64(start_date))[0]
    if start_idx.size == 0:
        print("Start date not found in the TIME variable.")
        start_bool_value = False
    else:
        start_idx = start_idx[0]

    # Find the index of the last time instant of the final_date
    final_idx = np.where(time_values == np.datetime64(final_date))[0]
    if final_idx.size == 0:
        print("Final date not found in the TIME variable.")
        final_bool_value = False
    else:
        final_idx = final_idx[-1]

    return start_idx, final_idx, start_bool_value, final_bool_value


def find_nearest(array, value, pprint=False):
    array = np.asarray(array)
    if pprint:
        print("model: ", array)
        print("obs: ", value)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# depth is to select the number of consequential mask points to fill
def seaoverland(input_matrix, depth=1):
    # depth loop
    for d in range(depth):
        if np.sum(input_matrix.mask) == 0:  # nothing to fill
            return input_matrix
        else:
            # Create a m x n x 8 3D matrix in which, third dimension fixed, the other dimensions
            #  contains values that are shifted in one of the 8 possible direction compared to the original matrix
            shift_matrix = ma.array(np.empty(shape=(input_matrix.shape[0], input_matrix.shape[1], 8)),
                                    mask=True, fill_value=1.e20, dtype=float)
            # up shift
            shift_matrix[: - 1, :, 0] = input_matrix[1:, :]
            # down shift
            shift_matrix[1:, :, 1] = input_matrix[0: - 1, :]
            # left shift
            shift_matrix[:, : - 1, 2] = input_matrix[:, 1:]
            # right shift
            shift_matrix[:, 1:, 3] = input_matrix[:, : - 1]
            # up-left shift
            shift_matrix[: - 1, : - 1, 4] = input_matrix[1:, 1:]
            # up-right shift
            shift_matrix[: - 1, 1:, 5] = input_matrix[1:, : - 1]
            # down-left shift
            shift_matrix[1:, : - 1, 6] = input_matrix[: - 1, 1:]
            # down-right shift
            shift_matrix[1:, 1:, 7] = input_matrix[: - 1, : - 1]
            # Mediate the shift matrix among the third dimension
            mean_matrix = ma.mean(shift_matrix, 2)
            # Replace input missing values with new ones belonging to the mean matrix
            input_matrix = ma.array(np.where(mean_matrix.mask + input_matrix.mask, mean_matrix, input_matrix),
                                    mask=mean_matrix.mask, fill_value=1.e20, dtype=float)
            input_matrix = ma.masked_where(mean_matrix.mask, input_matrix)
    return input_matrix


def Get_Max_Min_Interpolated_Model(idx1, idx2, averaged_ds, masked_subset_speed_model, x_subset_model, y_subset_model, lon_hfr, lat_hfr):
    min_value = 0
    max_value = 0

    min_value_rev = 0
    max_value_rev = 0

    min_obs_rev = 0
    max_obs_rev = 0
    
    for time_counter, index in enumerate(range(idx1, idx2+1)):
        U = averaged_ds['EWCT'][index, 0].data
        V = averaged_ds['NSCT'][index, 0].data
        speed_hfr = (U ** 2 + V ** 2) ** 0.5
        mask_hfr = np.ma.masked_invalid(speed_hfr).mask

        subset_speed_model_instant = seaoverland(
            masked_subset_speed_model[time_counter], 3)

        f = interpolate.interp2d(
            x_subset_model, y_subset_model, subset_speed_model_instant)
        speed_interpolated = f(lon_hfr, lat_hfr)
        masked_speed_interpolated = ma.masked_array(
            speed_interpolated, mask=mask_hfr)
        min_interpolated_subset_model = np.nanmin(
            masked_speed_interpolated.data)
        min_value = min(min_value, min_interpolated_subset_model)
        max_interpolated_subset_model = np.nanmax(
            masked_speed_interpolated.data)
        max_value = max(max_value, max_interpolated_subset_model)

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

        masked_subset_speed_model_instant = ma.masked_array(
            subset_speed_model_instant, mask=hfr_mask_interpolated)

        min_value_rev = min(min_value_rev, np.nanmin(
            masked_subset_speed_model_instant.data))
        max_value_rev = max(max_value_rev, np.nanmax(
            masked_subset_speed_model_instant.data))

        masked_U = ma.masked_array(U, mask=mask_hfr)
        masked_V = ma.masked_array(V, mask=mask_hfr)
        masked_speed_hfr = ma.masked_array(speed_hfr, mask=mask_hfr)
        sol_speed_hfr = seaoverland(masked_speed_hfr, 3)
        sol_u_hfr = seaoverland(masked_U, 3)
        sol_v_hfr = seaoverland(masked_V, 3)

        masked_hfr_speed_interpolated, *_ = interp_obs_to_mod(
            lon_hfr, lat_hfr, sol_speed_hfr, sol_u_hfr, sol_v_hfr, x_subset_model, y_subset_model, hfr_mask_interpolated)

        min_obs_rev = min(min_obs_rev, np.nanmin(
            masked_hfr_speed_interpolated.data))
        max_obs_rev = max(min_obs_rev, np.nanmax(
            masked_hfr_speed_interpolated.data))

    return min_value, max_value, min_value_rev, max_value_rev, min_obs_rev, max_obs_rev


def Get_Max_Min_Bias(idx1, idx2, averaged_ds, masked_subset_speed_model, x_subset_model, y_subset_model, lon_hfr, lat_hfr):
    min_value = 0.0
    max_value = 0.0

    min_value_rev = 0.0
    max_value_rev = 0.0
    for time_counter, index in enumerate(range(idx1, idx2+1)):
        U = averaged_ds['EWCT'][index, 0].data
        V = averaged_ds['NSCT'][index, 0].data
        speed_hfr = (U ** 2 + V ** 2) ** 0.5
        mask_hfr = np.ma.masked_invalid(speed_hfr).mask

        subset_speed_model_instant = seaoverland(
            masked_subset_speed_model[time_counter], 3)

        f = interpolate.interp2d(
            x_subset_model, y_subset_model, subset_speed_model_instant)
        speed_interpolated = f(lon_hfr, lat_hfr)
        masked_speed_interpolated = ma.masked_array(
            speed_interpolated, mask=mask_hfr)
        min_bias = np.nanmin(masked_speed_interpolated.data-speed_hfr.data)
        min_value = min(min_value, min_bias)
        max_bias = np.nanmax(masked_speed_interpolated.data-speed_hfr.data)
        max_value = max(max_value, max_bias)

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

        masked_subset_speed_model_instant = ma.masked_array(
            subset_speed_model_instant, mask=hfr_mask_interpolated)

        masked_U = ma.masked_array(U, mask=mask_hfr)
        masked_V = ma.masked_array(V, mask=mask_hfr)
        masked_speed_hfr = ma.masked_array(speed_hfr, mask=mask_hfr)
        sol_speed_hfr = seaoverland(masked_speed_hfr, 3)
        sol_u_hfr = seaoverland(masked_U, 3)
        sol_v_hfr = seaoverland(masked_V, 3)

        masked_hfr_speed_interpolated, *_ = interp_obs_to_mod(
            lon_hfr, lat_hfr, sol_speed_hfr, sol_u_hfr, sol_v_hfr, x_subset_model, y_subset_model, hfr_mask_interpolated)

        min_bias = np.nanmin(masked_hfr_speed_interpolated.data -
                             masked_subset_speed_model_instant.data)
        min_value_rev = min(min_value_rev, min_bias)

        max_bias = np.nanmax(masked_hfr_speed_interpolated.data -
                             masked_subset_speed_model_instant.data)
        max_value_rev = max(max_value_rev, max_bias)

    return min_value, max_value, min_value_rev, max_value_rev


def Get_Max_Min_Rmsd(idx1, idx2, averaged_ds, masked_subset_speed_model, x_subset_model, y_subset_model, lon_hfr, lat_hfr):
    min_value = 0.0
    max_value = 0.0

    min_value_rev = 0.0
    max_value_rev = 0.0
    for time_counter, index in enumerate(range(idx1, idx2+1)):
        U = averaged_ds['EWCT'][index, 0].data
        V = averaged_ds['NSCT'][index, 0].data
        speed_hfr = (U ** 2 + V ** 2) ** 0.5
        mask_hfr = np.ma.masked_invalid(speed_hfr).mask

        subset_speed_model_instant = seaoverland(
            masked_subset_speed_model[time_counter], 3)

        f = interpolate.interp2d(
            x_subset_model, y_subset_model, subset_speed_model_instant)
        speed_interpolated = f(lon_hfr, lat_hfr)
        masked_speed_interpolated = ma.masked_array(
            speed_interpolated, mask=mask_hfr)
        min_rmsd = np.nanmin(
            np.sqrt((masked_speed_interpolated.data-speed_hfr.data)**2))
        min_value = min(min_value, min_rmsd)
        max_rmsd = np.nanmax(
            np.sqrt((masked_speed_interpolated.data-speed_hfr.data)**2))
        max_value = max(max_value, max_rmsd)

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

        masked_subset_speed_model_instant = ma.masked_array(
            subset_speed_model_instant, mask=hfr_mask_interpolated)

        masked_U = ma.masked_array(U, mask=mask_hfr)
        masked_V = ma.masked_array(V, mask=mask_hfr)
        masked_speed_hfr = ma.masked_array(speed_hfr, mask=mask_hfr)
        sol_speed_hfr = seaoverland(masked_speed_hfr, 3)
        sol_u_hfr = seaoverland(masked_U, 3)
        sol_v_hfr = seaoverland(masked_V, 3)

        masked_hfr_speed_interpolated, *_ = interp_obs_to_mod(
            lon_hfr, lat_hfr, sol_speed_hfr, sol_u_hfr, sol_v_hfr, x_subset_model, y_subset_model, hfr_mask_interpolated)

        min_rmsd = np.nanmin(np.sqrt(
            (masked_hfr_speed_interpolated.data-masked_subset_speed_model_instant.data)**2))
        min_value_rev = min(min_value_rev, min_rmsd)

        max_rmsd = np.nanmax(np.sqrt(
            (masked_hfr_speed_interpolated.data-masked_subset_speed_model_instant.data)**2))
        max_value_rev = max(max_value_rev, max_rmsd)

    return min_value, max_value, min_value_rev, max_value_rev


def interp_mod_to_obs(x_mod, y_mod, speed_model, u_model, v_model, lon_obs, lat_obs, mask_obs):

    f = interpolate.interp2d(x_mod, y_mod, speed_model)
    speed_interpolated = f(lon_obs, lat_obs)
    masked_speed_interpolated = ma.masked_array(
        speed_interpolated, mask=mask_obs)
    spatial_mean_model_ts = masked_speed_interpolated.mean()

    f = interpolate.interp2d(x_mod, y_mod, u_model)
    u_interpolated = f(lon_obs, lat_obs)
    masked_u_interpolated = ma.masked_array(u_interpolated, mask=mask_obs)

    f = interpolate.interp2d(x_mod, y_mod, v_model)
    v_interpolated = f(lon_obs, lat_obs)
    masked_v_interpolated = ma.masked_array(v_interpolated, mask=mask_obs)

    return masked_speed_interpolated, masked_u_interpolated, masked_v_interpolated, spatial_mean_model_ts


def interp_hfr_mask_to_mod_mask(x_obs, y_obs, mask_hfr, x_model, y_model, threshold):

    f = interpolate.interp2d(x_obs, y_obs, mask_hfr)
    hfr_mask_interpolated = f(x_model, y_model)
    hfr_mask_interpolated[hfr_mask_interpolated < threshold] = 0
    hfr_mask_interpolated[hfr_mask_interpolated > threshold] = 1
    return hfr_mask_interpolated


def interp_obs_to_mod(lon_obs, lat_obs, sol_speed_hfr, sol_u_hfr, sol_v_hfr, x_model, y_model, mask_mod):

    f = interpolate.interp2d(lon_obs, lat_obs, sol_speed_hfr)
    speed_interpolated = f(x_model, y_model)
    masked_speed_interpolated = ma.masked_array(
        speed_interpolated, mask=mask_mod)
    spatial_mean_hfr_ts = masked_speed_interpolated.mean()

    f = interpolate.interp2d(lon_obs, lat_obs, sol_u_hfr)
    u_interpolated = f(x_model, y_model)
    masked_u_interpolated = ma.masked_array(u_interpolated, mask=mask_mod)

    f = interpolate.interp2d(lon_obs, lat_obs, sol_v_hfr)
    v_interpolated = f(x_model, y_model)
    masked_v_interpolated = ma.masked_array(v_interpolated, mask=mask_mod)

    return masked_speed_interpolated, masked_u_interpolated, masked_v_interpolated, spatial_mean_hfr_ts


def wind_direction(x, y):

    # Compute the angle in radians
    angle_rad = np.arctan2(y, x)

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    # Adjust the angle so that east is 0 degrees
    # and angles increase counterclockwise
    angle_deg = 90 - angle_deg

    # Ensure the angle is in the range [0, 360)
    angle_deg = angle_deg % 360
    return angle_deg


def append_value_ex(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        print("gia esistente: ", key)
        print(value)
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        print("non esistente: ", key)
        print(value)
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value

    print(dict_obj)


def append_value(dict_obj, key, value):
    dict_obj_copy = copy.deepcopy(dict_obj)  # Create a deep copy of dict_obj
    print(dict_obj_copy)
    if key in dict_obj_copy:
        if not isinstance(dict_obj_copy[key], list):
            dict_obj_copy[key] = [dict_obj_copy[key]]
        dict_obj_copy[key].append(value)
        print("appeso: ", dict_obj_copy[key])
    else:
        dict_obj_copy[key] = value
    print(dict_obj_copy)
    return dict_obj_copy


def unlist(nested_list):
    return [item for sublist in nested_list for item in (unlist(sublist) if isinstance(sublist, list) else [sublist])]