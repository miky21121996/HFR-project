import os
import datetime

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
        print(current_date)
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
        current_date += datetime.timedelta(days=1)