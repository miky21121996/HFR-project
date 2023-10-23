import subprocess
import os
import glob
from datetime import datetime, timedelta
from .cli import parse, Configuration
from .core_backup import Link_Files_mo, Destaggering


def main():
    args = parse()
    config = Configuration(
        '/work/oda/mg28621/HFR_project/HFR_project/configuration.ini')
    if args.link:
        print("sono in link")
        Link_Files_mo(config.paths, config.old_names, config.new_names, config.link_date_in,
                      config.link_date_fin, config.time_res, config.out_paths)
    if args.destaggering:
        Destaggering(config.destag_date_in, config.destag_date_fin, config.input_paths,
                     config.path_to_out_destag_model_folder, config.exp_names, config.destag_time_res, config.mask_paths)
    if args.concat:
        for destag_folder, name, time_res in zip(config.path_to_out_destag_model_folder, config.exp_names, config.destag_time_res):
            combined_file = f'{destag_folder}/{name}_{time_res}_grid_U2T_combined.nc'
            # Check if the output file already exists
            if not os.path.isfile(combined_file):

                file_list = []
                date_format = "%Y%m%d"

                current_date = config.plot_date_in
                while current_date <= config.plot_date_fin:
                    pattern = f"{destag_folder}/{name}_{time_res}_{current_date}_grid_U2T.nc"
                    file_list += glob.glob(pattern)

                    # Increment the current date
                    print(current_date)
                    current_date = (datetime.strptime(
                        current_date, date_format) + timedelta(days=1)).strftime(date_format)
                # Construct the command
                    #pattern = f'{destag_folder}/{name}_{time_res}*_grid_U2T.nc'
                    #file_list = glob.glob(pattern)
                file_list = sorted(file_list, key=os.path.getctime)
                cmd = ['ncrcat'] + file_list + [combined_file]
            #cmd = ['ncrcat',f'{destag_folder}/{name}_{time_res}*_grid_U2T.nc',out_file]

            # Execute the command
                #subprocess.run(cmd, check=True)

            combined_file = f'{destag_folder}/{name}_{time_res}_grid_V2T_combined.nc'

            if not os.path.isfile(combined_file):
                file_list = []
                date_format = "%Y%m%d"

                current_date = config.plot_date_in
                while current_date <= config.plot_date_fin:
                    pattern = f"{destag_folder}/{name}_{time_res}_{current_date}_grid_V2T.nc"
                    file_list += glob.glob(pattern)

                    # Increment the current date
                    current_date = (datetime.strptime(
                        current_date, date_format) + timedelta(days=1)).strftime(date_format)
                # Construct the command
                #pattern = f'{destag_folder}/{name}_{time_res}*_grid_V2T.nc'
                #file_list = glob.glob(pattern)
                file_list = sorted(file_list, key=os.path.getctime)
                cmd = ['ncrcat'] + file_list + [combined_file]
                #cmd = ['ncrcat',f'{destag_folder}/{name}_{time_res}*_grid_U2T.nc',out_file]

                # Execute the command
                subprocess.run(cmd, check=True)

    if args.plot_stats:
        permissive_memory_gb = 10
        if config.interpolation == "MTO":
            subprocess.run(['bsub', '-n', '1', '-q', 's_long', '-J', 'CURVAL', '-e', 'aderr_0', '-o', 'adout_0', '-P',
                            '0510', '-R', f'rusage[mem={permissive_memory_gb}G]', 'python', 'hfr_validation_MTO.py', config.plot_date_in, config.plot_date_fin, config.path_to_hfr_folder,
                            config.time_res_to_average, " ".join(config.grid_arr), config.interpolation, " ".join(
                                config.mesh_mask_arr), " ".join(config.u_combined_arr), " ".join(config.v_combined_arr),
                            " ".join(config.name_exp_arr), " ".join(config.label_plot_arr), " ".join(config.path_to_out_plot_folder_arr), config.path_to_out_plot_folder_comparison])
        elif config.interpolation == "OTM":
            subprocess.run(['bsub', '-n', '1', '-q', 's_long', '-J', 'CURVAL', '-e', 'aderr_0', '-o', 'adout_0', '-P',
                            '0510', '-R', f'rusage[mem={permissive_memory_gb}G]', 'python', 'hfr_validation_OTM.py', config.plot_date_in, config.plot_date_fin, config.path_to_hfr_folder,
                            config.time_res_to_average, " ".join(config.grid_arr), config.interpolation, " ".join(
                                config.mesh_mask_arr), " ".join(config.u_combined_arr), " ".join(config.v_combined_arr),
                            " ".join(config.name_exp_arr), " ".join(config.label_plot_arr), " ".join(config.path_to_out_plot_folder_arr), config.path_to_out_plot_folder_comparison])
        else:
            sys.exit(1)


if __name__ == '__main__':
    main()
