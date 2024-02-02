import argparse
import configparser
import datetime

class Configuration:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        
        self.paths = config['link_section']['path_to_mod_files'].split(',')
        self.old_names = config['link_section']['old_name_exp'].split(',')
        self.new_names = config['link_section']['new_name_exp'].split(',')
        self.link_date_in = datetime.datetime.strptime(
            config['link_section']['date_in'], '%Y%m%d').date()
        self.link_date_fin = datetime.datetime.strptime(
            config['link_section']['date_fin'], '%Y%m%d').date()
        self.time_res = config['link_section']['time_res_model'].split(',')
        self.out_paths = config['link_section']['path_to_out_model_folder'].split(',')

        self.destag_date_in = config['destaggering_section']['date_in']
        self.destag_date_fin = config['destaggering_section']['date_fin']
        self.input_paths = config['destaggering_section']['path_to_input_model_folder'].split(',')
        self.mask_paths = config['destaggering_section']['path_to_mask'].split(',')
        self.destag_time_res = config['destaggering_section']['time_res_input_model'].split(',')
        self.exp_names = config['destaggering_section']['name_exp'].split(',')
        self.path_to_out_destag_model_folder = config['destaggering_section']['path_to_out_destag_model_folder'].split(',')
        
        self.plot_date_in = config['plot_statistics_section']['date_in']
        self.plot_date_fin = config['plot_statistics_section']['date_fin']
        self.path_to_hfr_folder = config['plot_statistics_section']['path_to_hfr_files']
        self.time_res_to_average = config['plot_statistics_section']['time_res_to_average']
        self.grid_arr = config['plot_statistics_section']['grid'].split(',')
        self.interpolation = config['plot_statistics_section']['interp']
        self.mesh_mask_arr = config['plot_statistics_section']['mesh_mask'].split(',')
        self.u_combined_arr = config['plot_statistics_section']['u_combined'].split(',')
        self.v_combined_arr = config['plot_statistics_section']['v_combined'].split(',')
        self.name_exp_arr = config['destaggering_section']['name_exp'].split(',')
        self.label_plot_arr = config['plot_statistics_section']['label_plot'].split(',')
        self.path_to_out_plot_folder_arr = config['plot_statistics_section']['path_to_out_plot_folder'].split(',')
        self.path_to_out_plot_folder_comparison = config['plot_statistics_section']['path_to_out_plot_folder_comparison']
        
        
        
def parse():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add --link command
    parser.add_argument('--link', action='store_true', help='Link Model Files')
    
    # Add --destaggering command
    parser.add_argument('--destaggering', action='store_true', help='Destag Model Files')
    
    # Add --destaggering command
    parser.add_argument('--concat', action='store_true', help='Destag Model Files')

    # Add --plot_stats command
    parser.add_argument('--plot_stats', action='store_true', help='Plot Statistics') 
    
    # Parse arguments
    args = parser.parse_args()
    return args