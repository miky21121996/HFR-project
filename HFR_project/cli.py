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
        
def parse():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add --link command
    parser.add_argument('--link', action='store_true', help='Link Model Files')
    
    # Parse arguments
    args = parser.parse_args()
    return args