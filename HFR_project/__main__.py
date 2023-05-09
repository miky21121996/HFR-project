import subprocess
from .cli import parse, Configuration
from .core import Link_Files

def main():
    args = parse()
    config = Configuration('/work/oda/mg28621/MO_project/MO_project/configuration.ini')
    if args.link:
        Link_Files(config.paths, config.old_names, config.new_names, config.link_date_in,
                config.link_date_fin, config.time_res, config.out_paths)
