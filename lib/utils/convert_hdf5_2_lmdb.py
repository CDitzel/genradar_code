import time
import argparse
# import yaml
import os
from datetime import datetime
from shutil import copyfile
from pathlib import Path

from lib import load_config
from lib import InrasData, setup_logging

def scrape_files(top_level_path, suffix, recursive=True):
    path = Path(top_level_path)
    print(f"Searching for {suffix} files in", path)
    assert path.is_dir(), f"select a directory containing {suffix} files"
    if recursive:
        file_paths = sorted(path.glob(f"**/*.{suffix}"))
    else:
        file_paths = sorted(path.glob(f"*.{suffix}"))
    if not file_paths:
        raise RuntimeError(f"No {suffix} files found")
    return list(str(path.resolve()) for path in file_paths)


# Get optional description string that is part of the destination path
parser = argparse.ArgumentParser(description="Convert hdf5 data into LMDB")
parser.add_argument("--cfg", "-c", default="/home/ditzel/rad/lib/config.yaml")
parser.add_argument("--desc", "-d", default=str(datetime.now())[:16].replace(" ", "_").strip('"'))
args = parser.parse_args()

cfg, _, _ = load_config(args.cfg)
setup_logging(console_log_level=cfg.proc['log_level'])

#  dump the config as yaml to the lmdb output directory
# copyfile(args.cfg, cfg.proc.args["dst_dir"])

start = time.time()

# print(cfg.proc.args)
h5_files = scrape_files(cfg.proc.args['src_dir'], suffix='h5', recursive=cfg.proc.args['recursive'])
# print(len(h5_files))


for file_idx, h5_file in enumerate(h5_files):
    print(f'Converting file {file_idx}/{len(h5_files)}')
    cfg.proc.args['src_dir'] = h5_file
    InrasData.writer(**cfg.proc.args, desc=args.desc)
exit()


df = InrasData.writer(**cfg.proc.args, desc=args.desc)
path, stem = os.path.split(cfg.proc.args["dst_dir"])
# copyfile(args.cfg, path + f"/{stem}_{len(df)}_samples_{args.desc}.yaml")
copyfile(args.cfg, path + f"/{stem}_{len(df)}_samples.yaml")
stop = time.time()
print("Duration", stop - start)
