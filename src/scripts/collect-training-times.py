import os
import re
import argparse
import numpy as np
from colorama import Style

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "folder", help="Folder containing the experiments with training times to collect."
)

args = argparser.parse_args()

print(f"{Style.BRIGHT}{'METHOD':.<55}{'TIME (s)':.>8}{Style.RESET_ALL}")
for exp in sorted(os.listdir(args.folder)):
    try:
        meta_file_name = [
            filename
            for filename in os.listdir(os.path.join(args.folder, exp))
            if filename.endswith(".meta")
        ][0]
    except:
        print(f"Could not find a meta file for {exp} in {args.folder}")

    with open(os.path.join(args.folder, exp, meta_file_name), "r") as meta_file:
        try:
            training_time = re.findall(
                "(?<=Training time \(seconds\): )\d+", meta_file.read()
            )
            training_time = [int(t) for t in training_time]
            training_time = np.mean(training_time)
            training_time = int(np.round(training_time))
            print(f"{exp:.<55}{training_time:.>8}")
        except:
            print(
                f"Could not find the training time in {os.path.join(args.folder, exp, meta_file_name)}"
            )
