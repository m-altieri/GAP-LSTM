import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


_FOLDERS_PATH = "../../preds"


def load_preds(model, dataset, verbose=False):
    try:
        folder = os.path.join(
            _FOLDERS_PATH,
            list(
                filter(
                    lambda x: x is not None,
                    [
                        re.match(f"{model}.*-{dataset}.*", f)
                        for f in os.listdir(_FOLDERS_PATH)
                    ],
                )
            )[0][0],
        )
    except:
        print(
            f"[ERROR] Skipping {model} on {dataset}: Could not find preds folder in {_FOLDERS_PATH}."
        )

    if verbose:
        print(f"Found preds folder for {model} and {dataset} in {folder}.")

    try:
        preds_filename = list(
            filter(
                lambda x: x is not None,
                [re.match("preds-.*.csv", f) for f in os.listdir(folder)],
            )
        )[0][0]
    except:
        print(f"[ERROR] Could not find preds for {model} and {dataset} in {folder}.")
        sys.exit(1)

    if verbose:
        print(
            f"Found preds file for {model} and {dataset} in {folder}: {preds_filename}."
        )

    preds = pd.read_csv(os.path.join(folder, preds_filename))

    return preds


def count_overestimates_and_underestimates(truth, preds, model, dataset):
    over = len(np.where(preds - truth > 0)[0])
    under = len(np.where(preds - truth < 0)[0])
    exact = len(np.where(preds - truth == 0)[0])
    total = len(preds)
    print(
        f"{model:<20}-{dataset:<30}: {over:>6} over ({100*over/total:.2f}%), "
        + f"{under:>6} under ({100*under/total:.2f}%), "
        + f"{exact:>6} exact ({100*exact/total:.2f}%), "
        f"sum: {over + under + exact:>6} / {total:<6} ({100*(over+under+exact)/total:.2f}%)"
    )
    if over + under + exact != total:
        print(f"[WARNING] over + under + exact ({over+under+exact}) != total ({total})")

    return over, under, exact


def plot_preds(truth, preds, plot_name):
    plt.plot(preds, color="red", linewidth=1, alpha=0.4)
    plt.plot(truth, color="blue", linewidth=1, alpha=0.4)
    plt.plot(np.mean(preds, axis=-1), color="red", linewidth=4)
    plt.plot(np.mean(truth, axis=-1), color="blue", linewidth=4)
    plt.plot([], color="red", label="Predictions")
    plt.plot([], color="blue", label="Ground truth")
    plt.xlabel("Timesteps", fontsize=20)
    plt.ylabel("Target", fontsize=20)
    plt.xticks(
        np.arange(0, len(truth), 3), np.arange(0, len(truth), 3) + 1, fontsize=20
    )
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.savefig(f"{plot_name}.png")
    plt.savefig(f"{plot_name}.pdf")

    plt.clf()


dataset_config = {
    "beijing-multisite-airquality": {"timesteps": 6, "nodes": 12},
    "lightsource": {"timesteps": 19, "nodes": 7},
    "pems-sf-weather": {"timesteps": 6, "nodes": 163},
    "pv-italy": {"timesteps": 19, "nodes": 17},
    "wind-nrel": {"timesteps": 24, "nodes": 5},
}


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--models", nargs="+")
    argparser.add_argument("--datasets", nargs="+")
    argparser.add_argument("-v")
    args = argparser.parse_args()

    overestimates_and_underestimates = {
        "model": [],
        "dataset": [],
        "over": [],
        "under": [],
        "exact": [],
        "sum": [],
        "total": [],
    }

    for dataset in args.datasets:
        for model in args.models:
            try:
                preds_df = load_preds(model, dataset, verbose=args.v)
            except:
                continue
            truth = preds_df.iloc[:, 1]
            preds = preds_df.iloc[:, 2]

            T = dataset_config[dataset]["timesteps"]
            N = dataset_config[dataset]["nodes"]

            save_folder = f"../../pred_curves/{model}-{dataset}"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # COUNT OVERESTIMATES AND UNDERESTIMATES
            # over, under, exact = count_overestimates_and_underestimates(
            #     truth, preds, model, dataset
            # )
            # overestimates_and_underestimates["model"].append(model)
            # overestimates_and_underestimates["dataset"].append(dataset)
            # overestimates_and_underestimates["over"].append(over)
            # overestimates_and_underestimates["under"].append(under)
            # overestimates_and_underestimates["exact"].append(exact)
            # overestimates_and_underestimates["sum"].append(over + under + exact)
            # overestimates_and_underestimates["total"].append(len(truth))
            # overestimates_and_underestimates_df = pd.DataFrame(
            #     overestimates_and_underestimates
            # )
            # overestimates_and_underestimates_df.to_csv(
            #     "overestimates_underestimates.csv"
            # )

            # PLOT PREDICTION AND TRUTH CURVES
            # pbar = tqdm(range(len(preds_df) // (T * N)))
            # for seq in pbar:
            #     pbar.set_description(f"Plotting sequence {seq}")
            #     plot_name = f"{save_folder}/{model}-{dataset}-s{seq}"
            #     plot_preds(
            #         np.reshape(truth[seq * T * N : (seq + 1) * T * N], (T, N)),
            #         np.reshape(preds[seq * T * N : (seq + 1) * T * N], (T, N)),
            #         plot_name,
            #     )


if __name__ == "__main__":
    main()
