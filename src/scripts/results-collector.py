import os
import re
import sys
import pandas as pd
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "-i", "--input-path", default="../../experiments/debug", action="store"
)
argparser.add_argument(
    "-o", "--output-path", default="collected-results.csv", action="store"
)
argparser.add_argument(
    "--one-deep",
    action="store_const",
    const=True,
    help="set it if experiments are not "
    + "contained each in its own folder, but rather the preds files are directly inside the top-level folder."
    + " It automatically disabled the meta file, and therefore the training time."
    + " It assumes there is one and only one preds file for experiment, and nothing else.",
)

args = argparser.parse_args()

# @TODO usare direttamente gli args al posto delle occorrenze di PATH e OUTPUT_PATH
PATH = args.input_path
OUTPUT_PATH = args.output_path

preds_regex = "^preds\-.*\.csv$"
meta_regex = "^.*-(lightsource|wind-nrel|pv-italy|pems-sf-weather|beijing-(multisite-)?airquality).meta$"

dataset_config = {
    "beijing-multisite-airquality": {"timesteps": 6, "nodes": 11},
    "lightsource": {"timesteps": 19, "nodes": 7},
    "pems-sf-weather": {"timesteps": 6, "nodes": 163},
    "pv-italy": {"timesteps": 19, "nodes": 17},
    "wind-nrel": {"timesteps": 24, "nodes": 5},
}

experiments = os.listdir(PATH)
if "config" in experiments:
    experiments.remove("config")

# all metrics are recalculated from the preds
collected_results = pd.DataFrame(
    columns=[
        "model",
        "dataset",
        # "conf",
        "avg_global_mae",
        "var_global_mae",
        # "intrasequence_var_mae",
        "avg_global_rmse",
        "avg_datewise_rmse",
        "var_datewise_rmse",
        # "intrasequence_var_rmse",
        # "avg_datewise_mape",
        # "std_datewise_mape",
        "avg_datewise_smape",
        "var_datewise_smape",
        # "intrasequence_var_smape",
        "training_time",
    ]
)

lightsource_datewise_rmse = pd.DataFrame()
wind_datewise_rmse = pd.DataFrame()
pv_italy_datewise_rmse = pd.DataFrame()
pems_datewise_rmse = pd.DataFrame()
beijing_datewise_rmse = pd.DataFrame()

all_datewise_rmses = {}

for exp_index, experiment in enumerate(experiments):
    if not args.one_deep:
        exp_files = os.listdir(os.path.join(PATH, experiment))

        # Take the preds file
        preds_file = [file for file in exp_files if re.match(preds_regex, file)]
        if len(preds_file) != 1:
            print(
                "{:0>3}) Experiment:   {}\n     >>> ERROR <<< CAN'T FIND CORRECT PREDS FILE!".format(
                    exp_index + 1, experiment
                )
            )
            continue
        preds_file = preds_file[0]
        print(
            "{:0>3}) Experiment:   {}\n     Preds File: {}".format(
                exp_index + 1, experiment, preds_file
            )
        )

        # Take the meta file
        meta_file = [file for file in exp_files if re.match(meta_regex, file)]
        if len(meta_file) != 1:
            print(
                "{:0>3}) Experiment:   {}\n     >>>>> CANNOT FIND META FILE! <<<<<".format(
                    exp_index + 1, experiment
                )
            )
            # continue
        else:
            meta_file = meta_file[0]
            print(
                "{:0>3}) Experiment:   {}\n     Results File: {}".format(
                    exp_index + 1, experiment, meta_file
                )
            )

        preds_df = pd.read_csv(os.path.join(PATH, experiment, preds_file))

    else:
        preds_df = pd.read_csv(os.path.join(PATH, experiment))

    try:
        # there was (?<=test-) at the beginning
        model = re.search(
            f"(ARIMA|Prophet|VAR|((Bi|CNN|SVD|GCN|Attention)-)?LSTM|GRU|GAP-LSTM(-Default|-Weighted)?|GCLSTM|GraphWaveNet|ESG|MTGNN|RGSL|Triformer)(?=-)",
            experiment,
        ).group(0)
        dataset = re.search(
            "(?<=-)(lightsource|wind-nrel|pv-italy|pems-sf-weather|beijing-(multisite-)?airquality)",
            experiment,
        ).group(0)
        conf = re.search(
            "(?<=-)(AUTO|Prophet|O\d|BS\d\d?LR1e-\d(LRS)?(ES)?)", preds_df.columns[2]
        ).group(0)

    except AttributeError:  # Skippa experiment con nome non valido (es. val-...)
        print(
            "{:0>3}) Experiment:   {}\n     >>>>> CANNOT EXTRACT ALL EXP INFORMATION! <<<<<".format(
                exp_index + 1, experiment
            )
        )
    model_regex = ".*(?=-(AUTO|Prophet|O\d|BS\d\d?))"
    pred_column = preds_df.columns[2]
    print(pred_column)

    preds_df["abs_diff"] = (preds_df["truth"] - preds_df[pred_column]).abs()
    preds_df["mse"] = preds_df["abs_diff"] ** 2
    preds_df["rel_diff"] = (
        (preds_df["truth"] - preds_df[pred_column]) / preds_df["truth"]
    ).abs()
    preds_df["rel_sym_diff"] = (preds_df["truth"] - preds_df[pred_column]).abs() / (
        preds_df["truth"].abs() + preds_df[pred_column].abs()
    )

    t = dataset_config[dataset]["timesteps"]
    n = dataset_config[dataset]["nodes"]

    avg_global_mae = np.mean(preds_df["abs_diff"])
    var_global_mae = np.var(preds_df["abs_diff"])

    # intrasequence_var_mae = np.mean(
    #     np.var(
    #         [
    #             preds_df["abs_diff"][i * t * n : (i + 1) * t * n]
    #             for i in range(round(len(preds_df) / t / n))
    #         ],
    #         axis=1,
    #     )
    # )
    avg_global_rmse = np.sqrt(np.mean(preds_df["mse"]))
    datewise_mse = np.mean(
        [
            preds_df["mse"][i * t * n : (i + 1) * t * n]
            for i in range(round(len(preds_df) / t / n))
        ],
        axis=1,
    )
    datewise_rmse = [np.sqrt(datewise_mse[i]) for i in range(len(datewise_mse))]

    # Collect all metrics (for all datasets) for each model to use for statistical tests
    if model not in all_datewise_rmses:
        all_datewise_rmses[model] = {}
    all_datewise_rmses[model][dataset] = datewise_rmse

    avg_datewise_rmse = np.mean(datewise_rmse, axis=0)
    var_datewise_rmse = np.var(datewise_rmse)
    # intrasequence_var_rmse = np.mean(
    #     np.sqrt(
    #         np.var(
    #             [
    #                 preds_df["mse"][i * t * n : (i + 1) * t * n]
    #                 for i in range(round(len(preds_df) / t / n))
    #             ],
    #             axis=1,
    #         )
    #     )
    # )
    # datewise_mape = np.mean(
    #     [
    #         preds_df["rel_diff"][i * t * n : (i + 1) * t * n]
    #         for i in range(round(len(preds_df) / t / n))
    #     ],
    #     axis=1,
    # )
    # avg_datewise_mape = np.mean(datewise_mape, axis=0)
    # std_datewise_mape = np.std(datewise_mape, axis=0)
    datewise_smape = np.mean(
        [
            preds_df["rel_sym_diff"][i * t * n : (i + 1) * t * n]
            for i in range(round(len(preds_df) / t / n))
        ],
        axis=1,
    )
    avg_datewise_smape = 100 * np.mean(datewise_smape, axis=0)
    var_datewise_smape = np.var(datewise_smape, axis=0)
    # intrasequence_var_smape = np.mean(
    #     np.var(
    #         [
    #             preds_df["rel_sym_diff"][i * t * n : (i + 1) * t * n]
    #             for i in range(round(len(preds_df) / t / n))
    #         ],
    #         axis=1,
    #     )
    # )

    print("Avg global MAE: {}".format(avg_global_mae))
    print("Var global MAE: {}".format(var_global_mae))
    # print("Intrasequence Var MAE: {}".format(intrasequence_var_mae))
    print("Avg global RMSE: {}".format(avg_global_rmse))
    print("Date-wise MSE: {}".format(datewise_mse))
    print("Date-wise RMSE: {}".format(datewise_rmse))
    print("Avg date-wise RMSE : {}".format(avg_datewise_rmse))
    print("Var date-wise RMSE: {}".format(var_datewise_rmse))
    # print("Intrasequence Var RMSE: {}".format(intrasequence_var_rmse))
    # print("Avg date-wise MAPE: {}".format(avg_datewise_mape))
    # print("Std date-wise MAPE: {}".format(std_datewise_mape))
    print("Avg date-wise SMAPE: {}".format(avg_datewise_smape))
    print("Var date-wise SMAPE: {}".format(var_datewise_smape))
    # print("Intrasequence Var SMAPE: {}".format(intrasequence_var_smape))

    training_time = None
    if not args.one_deep:
        if meta_file != []:
            with open(os.path.join(PATH, experiment, meta_file), "r") as f:
                training_time = re.search(
                    "Training time \(seconds\): (\d*)", f.read()
                ).group(1)
        else:
            training_time = 0

    print("Training time: {}".format(training_time))

    row = [
        model,
        dataset,
        # conf,
        avg_global_mae,
        var_global_mae,
        # intrasequence_var_mae,
        avg_global_rmse,
        avg_datewise_rmse,
        var_datewise_rmse,
        # intrasequence_var_rmse,
        # avg_datewise_mape,
        # std_datewise_mape,
        avg_datewise_smape,
        var_datewise_smape,
        # intrasequence_var_smape,
        training_time,
    ]
    collected_results.loc[len(collected_results)] = row

collected_results.to_csv(OUTPUT_PATH, index=False)


# CREATE DATAFRAME FOR STATISTICAL TESTS
# Temporary, remove: fill ARIMA-pems so that they all have the same number of rows
if not os.path.exists("datewise_rmses"):
    os.makedirs("datewise_rmses")

if "ARIMA" in all_datewise_rmses:
    all_datewise_rmses["ARIMA"]["pems-sf-weather"] = np.ones_like(
        all_datewise_rmses["LSTM"]["pems-sf-weather"]
    )

# Dataset-wise
datasetwise_datewise_rmses = {
    dataset: {model: all_datewise_rmses[model][dataset] for model in all_datewise_rmses}
    for dataset in dataset_config
}
for dataset in datasetwise_datewise_rmses:
    df = pd.DataFrame(datasetwise_datewise_rmses[dataset])
    df.to_csv(f"datewise_rmses/all-datewise-rmses-{dataset}.csv", index=False)
    (1 - df).to_csv(
        f"datewise_rmses/oneminus-all-datewise-rmses-{dataset}.csv", index=False
    )

# All datasets combined
all_datewise_rmses = {
    model: np.concatenate(
        ([all_datewise_rmses[model][dataset] for dataset in dataset_config])
    )
    for model in all_datewise_rmses
}
df = pd.DataFrame(all_datewise_rmses)
df.to_csv(f"datewise_rmses/all-datewise-rmses.csv", index=False)
(1 - df).to_csv(f"datewise_rmses/oneminus-all-datewise-rmses.csv", index=False)
