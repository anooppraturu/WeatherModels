import argparse
from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from tyndall.models import build_model
from tyndall.model_utils import (
    autoregressive_rollout,
    make_rollout_gif,
    make_comparison_gif,
)
from tyndall.metrics import dataset_rollout_rmse, get_spectrum
from tyndall.data import ShardedWeatherBenchDataset

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
)

labels = ["T850", "U850", "V850", "Z500"]
colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]


def make_rmse_plots(model, dataset, roll_len, save_dir):
    error_mean, error_std = dataset_rollout_rmse(
        model=model, dataset=dataset, roll_len=roll_len
    )

    fig, axs = plt.subplots(2, 2, figsize=(7.5, 7.5))
    for i, ax in enumerate(np.ravel(axs)):
        ax.errorbar(
            x=range(len(error_mean[:, i])),
            y=error_mean[:, i],
            yerr=error_std[:, i],
            c=colors[i],
            lw=3,
            capsize=4,
        )
        ax.set_xlabel("Rollout Steps", fontsize=15)
        ax.set_ylabel("RMSE", fontsize=15)
        ax.set_title(labels[i], fontsize=15)
    fig.tight_layout()

    save_path = save_dir / "rmse_plot.png"
    fig.savefig(save_path)

    return


def make_gifs(model, dataset, start_idx, roll_len, save_dir):
    x0, _ = dataset[start_idx]
    x0 = torch.unsqueeze(x0, 0)

    model_trajectory = autoregressive_rollout(model=model, x_init=x0, nsteps=roll_len)
    dataset_trajectory = torch.stack(
        [dataset[start_idx + t][0][-4:] for t in range(roll_len + 1)]
    )

    gif_path = save_dir / "rollout.gif"
    make_rollout_gif(
        x=model_trajectory,
        save_path=gif_path,
    )
    gif_path = save_dir / "error_rollout.gif"
    make_rollout_gif(
        x=model_trajectory - dataset_trajectory, save_path=gif_path, suptitle="Errors"
    )
    gif_path = save_dir / "rollout_comparison.gif"
    make_comparison_gif(
        x1=model_trajectory,
        x2=dataset_trajectory,
        save_path=gif_path,
        variable_names=labels,
    )

    return

def make_spectrum_plots(model, dataset, start_idx, times, save_dir):
    x0, _ = dataset[start_idx]
    x0 = torch.unsqueeze(x0, 0)
    T = len(times)

    roll_len = max(times)
    model_trajectory = autoregressive_rollout(model=model, x_init=x0, nsteps=roll_len)
    dataset_trajectory = torch.stack(
        [dataset[start_idx + t][0][-4:] for t in range(roll_len + 1)]
    )

    fig, axs = plt.subplots(T, 4, figsize=(20, 5*T))

    for t in range(T):
        error_bins, error_spec = get_spectrum(model_trajectory[times[t]] - dataset_trajectory[times[t]])
        for i, label in enumerate(labels):
            axs[t][i].plot(error_bins, error_spec[i], lw=7.5, c=colors[i])
            axs[t][i].set_title(f'{label} Error Spectrum at T={times[t]}', fontsize=12.5)
            axs[t][i].set_yscale('log')
            axs[t][i].set_xscale('log')
            axs[t][i].set_xlabel('$|k|$', fontsize=15)
            axs[t][i].set_ylabel('$|\hat{e}(k)|^2$', fontsize=15)

    fig.tight_layout()
    output_path = save_dir / "error_spectrum_plots.png"
    fig.savefig(output_path)
    
    fig, axs = plt.subplots(T, 4, figsize=(20, 5*T))

    for t in range(T):
        model_bins, model_spec = get_spectrum(model_trajectory[times[t]])
        data_bins, data_spec = get_spectrum(dataset_trajectory[times[t]])
        for i, label in enumerate(labels):
            axs[t][i].plot(0.5*(model_bins + data_bins), model_spec[i]/data_spec[i], lw=7.5, c=colors[i])
            axs[t][i].set_title(f'{label} Prediction/Data Power Ratio at T={times[t]}', fontsize=12.5)
            axs[t][i].set_xlabel('$|k|$', fontsize=15)
            axs[t][i].set_ylabel('$|\hat{m}(k)|^2 / |\hat{d}(k)|^2$', fontsize=15)

    fig.tight_layout()
    output_path = save_dir / "power_ratio_plots.png"
    fig.savefig(output_path)

   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # get directory
    data_dir = Path(config["eval"]["data_dir"])
    train_config = data_dir / "config.yaml"
    model_path = data_dir / "best.pt"

    # load train config
    with open(train_config, "r") as f:
        train_config = yaml.safe_load(f)

    # load trained model
    model = build_model(train_config)
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict["model"])
    model.eval()

    # load dataset
    dataset = ShardedWeatherBenchDataset(
        root=config["eval"]["eval_data_path"],
        split=config["eval"]["split"],
        input_steps=train_config["data"]["input_steps"],
        lead_steps=train_config["data"]["lead_steps"],
        flatten_time=train_config["data"]["flatten_time"],
    )

    print("Computing Rollout RMSEs\n")
    make_rmse_plots(
        model=model,
        dataset=dataset,
        roll_len=config["eval"]["rollout_length"],
        save_dir=data_dir,
    )

    print("Making Rollout GIFs\n")
    make_gifs(
        model=model,
        dataset=dataset,
        start_idx=config["eval"]["start_idx"],
        roll_len=config["eval"]["rollout_length"],
        save_dir=data_dir,
    )

    print("Making Spectral Plots\n")
    make_spectrum_plots(
        model=model,
        dataset=dataset,
        start_idx=config["eval"]["start_idx"],
        times=config["eval"]["eval_times"],
        save_dir=data_dir,
    )

    return


if __name__ == "__main__":
    main()
