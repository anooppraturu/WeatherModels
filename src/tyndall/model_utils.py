import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


@torch.no_grad()
def autoregressive_rollout(model, x_init, nsteps, n_var=4):
    trajectory = [x_init[:,-n_var:]]

    x = x_init.clone().detach()

    for n in range(nsteps):
        x_next = model(x)
        trajectory.append(x_next)
        x = torch.cat([x[:, -n_var*3:], x_next], dim=1)

    return torch.cat(trajectory, dim=0)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def make_rollout_gif(
    x,
    save_path,
    variable_names=None,
    fps=4,
    interval=250,
    cmap="viridis",
    latitudes=None,
    longitudes=None,
    suptitle="Rollout",
):
    """
    x: array/tensor of shape (T, V, H, W)
    save_path: output .gif path
    variable_names: list of variable names of length V
    fps: frames per second in the saved gif
    interval: ms between frames for animation
    latitudes, longitudes: optional coordinate arrays for axis extent
    """
    x = _to_numpy(x)
    T, V, H, W = x.shape

    if variable_names is None:
        variable_names = [f"Var {i}" for i in range(V)]

    # Use a fixed color scale per variable across the whole rollout
    # this only really works because all variables are standardized
    vmins = x.min(axis=(0, 2, 3))
    vmaxs = x.max(axis=(0, 2, 3))

    fig, axes = plt.subplots(1, V, figsize=(4 * V, 4), constrained_layout=True)
    if V == 1:
        axes = [axes]

    ims = []

    for j, ax in enumerate(axes):
        if latitudes is not None and longitudes is not None:
            im = ax.imshow(
                x[0, j],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmins[j],
                vmax=vmaxs[j],
                extent=[longitudes[0], longitudes[-1], latitudes[0], latitudes[-1]],
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            im = ax.imshow(
                x[0, j],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmins[j],
                vmax=vmaxs[j],
            )

        ax.set_title(variable_names[j])
        fig.colorbar(im, ax=ax, shrink=0.8)
        ims.append(im)

    title = fig.suptitle(f"{suptitle} | t = 0")

    def update(frame):
        for j in range(V):
            ims[j].set_data(x[frame, j])
        title.set_text(f"{suptitle} | t = {frame}")
        return ims + [title]

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_comparison_gif(
    x1,
    x2,
    save_path,
    variable_names=None,
    row_names=("Rollout", "Data"),
    fps=4,
    interval=250,
    cmap="viridis",
    latitudes=None,
    longitudes=None,
    suptitle="Rollout Comparison",
):
    """
    x1, x2: arrays/tensors of shape (T, V, H, W)
    save_path: output .gif path
    variable_names: list of variable names of length V
    row_names: names for the two rows
    """
    x1 = _to_numpy(x1)
    x2 = _to_numpy(x2)

    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    T, V, H, W = x1.shape

    if variable_names is None:
        variable_names = [f"Var {i}" for i in range(V)]

    # Use shared color scale per variable across BOTH tensors and all times
    both = np.concatenate([x1, x2], axis=0)  # shape (2T, V, H, W)
    vmins = both.min(axis=(0, 2, 3))
    vmaxs = both.max(axis=(0, 2, 3))

    fig, axes = plt.subplots(2, V, figsize=(4 * V, 8), constrained_layout=True)

    ims_top = []
    ims_bottom = []

    for j in range(V):
        # Top row: x1
        ax = axes[0, j]
        if latitudes is not None and longitudes is not None:
            im1 = ax.imshow(
                x1[0, j],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmins[j],
                vmax=vmaxs[j],
                extent=[longitudes[0], longitudes[-1], latitudes[0], latitudes[-1]],
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            im1 = ax.imshow(
                x1[0, j],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmins[j],
                vmax=vmaxs[j],
            )

        ax.set_title(f"{row_names[0]}: {variable_names[j]}")
        fig.colorbar(im1, ax=ax, shrink=0.8)
        ims_top.append(im1)

        # Bottom row: x2
        ax = axes[1, j]
        if latitudes is not None and longitudes is not None:
            im2 = ax.imshow(
                x2[0, j],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmins[j],
                vmax=vmaxs[j],
                extent=[longitudes[0], longitudes[-1], latitudes[0], latitudes[-1]],
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            im2 = ax.imshow(
                x2[0, j],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmins[j],
                vmax=vmaxs[j],
            )

        ax.set_title(f"{row_names[1]}: {variable_names[j]}")
        fig.colorbar(im2, ax=ax, shrink=0.8)
        ims_bottom.append(im2)

    title = fig.suptitle(f"{suptitle} | t = 0")

    def update(frame):
        for j in range(V):
            ims_top[j].set_data(x1[frame, j])
            ims_bottom[j].set_data(x2[frame, j])
        title.set_text(f"{suptitle} | t = {frame}")
        return ims_top + ims_bottom + [title]

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)