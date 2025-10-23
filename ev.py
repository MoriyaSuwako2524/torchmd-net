# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from torchmdnet.models.model import load_model

EV_TO_KCAL = 23.0605
ANG_CONV = 1.0


def _prep_xy(x, y, scale=1.0, manual_axis=False):
    """??? -> ?? -> ?? NaN/Inf -> ??????"""
    x = np.asarray(x, dtype=float).copy() * scale
    y = np.asarray(y, dtype=float).copy() * scale
    x = x.ravel();
    y = y.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask];
    y = y[mask]
    if manual_axis:
        ref = x.mean()
        x = x - ref
        y = y - ref
    return x, y


def _equal_limits(x, y, pad_ratio=0.05):
    data_min = min(x.min(), y.min())
    data_max = max(x.max(), y.max())
    span = data_max - data_min
    pad = pad_ratio * span if span > 0 else 1.0
    lo, hi = data_min - pad, data_max + pad
    return lo, hi


def hexbin_on_ax(ax, x, y, xlabel, ylabel, title, panel_letter, scale=1.0, manual_axis=False):
    """??? ax ??? hexbin + ?? + ??"""
    x, y = _prep_xy(x, y, scale=scale, manual_axis=manual_axis)

    # ?????
    slope, intercept, r_value, _, _ = linregress(x, y)
    mae = np.mean(np.abs(y - x))
    rmse = np.sqrt(np.mean((y - x) ** 2))

    # ????????
    lo, hi = _equal_limits(x, y)
    hb = ax.hexbin(x, y, gridsize=60, cmap='viridis', bins='log')
    ax.set_xlim(lo, hi);
    ax.set_ylim(lo, hi)
    xs = np.linspace(lo, hi, 200)
    ax.plot(xs, slope * xs + intercept, 'r--', linewidth=1)
    ax.set_aspect('equal', adjustable='box')

    # ?????????
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.text(0.18, 0.96, f"RMSE = {rmse:.4f}\nMAE = {mae:.4f}",
            transform=ax.transAxes, va='top', ha='left')
    ax.text(0.02, 0.98, panel_letter, transform=ax.transAxes,
            va='top', ha='left', fontsize=12, fontweight='bold')

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(hb, cax=cax)
    cbar.set_label('log$_{10}$(count)')


def evaluate(model_path, coord_path, type_path, energy_path, grad_path, dipole_path, split_path=None, batch_size=1,
             output_path="eval_plots"):
    os.makedirs(output_path, exist_ok=True)

    # ===== Load model =====
    model = load_model(model_path, derivative=True)
    model.eval()

    # ===== Load data =====
    coords = np.load(coord_path)  # (n_frames, n_atoms, 3)
    types = np.load(type_path)  # (n_atoms,)
    energies_ref = np.load(energy_path)  # (n_frames,)
    forces_ref = np.load(grad_path)  # (n_frames, n_atoms, 3)
    dipole_ref = np.load(dipole_path)

    # === ??1??? split ??? test ===
    if split_path is not None:
        split = np.load(split_path)
        test_idx = split["idx_test"]  # test é?¨???????′￠???

        coords = coords[test_idx]
        energies_ref = energies_ref[test_idx]
        forces_ref = forces_ref[test_idx]
        dipole_ref = dipole_ref[test_idx]

    n_frames, n_atoms, _ = coords.shape
    z0 = torch.tensor(types, dtype=torch.long)

    energies_pred = []
    forces_pred = []
    dipole_pred = []

    # ===== Loop over frames =====
    for i in range(0, n_frames, batch_size):
        j = min(i + batch_size, n_frames)

        pos_batch = torch.tensor(coords[i:j], dtype=torch.float32)  # (B, n_atoms, 3)
        z_batch = z0.repeat(j - i)  # (B*n_atoms,)
        pos = pos_batch.reshape(-1, 3).requires_grad_(True)
        batch = torch.arange(j - i).repeat_interleave(n_atoms)

        output = model(z_batch, pos, batch)
        y_pred = output["y"]
        neg_dy_pred = output["neg_dy"]
        vec_pred = output["vec"]
        energies_pred.append(y_pred.cpu())
        forces_pred.append(neg_dy_pred.reshape(j - i, n_atoms, 3).cpu())
        vec_pred = vec_pred.reshape(j - i, 3).cpu()
        dipole_pred.append(vec_pred)

    energies_pred = torch.cat(energies_pred, dim=0)
    forces_pred = torch.cat(forces_pred, dim=0)
    dipole_pred = torch.cat(dipole_pred, dim=0)

    # Convert refs
    energies_ref = torch.tensor(energies_ref, dtype=torch.float32)
    forces_ref = torch.tensor(forces_ref, dtype=torch.float32)
    dipole_ref = torch.tensor(dipole_ref, dtype=torch.float32)

    # ===== Compute errors =====
    energies_pred_centered = energies_pred - energies_pred.mean()
    energies_ref_centered = energies_ref - energies_ref.mean()

    mae_e = torch.mean(torch.abs(energies_pred_centered - energies_ref_centered))
    rmse_e = torch.sqrt(torch.mean((energies_pred_centered - energies_ref_centered) ** 2))
    mae_f = torch.mean(torch.abs(forces_pred - forces_ref))
    rmse_f = torch.sqrt(torch.mean((forces_pred - forces_ref) ** 2))
    mae_d = torch.mean(torch.abs(dipole_pred - dipole_ref))
    rmse_d = torch.sqrt(torch.mean((dipole_pred - dipole_ref) ** 2))

    print("==== Evaluation Results ====")
    print(f"Energy MAE  : {mae_e.item():.4f} kcal/mol")
    print(f"Energy RMSE : {rmse_e.item():.4f} kcal/mol")
    print(f"Force MAE   : {mae_f.item():.4f} kcal/mol")
    print(f"Force RMSE  : {rmse_f.item():.4f} kcal/mol")
    print(f"Dipole MAE   : {mae_d.item():.4f} kcal/mol")
    print(f"Dipole RMSE  : {rmse_d.item():.4f} kcal/mol")


    # ===== Plotting =====
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.5, right=0.92, left=0.08, top=0.9, bottom=0.15)

    # Energy (A)
    hexbin_on_ax(
        axes[0], energies_ref.detach().numpy(), energies_pred.detach().numpy(),
        "Reference Energy (kcal/mol)", "Predicted Energy (kcal/mol)",
        "Energy Prediction", "A", scale=1.0, manual_axis=True
    )

    # Force (B)
    hexbin_on_ax(
        axes[1], forces_ref.detach().numpy().ravel(), forces_pred.detach().numpy().ravel(),
        r"Reference Force ($kcal/mol \cdot \AA$)", r"Predicted Force ($kcal/mol \cdot \AA$)",
        "Force Prediction", "B", scale=1.0, manual_axis=False
    )
    hexbin_on_ax(
        axes[2], dipole_ref.detach().numpy().ravel(), dipole_pred.detach().numpy().ravel(),
        r"Reference Dipole ($au$)", r"Predicted Dipole ($au$)",
        "Dipole Prediction", "C", scale=1.0, manual_axis=False
    )
    axes[2].set_aspect('equal', adjustable='box')
    out_file = os.path.join(output_path, "energy_force_hexbin.png")
    fig.savefig(out_file, dpi=300)
    # plt.show()
    plt.close(fig)
    print(f"Saved plot to {out_file}")



if __name__ == "__main__":
    path = "/scratch/moriya2524/mlp/phbdi_tddft_datas/"
    model_path = f"/scratch/moriya2524/mlp/tensornet_phbdi_tddft_ground_dipolemom_1000_s/epoch=849-val_loss=1.1341-test_loss=0.0000.ckpt"
    coord_path = f"{path}full_coord.npy"
    type_path = f"{path}full_type.npy"
    energy_path = f"{path}full_S1_energy.npy"
    grad_path = f"{path}full_force.npy"
    dipole_path = f"{path}full_dipolemom.npy"
    split_path = f"{path}2000_split.npz"
    evaluate(model_path, coord_path, type_path, energy_path, grad_path, dipole_path, split_path, batch_size=8)
