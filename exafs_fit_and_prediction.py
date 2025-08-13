#!/usr/bin/env python3
# Combined Script: FEFF + EXAFS Fitting (XrayLarch) + PINN Prediction
"""
By Latif Ullah Khan
BM08-XAFS/XRF Beamline
SESAME Synchrotron Light Source
29 June 2025
"""


import os
import shutil
import sys
sys.setrecursionlimit(5000)
import numpy as np
import re
import joblib
import py3Dmol
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.interpolate import interp1d
from pymatgen.core import Structure
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool

# Larch imports
from larch import Group
from larch.xafs import (
    feff8l, autobk, feffpath, feffit_transform,
    feffit_dataset, feffit, feffit_report
)
from larch.xafs.xafsft import xftf
from larch.xrd.struct2xas import Struct2XAS
from larch.io import read_ascii
from larch.fitting import param, param_group

# TensorFlow / Keras
from tensorflow.keras.models import load_model

# --- Constants ------------------------------------------------------------
ABSORBER_ATOM = "Cd"           # e.g. the edge atom in your CIFs
FEFF_RADIUS   = 8.0            # Å sphere around absorber
EXP_DATA_FILE = "CdS_10K_01.xdi.txt.nor"  # μ(E) already background‑removed
RMAX          = 6.0            # Å – include paths with Reff ≤ RMAX in fit
ARTIFACT_DIR  = "artifacts"    # folder with pinn_model.keras & scalers.pkl

# --------------------------------------------------------------------------
# Load PINN + scalers (robust)
# --------------------------------------------------------------------------
try:
    model = load_model(Path(ARTIFACT_DIR) / "pinn_model.keras")
except Exception as exc:
    sys.exit(f"[FATAL] Could not load PINN model: {exc}")

try:
    scalers = joblib.load(Path(ARTIFACT_DIR) / "scalers.pkl")
    # ---- expected keys ---------------------------------------------------
    #  • "k"     StandardScaler (k‑grid)
    #  • "feff"  StandardScaler (χ_FEFF·k²)
    #  • "y"     StandardScaler (χ_exp·k²)
    scaler_X = {k: scalers[k] for k in ("k", "feff") if k in scalers}
    scaler_y = scalers["y"]
    if scaler_X is None:
        raise KeyError("'feff' scaler missing in scalers.pkl")
except Exception as exc:
    sys.exit(f"[FATAL] Could not load scalers: {exc}")

# --------------------------------------------------------------------------
# FEFF runner for *one* absorber site
# --------------------------------------------------------------------------
def run_feff_for_site(args):
    site_index, mat_obj, DFT_path, abs_atom, feff_radius = args
    try:
        print(f"\n[INFO] Processing absorber site {site_index + 1}/{mat_obj.nabs_sites}")
        mat_obj.set_abs_site(site_index)
        try:
            mat_obj.get_coord_envs_info()
        except UnboundLocalError:
            print("[WARNING] Could not determine coordination environment.")
        mat_obj.get_coord_envs()
        mat_obj.make_input_feff(radius=feff_radius, template=None, parent_path=DFT_path)
        feff_inp = os.path.join(mat_obj.outdir, 'feff.inp')
        if not os.path.exists(feff_inp):
            print(f"[ERROR] feff.inp not found for site {site_index}. Skipping.")
            return
        feff_outdir = os.path.join(DFT_path, f'feff_site_{site_index}')
        if os.path.exists(feff_outdir):
            shutil.rmtree(feff_outdir)
        os.makedirs(feff_outdir, exist_ok=True)
        # ---- patch feff.inp ------------------------------------------------
        with open(feff_inp, 'r') as f:
            lines = f.readlines()
        modified_lines = []
        for line in lines:
            if line.strip().startswith('S02'):
                line = 'S02       1.0\n'
            elif line.strip().startswith('PRINT'):
                line = 'PRINT     1      0     0     0     0      3\n'
            #elif line.strip().startswith('*SCF'):
            #    line = line.replace('*SCF', 'SCF')
            modified_lines.append(line)
        with open(os.path.join(feff_outdir, 'feff.inp'), 'w') as f:
            f.writelines(modified_lines)
        # ---- run FEFF -----------------------------------------------------
        print(f"[INFO] Running FEFF at {feff_outdir}...")
        feff8l(folder=feff_outdir, feffinp='feff.inp', verbose=True)
        print(f"[SUCCESS] FEFF completed for site {site_index}.")
    except Exception as e:
        print(f"[ERROR] Exception for site {site_index}: {e}")

# --------------------------------------------------------------------------
# Fitting FEFF paths to experimental EXAFS
# --------------perform_fitting() with FEFF path sorting by Reff:------------
def perform_fitting(feff_dir, exp_data_file=EXP_DATA_FILE, rmax=RMAX):
    try:
        # --- experimental χ(k) extraction ----------------------------------
        exp_data = read_ascii(exp_data_file, labels='e, norm, nbkg, flat, fbkg, nder, nsec')
        autobk(exp_data.e, exp_data.norm, group=exp_data, rbkg=1.0, kweight=2)
        trans = feffit_transform(kmin=2.0, kmax=12.0 , dk=4.0, kw=2, window='Kaiser-Bessel', fitspace='r', rmin=1.0, rmax=rmax)
        # --- parameter group ----------------------------------------------
        params = param_group(reff=-1.0)
        params.s02 = param(1.0, vary=True)
        params.e0 = param(0.001, vary=True)

        # ---------- build path list ---------------------------------------
        # ---- collect and sort FEFF paths by Reff ------------------------
        feff_paths = []
        for filename in os.listdir(feff_dir):
            if re.match(r'feff\d{4}\.dat', filename):
                fpath = os.path.join(feff_dir, filename)
                try:
                    fpath_obj = feffpath(fpath)
                    if fpath_obj.reff <= rmax:
                        feff_paths.append((fpath_obj.reff, fpath, fpath_obj))
                except Exception as e:
                    print(f"[WARNING] Could not parse {filename}: {e}")

        # Sort by Reff (ascending)
        feff_paths.sort(key=lambda x: x[0])

        # ---- construct path list and parameter group ---------------------
        paths = []
        for i, (reff, fpath, fpath_obj) in enumerate(feff_paths, start=1):
            idx = int(Path(fpath).stem[4:8])
            setattr(params, f'delr_{idx:04d}', param(0.001, min=-0.75, max=0.75, vary=True))
            setattr(params, f'sigma2_{idx:04d}', param(0.0075 + (i * 0.0001), min=0.0, max=1.0, vary=True))
            paths.append(feffpath(fpath, f'{idx:04d}', s02='s02', e0='e0',
                                  deltar=f'delr_{idx:04d}', sigma2=f'sigma2_{idx:04d}'))

        if not paths:
            print(f"[WARNING] No valid FEFF paths in {feff_dir} (reff < {rmax}). Skipping fit.")
            return

        # ---- build dataset and perform fit -------------------------------------
        dataset = feffit_dataset(data=exp_data, transform=trans, refine_bkg=False, paths=paths)
        result = feffit(params, dataset)
        report = feffit_report(result)
        print(report)
        outbase = os.path.basename(feff_dir)
        with open(os.path.join(feff_dir, f"{outbase}_feffit_report.txt"), "w") as f:
            f.write(report)

        # ---------- save χ(k) -----------------------------------
        k_data, chi_data = dataset.data.k, dataset.data.chi
        k_model, chi_model = dataset.model.k, dataset.model.chi
        np.savetxt(os.path.join(feff_dir, f"{outbase}_chik_k2.dat"),
                   np.column_stack((k_data, chi_data * k_data**2, k_model, chi_model * k_model**2)),
                   header="k_data chi_data_k2 k_model chi_model_k2", fmt="%.6e")
        
        # ---------- save χ(R) -----------------------------------
        r_data, chir_data = dataset.data.r, dataset.data.chir_mag
        r_model, chir_model = dataset.model.r, dataset.model.chir_mag
        chir_data_real, chir_model_real = dataset.data.chir_re, dataset.model.chir_re
        np.savetxt(os.path.join(feff_dir, f"{outbase}_chir.dat"),
                   np.column_stack((r_data, chir_data, chir_data_real, r_model, chir_model, chir_model_real)),
                   header="r_data chir_mag_data chir_real_data r_model chir_mag_model chir_real_model",
                   fmt="%.6e")
                
        # --------------------Plot EXAFS Fit--------------------
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Chi(k).k²", "Chi(R): Magnitude & Real Parts"])
        # Increase subplot title font size
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=24)  # Change to desired size
        fig.add_trace(go.Scatter(x=k_data, y=chi_data * k_data**2, mode='lines+markers', name='Expt. Chi(k).k²', 
                                 line=dict(color='lightsalmon', width=4), marker=dict(size=7, color="lightsalmon")), row=1, col=1)
        fig.add_trace(go.Scatter(x=k_model, y=chi_model * k_model**2, mode='lines', name='Model Chi(k).k²',
                                 line=dict(color='mediumseagreen', width=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=r_data, y=chir_data, mode='lines+markers', name='Expt. |χ(R)|',
                                 line=dict(color='lightsalmon', width=4), marker=dict(size=7, color="lightsalmon")), row=1, col=2)
        fig.add_trace(go.Scatter(x=r_model, y=chir_model, mode='lines', name='Model |χ(R)|',
                                 line=dict(color='mediumseagreen', width=4)), row=1, col=2)
        fig.add_trace(go.Scatter(x=r_data, y=chir_data_real, mode='lines+markers', name='Expt. Re χ(R)', 
                                 line=dict(color='lightsalmon', dash='dot', width=4), marker=dict(size=7, color="lightsalmon")), row=1, col=2)
        fig.add_trace(go.Scatter(x=r_model, y=chir_model_real, mode='lines', name='Model Re χ(R)',
                                 line=dict(color='mediumseagreen', dash='dot', width=4)), row=1, col=2)
        fig.update_layout(
            title="EXAFS Fit using FEFF/Larch",
            xaxis_title="k (Å⁻¹)",
            yaxis_title="k²χ(k) (Å⁻²)",
            xaxis2_title="R (Å)",
            yaxis2_title="χ(R) (Å⁻³)",
            template='plotly_dark',
            font=dict(size=26),  # Title and general font size
            legend=dict(
                font=dict(size=24),  # Legend font size
                bgcolor="rgba(0,0,0,0.5)",  # Optional background
                bordercolor="gray",
                borderwidth=1
            ),
            xaxis=dict(tickfont=dict(size=24)),
            yaxis=dict(tickfont=dict(size=24)),
            xaxis2=dict(tickfont=dict(size=24)),
            yaxis2=dict(tickfont=dict(size=24))
        )

        fig.write_html(os.path.join(feff_dir, f"{outbase}_FEFFLarch_fit_plot.html"))
        
    except Exception as e:
        print(f"[ERROR] Fitting failed for {feff_dir}: {e}")

# --------------------------------------------------------------------------
# PINN prediction & Chi(R) Fourier transform
# --------------------------------------------------------------------------
def predict_with_pinn(model, scaler_X, scaler_y, chik_file, output_html):
    """
    Predict χ(k)*k² with a pretrained PINNs DNNs and compare with experimental + FEFF/Larch Fit.

    Parameters
    ----------
    model        Trained Keras model
    scaler_X     Scaler used when training (features = χ_FEFF*k²)
    scaler_y     Scaler for inverse-transforming PINN output
    chik_file    Path to *_chik_k2.dat  (k, χ_exp·k², χ_model_k?, χ_FEFF·k²)
    output_html  Where to save interactive χ(k)·k² + χ(R) figure
    """
    try:
        # ------------------------------------------------------------------
        # 1) Load χ(k)·k² file (must have 4 columns)
        # ------------------------------------------------------------------
        data = np.loadtxt(chik_file, skiprows=1)
        if data.shape[1] < 4:
            raise ValueError(f"{chik_file} has {data.shape[1]} columns - need ≥4")

        k               = data[:, 0]
        chi_exp_k2      = data[:, 1]
        chi_feff_k2     = data[:, 3].reshape(-1, 1)   # FEFF/Larch column (#4)

        # ------------------------------------------------------------------
        # 2) Determine number of inputs & prepare input(s)
        # ------------------------------------------------------------------
        print(f"[INFO] Model input count: {len(model.inputs)}")
        if len(model.inputs) == 1:
            # blind mode (k only) or theory-only
            if hasattr(scaler_X, 'transform'):
                # theory-only or blind (with shared scaler)
                X_input = scaler_X.transform(chi_feff_k2)
            else:
                raise ValueError("scaler_X must be a transformer or dict of scalers")
        elif len(model.inputs) == 2:
            if not isinstance(scaler_X, dict) or "k" not in scaler_X or "feff" not in scaler_X:
                raise ValueError("scaler_X must be a dict with keys 'k' and 'feff' for hybrid model")

            k_scaled    = scaler_X["k"].transform(k.reshape(-1, 1))
            feff_scaled = scaler_X["feff"].transform(chi_feff_k2)
            X_input     = [k_scaled, feff_scaled]
        else:
            raise ValueError(f"Unsupported model with {len(model.inputs)} inputs")

        # ------------------------------------------------------------------
        # 3) PINN prediction on χ_FEFF·k²
        # ------------------------------------------------------------------
        y_pred_scaled = model.predict(X_input, verbose=0)
        chi_pinn_k2   = scaler_y.inverse_transform(y_pred_scaled).flatten()

        # ------------------------------------------------------------------
        # 4) Save χ(k)·k² (exp, FEFF, PINN)
        # ------------------------------------------------------------------
        base_path = Path(chik_file)
        out_dir = base_path.parent
        base_name = base_path.stem.replace("_chik_k2", "")

        k2_file = out_dir / f"{base_name}_pinn_prediction_chi-k2.dat"
        k2_data = np.column_stack([k, chi_exp_k2, chi_feff_k2, chi_pinn_k2])
        np.savetxt(k2_file, k2_data, fmt="%.8e", header="k  chi_exp*k2  chi_FEFF/Larch-model*k2  chi_PINNs-prediction*k2")
        print(f"[INFO] Saved χ(k)·k² → {k2_file}")

        # ------------------------------------------------------------------
        # 5) Build χ(k)·k² Plotly figure------------------------------------
        fig_k = go.Figure()
        fig_k.add_scatter(x=k, y=chi_exp_k2,       mode="lines+markers",
                          name="Expt. χ(k)·k²", line=dict(color='lightsalmon', width=4), marker=dict(size=7, color="lightsalmon"))
        fig_k.add_scatter(x=k, y=chi_feff_k2.flatten(), mode="lines",
                          name="FEFF/Larch χ(k)·k²", line=dict(color="mediumseagreen", width=4))
        fig_k.add_scatter(x=k, y=chi_pinn_k2, mode="lines",
                          name="PINNs χ(k)·k²", line=dict(color="red", width=4))
        fig_k.update_layout(
            title="EXAFS χ(k)·k² — Experimental vs FEFF/Larch vs PINNs Prediction",
            xaxis_title="k (Å⁻¹)",
            yaxis_title="k²χ(k) (Å⁻²)",
            template="plotly_dark",
            font=dict(size=26),
            legend=dict(
                font=dict(size=24),  # ← Increase legend font size here
                bordercolor="gray",
                borderwidth=1,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            xaxis=dict(tickfont=dict(size=24)),
            yaxis=dict(tickfont=dict(size=24))
        )


        # ------------------------------------------------------------------
        # 6) Prepare plain χ(k) for FT  (divide by k², handle k=0 safely)---
        safe_k          = np.where(k == 0, 1e-9, k)
        chi_exp         = chi_exp_k2 / safe_k**2
        chi_feff        = chi_feff_k2.flatten() / safe_k**2
        chi_pinn        = chi_pinn_k2 / safe_k**2

        # χ(k) → χ(R) via xftf  (needs a Group container)----------------
        def ft_to_r(k_arr, chi_arr, label):
            g = Group()
            xftf(k_arr, chi_arr, group=g,
                 kmin=2.0, kmax=12.0, dk=4.0,
                 kweight=2, window="kaiser")
            if g.r is None:
                raise RuntimeError(f"xftf failed for {label}")
            return g.r, g.chir_mag, g.chir_re

        r_exp,  chir_exp_mag,  chir_exp_re  = ft_to_r(k, chi_exp,  "Experimental")
        r_feff, chir_feff_mag, chir_feff_re = ft_to_r(k, chi_feff, "FEFF/Larch")
        r_pinn, chir_pinn_mag, chir_pinn_re = ft_to_r(k, chi_pinn, "PINNs")

        # ------------------------------------------------------------------
        # 7) Save χ(R)------------------------------------------------------ 
        r_file = out_dir / f"{base_name}_pinn_prediction_chir.dat"
        r_data = np.column_stack([r_exp,
                                  chir_exp_mag,  chir_exp_re,
                                  chir_feff_mag, chir_feff_re,
                                  chir_pinn_mag, chir_pinn_re])
        np.savetxt(r_file, r_data, fmt="%.8e",
                   header=("R  |Chir_mag_Exp|  Chir-Re_Exp  "
                           "|Chir_mag_FEFF/Larch-model|  Chir-Re_FEFF/larch-model  "
                           "|Chir_mag_PINNs-prediction|  Chir-Re_PINNs-prediction"))
        print(f"[INFO] Saved |χ(R)| & χ(R)Re → {r_file}")

        # ------------------------------------------------------------------
        # 8) Build combined χ(k) + χ(R) figure and write HTML
        # ------------------------------------------------------------------
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Chi(k).k²", "Chi(R): Magnitude & Real Parts"])
        
        # Increase subplot title font size
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=24)  # Change to desired size

        # ---- panel 1  χ(k)·k²
        for trace in fig_k.data:
            fig.add_trace(trace, row=1, col=1)

        # ---- panel 2  χ(R)
        fig.add_scatter(x=r_exp,  y=chir_exp_mag,  mode="lines+markers",
                        name="Expt. |χ(R)|",  line=dict(color='lightsalmon', width=4), marker=dict(size=7, color="lightsalmon"), row=1, col=2)
        fig.add_scatter(x=r_feff, y=chir_feff_mag, mode="lines",
                        name="FEFF/Larch |χ(R)|", line=dict(color="mediumseagreen", width=4),  row=1, col=2)
        fig.add_scatter(x=r_pinn, y=chir_pinn_mag, mode="lines",
                        name="PINNs |χ(R)|", line=dict(color="red", width=4),    row=1, col=2)

        fig.add_scatter(x=r_exp,  y=chir_exp_re,  mode="lines+markers",
                        name="Expt. Re χ(R)", line=dict(color='lightsalmon', dash="dot", width=4), marker=dict(size=7, color="lightsalmon"), row=1, col=2)
        fig.add_scatter(x=r_feff, y=chir_feff_re, mode="lines",
                        name="FEFF/Larch Re χ(R)", line=dict(color="mediumseagreen",  dash="dot", width=4), row=1, col=2)
        fig.add_scatter(x=r_pinn, y=chir_pinn_re, mode="lines",
                        name="PINNs Re χ(R)", line=dict(color="red", dash="dot", width=4), row=1, col=2)

        fig.update_layout(
            title="EXAFS Simulation - Experimental vs FEFF/Larch Fit vs PINNs Prediction",
            xaxis_title="k (Å⁻¹)",
            yaxis_title="k²χ(k) (Å⁻²)",
            xaxis2_title="R (Å)",
            yaxis2_title="χ(R) (Å⁻³)",
            template='plotly_dark',
            font=dict(size=26),
            legend=dict(
                font=dict(size=24),   # ← Larger legend text
                bordercolor="gray",
                borderwidth=1,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            xaxis=dict(tickfont=dict(size=24)),
            yaxis=dict(tickfont=dict(size=24)),
            xaxis2=dict(tickfont=dict(size=24)),
            yaxis2=dict(tickfont=dict(size=24))
        )
      

        fig.write_html(output_html)
        print(f"[INFO] Combined PINNs and FEFF/Larch Fit - χ(k)+χ(R) - plot saved → {output_html}")

    except Exception as exc:
        print(f"[ERROR] PINNs prediction failed for {chik_file}: {exc}")



# --------------------------------------------------------------------------
# Optional: render CIF as interactive 3D HTML using py3Dmol
# --------------------------------------------------------------------------
def render_structure_3D(cif_file, output_image_path):
    try:
        structure = Structure.from_file(cif_file)
        cif_str = structure.to(fmt="cif")
        view = py3Dmol.view(width=400, height=400)
        view.addModel(cif_str, 'cif')
        view.setStyle({'stick': {}})
        view.zoomTo()
        html_path = output_image_path.replace('.png', '.html')
        with open(html_path, 'w') as f:
            f.write(view._make_html())
        print(f"[INFO] Interactive 3D HTML saved to {html_path}")
    except Exception as e:
        print(f"[ERROR] Failed to render 3D structure for {cif_file}: {e}")

# --------------------------------------------------------------------------
# Main driver --------------------------------------------------------------
# --------------------------------------------------------------------------
def main():
    abs_atom = ABSORBER_ATOM
    feff_radius = FEFF_RADIUS
    current_dir = os.getcwd()
    cif_files = [f for f in os.listdir(current_dir) if f.endswith(".cif")]
    if not cif_files:
        print("[ERROR] No CIF files found in directory.")
        return
    for cif_file in cif_files:
        print(f"\n[INFO] Starting FEFF run for {cif_file}")
        DFT_path = os.path.splitext(cif_file)[0]
        os.makedirs(DFT_path, exist_ok=True)
        mat_obj = Struct2XAS(file=cif_file, abs_atom=abs_atom)
        args = [(i, mat_obj, DFT_path, abs_atom, feff_radius) for i in range(mat_obj.nabs_sites)]
        with ThreadPool(cpu_count()) as pool:
            pool.map(run_feff_for_site, args)
        for i in range(mat_obj.nabs_sites):
            feff_dir = os.path.join(DFT_path, f'feff_site_{i}')
            if os.path.exists(feff_dir):
                perform_fitting(feff_dir)
                chik_file = os.path.join(feff_dir, f"{os.path.basename(feff_dir)}_chik_k2.dat")
                exp_file = EXP_DATA_FILE
                output_html = os.path.join(feff_dir, f"{os.path.basename(feff_dir)}_pinn_prediction_plot.html")
                if os.path.exists(chik_file):
                    predict_with_pinn(model, scaler_X, scaler_y, chik_file, output_html)
        structure_img = os.path.join(feff_dir, f'{os.path.basename(feff_dir)}_structure.png')
        render_structure_3D(cif_file, structure_img)

if __name__ == "__main__":
    main()
