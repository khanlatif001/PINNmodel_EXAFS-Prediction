# Physics-Informed Neural Networks for EXAFS Analysis

This repository provides two integrated workflows using **Physics-Informed Neural Networks (PINNs)** for X-ray Absorption Fine Structure (XAFS) analysis:

1. **Training pipeline** — trains a PINN on EXAFS datasets and evaluates it on unseen data.
2. **Experimental fit & PINN Prediction comparison pipeline** — fits experimental EXAFS data with FEFF + XrayLarch, Model prediction EXAFS using the trained model, and compares results.

> **Note:**  
> - The **first script** (train_pinn_exafs.py) is for **training the PINN model on EXAFS datasets** and testing it on unseen data.  
> - The **second script** (exafs_fit_and_prediction.py) performs **experimental EXAFS data fitting using FEFF + XrayLarch**, then compares the fit results with the PINN model predictions.

---

## 🚀 Features
- End-to-end training and evaluation of a PINN on EXAFS datasets.
- Physics-informed loss functions incorporating theoretical constraints.
- Automatic FEFF input/output generation from CIF crystal structures.
- Experimental data fitting via XrayLarch’s `feffit`.
- Parallel FEFF runs for faster computation.
- Interactive Plotly visualizations.
- Direct comparison between fitted and predicted EXAFS data.

---

## 📂 Repository Structure

├── train_pinn_exafs.py            # Script 1: Train & evaluate PINN model on EXAFS data
├── exafs_fit_and_prediction.py    # Script 2: Experimental fit & model prediction comparison
├── artifacts/                     # Stores trained models & scalers
│   ├── pinn_model.h5
│   └── scalers.pkl
├── Input_data/                    # Theoretical EXAFS χ(k)·k² from CIF files
├── Labels/                        # Experimental EXAFS χ(k)·k² data
├── requirements.txt               # Python dependencies
└── README.md

---

## 🛠 Prerequisites

### System
- **OS**: Linux, macOS, or Windows (Linux recommended for FEFF/XrayLarch integration).
- **RAM**: 8 GB minimum (16+ GB recommended).
- **Optional GPU**: NVIDIA CUDA-compatible GPU for faster TensorFlow training.

---

### External Software
1. **Larch** (Xraylarch)

Install XrayLarch via conda:


conda install -yc conda-forge xraylarch

pip install "xraylarch[larix]"


Python Version 3.8–3.11 with pip or conda.


Python Dependencies Listed in requirements.txt:


Please Install:


pip install -r requirements.txt

Note: Use conda install for heavy packages like TensorFlow, pymatgen, and XrayLarch to avoid build issues.


Quick Setup (Conda)


conda create -n pinn-xafs python=3.10 -y


conda activate pinn-xafs


conda install -c conda-forge numpy scipy pandas plotly pymatgen joblib scikit-learn tensorflow xraylarch -y


pip install "xraylarch[larix]"


pip install py3Dmol


Verify:


python -c "import xraylarch, tensorflow, pymatgen, py3Dmol, plotly; print('imports ok')"


One can also create a virtual environment in Linux and install the above dependencies and packages/softwares


• Install venv
sudo pipX.X install virtualenv
• Create venv


pythonX.X-m venvmyvenv
virtualenv-p pythonX.Xmyvenv


• Activate venv
source myenv /bin/activate


• Deactivate venv
deactivate



📌 Usage


1️⃣ Training the PINN model


Train on EXAFS dataset, save the model, and evaluate on unseen data.


python train_pinn_exafs.py ## Train the PINN Model on EXAFS datasets


python train_pinn_exafs.py --test ## Test on unseen EXAFS data

Output:
Trained model: artifacts/pinn_model.h5


Scalers: artifacts/scalers.pkl


Training logs and metrics in console.


2️⃣ Experimental EXAFS data fit & model Prediction comparison


Use experimental EXAFS data, run FEFF + XrayLarch fitting, predict with the trained PINN, and compare.



python exafs_fit_and_prediction.py ## Compare Experimental EXAFS data fit with XrayLarch and PINN prediction

  
Output:

FEFF folders: cifFile/feff_site_*/

Interactive plots: cifFile/feff_site_*/plots/*.html, save as following in subfolder of each cifFile_folder (feff_site_x):


- feff_site_x_FEFFLarch_fit_plot

  
- feff_site_x_pinn_prediction_plot

Fit and Model Prediction EXAFS Data: Save to the feff_site_x subfolder:

- feff_site_x_pinn_prediction_chi-k2.txt

  
- feff_site_x_pinn_prediction_chir.txt

Fit metrics (parameters) printed to console and save as feff_site_0_feffit_report.tx.

----
Files in Working Directory needed For Experimental EXAFS data fit & model Prediction comparison

*.cif files in the working directory — the script loops over CIFs.

An experimental EXAFS file (background-removed) assigned to EXP_DATA_FILE (default in script: CdS_10K_01.xdi.txt.nor). This path can be absolute or relative.

artifacts/ directory containing:

pinn_model.h5 — your pretrained Keras model.

scalers.pkl — joblib dictionary referencing the scalers used during training.

----

📜 Citation

If you use this repository in your research, please cite:

Latif U. Khan, Physics-Informed Neural Networks (PINNs) for Extended X-ray Absorption Fine Structure (EXAFS) Data Analysis: Fast and Accurate Local Atomic Structure Prediction. BM08-XAFS/XRF Beamline, Synchrotron-light for Experimental Science and Applications in the Middle East (SESAME) 2025.  https://zenodo.org/records/16826935 (DOI 10.5281/zenodo.16826934)



