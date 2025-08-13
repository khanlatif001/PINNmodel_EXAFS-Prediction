# Physics-Informed Neural Networks for EXAFS Analysis

**Author:** Latif Ullah Khan  
**Affiliation:** BM08-XAFS/XRF Beamline, SESAME Synchrotron Light Source  
**Date:** 13 August 2025

This **repository** provides two integrated workflows using **Physics-Informed Neural Networks (PINNs)** for **X-ray Absorption Fine Structure (XAFS)** analysis:

1. **PINN Model training pipeline** — trains a PINN on EXAFS datasets and evaluates it on unseen data.
2. **Experimental fit & PINN Prediction pipeline** - Fit the experimental EXAFS data with **FEFF + XrayLarch**, - Predict EXAFS for the experimental data using the **trained PINN model** (plot comparison).

> **Note:**  
> - The **first script** (train_pinn_exafs.py) is for **training the PINN model on EXAFS datasets** and testing it on unseen data.  
> - The **second script** (exafs_fit_and_prediction.py) performs **experimental EXAFS data fitting using FEFF + XrayLarch**, then compares the fit results with the PINN model predictions.

---

## 🚀 Features
- **End-to-end training and evaluation** of a PINN on EXAFS datasets.
- **Physics-informed loss functions** incorporating theoretical constraints.
- **Multi-site FEFF automation** from CIF files (via `Struct2XAS`).
- **Path sorting by R<sub>eff</sub>** for optimal EXAFS fitting.
- **Simultaneous fitting** of multiple absorber sites in lattice using XrayLarch.
- **PINN model integration** for EXAFS prediction.
- **Direct comparison** between fitted and predicted EXAFS data.
- **Interactive Plotly HTML plots** for χ(k)·k² and χ(R) (magnitude & real part).
- **3D CIF visualization** with py3Dmol.

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

**1.** **FEFF CODES ( FEFF6-lite,  FEFF8-lite)**

**FEFF codes** have implemented in the python sceipt.

Running **FEFF simulations** for all the **absorber sites in `.cif` crystal structures** automatically (via `Struct2XAS`).

**2.** **Larch Package (Xraylarch)**

**Install XrayLarch via conda:**


conda install -yc conda-forge xraylarch


pip install "xraylarch[larix]"


Python Version 3.8–3.11 with pip or conda.


**Python Dependencies Listed in requirements.txt, please Install:**


pip install -r requirements.txt


**Note:** Use conda install for heavy packages like **TensorFlow, pymatgen, and XrayLarch** to avoid build issues - Quick Setup (Conda):


conda create -n pinn-xafs python=3.10 -y


conda activate pinn-xafs


conda install -c conda-forge numpy scipy pandas plotly pymatgen joblib scikit-learn tensorflow xraylarch -y


pip install "xraylarch[larix]"


pip install py3Dmol


**Verify:**


python -c "import xraylarch, tensorflow, pymatgen, py3Dmol, plotly; print('imports ok')"


**One can also create a virtual environment in Linux and install the above dependencies and packages/softwares:**


• **Install venv:**

sudo pipX.X install virtualenv


• **Create venv:**

pythonX.X-m venvmyvenv


virtualenv-p pythonX.Xmyvenv


• **Activate venv:**


source myenv /bin/activate


• **Deactivate venv:**

deactivate

---

📌 **Usage**


1️⃣ **Training the PINN Model**

**Running the Script**
- **Training model** from scratch:
  
  python3 train_pinn_exafs.py

- **Prediction only** (test on unseen data):

  python3 train_pinn_exafs.py --test

This script is implemented **PINN model** on EXAFS data traning/prediction
1. **Data Loading:** Reads paired .dat files from experimental and theoretical folders.
2. **Preprocessing:** Standardizes each input (chik.k2) and output using StandardScaler
3. Saves scalers for use during prediction on unseen data.
4. **Defining and Training model:**  Traning model on **EXAFS datasets** (Input data and Lables).
5. **Trained PINN Model Evaluation and Test:** Evaluating and testing model on unseen data.

**Output:**
- **Trained model:** artifacts/pinn_model.keras


- **Scalers:** artifacts/scalers.pkl


- **Training logs and metrics** in console.

---

2️⃣ **Experimental EXAFS Data Fit & PINN Model Prediction EXAFS**


This script provides a **complete automated workflow** for:
1. Running **FEFF** simulations for all absorber sites in `.cif` crystal structures.
2. Fitting FEFF-generated scattering paths to **experimental EXAFS χ(k)** data using [XrayLarch](https://xraypy.github.io/xraylarch/).
3. Predicting χ(k) and χ(R) using a pretrained **Physics-Informed Neural Network (PINN)** model.
4. Generating **interactive HTML visualizations** of χ(k)·k² and χ(R), and **3D structural renderings** of CIF files.

  
**Output:**

**FEFF folders:** feff_site_*/

**Interactive plots:** feff_site_**/plots.html:**


- feff_site_x_FEFFLarch_fit_plot

  
- feff_site_x_pinn_prediction_plot

**XrayLarch Fit and PINN Model Prediction EXAFS Data:** 

- feff_site_x_pinn_prediction_chi-k2.txt

  
- feff_site_x_pinn_prediction_chir.txt
 

**Fit metrics (parameters):**

feff_site_0_feffit_report.txt


**Files in Working Directory:**

1. ***.cif files** in the working directory — the script loops over CIFs.

2. **Experimental EXAFS** file (background-removed) assigned to EXP_DATA_FILE (default in script: CdS_10K_01.xdi.txt.nor). This path can be absolute or relative.

3. **artifacts/ directory containing:**

- pinn_model.keras — your pretrained Keras model.

- scalers.pkl — joblib dictionary referencing the scalers used during training.

----

📜 **Citation**

If you use this repository in your research, please cite:

Latif U. Khan, Physics-Informed Neural Networks (PINNs) for Extended X-ray Absorption Fine Structure (EXAFS) Data Analysis: Fast and Accurate Local Atomic Structure Prediction. BM08-XAFS/XRF Beamline, Synchrotron-light for Experimental Science and Applications in the Middle East (SESAME) 2025.  https://zenodo.org/records/16826935 (DOI: https://doi.org/10.5281/zenodo.16826934 )



