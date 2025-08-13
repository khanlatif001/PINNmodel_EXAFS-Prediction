#!/usr/bin/env python3

"""
THE MODEL CAN TRAIN SUCCESSFULLY IN HYBRID MODE ONLY! OTHER OPTIONS NO
---------------------------------------------------------------------
train from scratch and save artifacts
python3 hybrid_pinn_exafs_testUnseen.py

run prediction only on new spectra
python3 hybrid_pinn_exafs_test Unseen.py --test
---------------------------------------------------------------------
"""
###########################################################################
"""
Hybrid PINN for EXAFS χ(k)·k² prediction
---------------------------------------

• “theory”  mode  :  χ_FEFF(k)·k²  ➜  χ_exp(k)·k²
• “blind”   mode  :  k-grid        ➜  χ_exp(k)·k²
• “hybrid”  mode  :  [k , χ_FEFF]  ➜  χ_exp(k)·k²  (default)

Set `MODE = "hybrid" | "blind" | "theory"` in `main()`.
"""

# ──────────────────────────────────────────────────────────────────────────
#Important NOTES
#  Run Script on test or train
#train from scratch and save artifacts
#python3 hybrid_pinn_exafs_testUnseen.py

#run prediction only on new spectra
#python3 hybrid_pinn_exafs_test Unseen.py --test
# ──────────────────────────────────────────────────────────────────────────
#-------------------------------------------------------------------------------
# ──────────────────────────────────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────────────────────────────────
import os, logging, argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint,
                                        ReduceLROnPlateau, EarlyStopping)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")

# ──────────────────────────────────────────────────────────────────────────
#  1 · Data loading
# ──────────────────────────────────────────────────────────────────────────
"""
Extracts:
    k      : k values from experimental data
    feff   : χ_FEFF·k² values from theoretical files
    y      : χ_exp·k² values from experimental files
Returns NumPy arrays of:
    k (input)
    feff (input)
    y = χ_exp·k² (target output)
Which arrays are included depends on the mode:
    "blind": uses only k as input
    "theory": uses only feff as input
    "hybrid": uses both k and feff
___________________________________________________________________________   
THE MODEL CAN TRAIN SUCCESSFULLY IN HYBRID MODE ONLY! OTHER OPTIONS NO
___________________________________________________________________________
"""
def load_paired_data(exp_folders, theo_folders, mode="hybrid"):  
    k_list, feff_list, y_list = [], [], []

    for exp_dir, theo_dir in zip(exp_folders, theo_folders):
        exp_files  = sorted(Path(exp_dir).glob("*.dat"))
        theo_files = sorted(Path(theo_dir).glob("*.dat"))
        for exp_fp, theo_fp in zip(exp_files, theo_files):
            exp = np.loadtxt(exp_fp)          # cols: k , χ_exp·k²
            theo = np.loadtxt(theo_fp)        # cols: k , χ_FEFF·k²
            if exp.shape != theo.shape or exp.shape[1] != 2:
                logging.warning(f"Shape mismatch: {exp_fp.name}")
                continue
            if mode != "theory":
                k_list.append(exp[:, 0:1])           # (N_i,1)
            if mode != "blind":
                feff_list.append(theo[:, 1:2])       # (N_i,1)
            y_list.append(exp[:, 1])                 # (N_i,)

    if not y_list:
        raise RuntimeError("No valid spectra found.")

    y = np.concatenate(y_list).astype(np.float32)
    k    = np.concatenate(k_list).astype(np.float32)    if k_list   else None
    feff = np.concatenate(feff_list).astype(np.float32) if feff_list else None
    return k, feff, y

# ──────────────────────────────────────────────────────────────────────────
#  2 · Scaling / preprocessing
# ──────────────────────────────────────────────────────────────────────────
def preprocess_hybrid(k, feff, y):
    scaler_k = scaler_feff = None
    if k is not None:
        scaler_k = StandardScaler()
        k = scaler_k.fit_transform(k).astype(np.float32)
    if feff is not None:
        scaler_feff = StandardScaler()
        feff = scaler_feff.fit_transform(feff).astype(np.float32)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1,1)).astype(np.float32)
    return k, feff, y, scaler_k, scaler_feff, scaler_y

# ──────────────────────────────────────────────────────────────────────────
#  3 · Model
# ──────────────────────────────────────────────────────────────────────────
def create_hybrid_pinn(input_dim_k=None, input_dim_feff=None,
                       hidden_layers=3, neurons=64, mode="hybrid"):  # changed hidden_layers from 3 and neurons from 64
    inputs, branches = [], []

    if mode != "theory":
        k_in = Input(shape=(input_dim_k,), name="k_input")
        inputs.append(k_in); branches.append(k_in)
    if mode != "blind":
        f_in = Input(shape=(input_dim_feff,), name="feff_input")
        inputs.append(f_in); branches.append(f_in)

    x = branches[0] if len(branches)==1 else Concatenate()(branches)
    for _ in range(hidden_layers):
        x = Dense(neurons, activation="tanh")(x)
    out = Dense(1)(x)
    return Model(inputs, out, name=f"PINN_{mode}")

# ──────────────────────────────────────────────────────────────────────────
#  4 · Physics‑informed term (∂²χ/∂k²)
# ──────────────────────────────────────────────────────────────────────────
def physics_informed_loss(model, k_tensor, feff_tensor, mode):
    if k_tensor is None:          # “theory” mode
        return tf.constant(0.0, tf.float32)
    k = tf.cast(k_tensor,   tf.float32)
    feff = tf.cast(feff_tensor, tf.float32) if feff_tensor is not None else None

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(k)
        with tf.GradientTape() as tape1:
            tape1.watch(k)
            if mode == "blind":
                y_pred = model([k], training=True)
            elif mode == "hybrid":
                y_pred = model([k, feff], training=True)
            else:  # theory
                return tf.constant(0.0, tf.float32)
        dy_dk = tape1.gradient(y_pred, k)
    d2y_dk2 = tape2.gradient(dy_dk, k)
    return tf.reduce_mean(tf.square(d2y_dk2))

# ──────────────────────────────────────────────────────────────────────────
#  5 · Training model
# ──────────────────────────────────────────────────────────────────────────
@tf.function
def train_step(model, opt,
               k_batch, feff_batch, y_batch,
               λ_data, λ_phys, mode):

    # enforce float32
    if k_batch   is not None: k_batch   = tf.cast(k_batch,   tf.float32)
    if feff_batch is not None: feff_batch = tf.cast(feff_batch, tf.float32)
    y_batch = tf.cast(y_batch, tf.float32)

    with tf.GradientTape() as tape:
        if mode == "blind":
            y_pred = model([k_batch], training=True)
        elif mode == "theory":
            y_pred = model([feff_batch], training=True)
        else:  # hybrid
            y_pred = model([k_batch, feff_batch], training=True)

        data_loss = tf.reduce_mean(tf.square(y_batch - y_pred))
        phys_loss = physics_informed_loss(model, k_batch, feff_batch, mode)
        loss = λ_data*data_loss + λ_phys*phys_loss

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss, data_loss, phys_loss

def train_pinn(model, k_tr, feff_tr, y_tr,
               k_val, feff_val, y_val,
               epochs=1500, batch_size=256,
               λ_data=1.0, λ_phys=1e-2, mode="hybrid"):
    ds = tf.data.Dataset.from_tensor_slices((
        k_tr.astype(np.float32)   if k_tr   is not None else np.zeros_like(y_tr),
        feff_tr.astype(np.float32) if feff_tr is not None else np.zeros_like(y_tr),
        y_tr.astype(np.float32)))
    ds = ds.shuffle(len(y_tr)).batch(batch_size, drop_remainder=False)  

    opt = Adam(1e-3)
    history = {"loss":[], "val_loss":[], "phys":[], "val_phys":[]}

    for epoch in range(1, epochs+1):
        epoch_loss = epoch_phys = 0.0
        for k_b, f_b, y_b in ds:
            loss, d_loss, p_loss = train_step(model, opt, k_b, f_b, y_b,
                                              λ_data, λ_phys, mode)
            epoch_loss += loss.numpy()*len(y_b)
            epoch_phys += p_loss.numpy()*len(y_b)
        epoch_loss /= len(y_tr); epoch_phys /= len(y_tr)

        # validation
        phys_val = physics_informed_loss(model, k_val, feff_val, mode).numpy()
        if mode=="blind":
            val_pred = model.predict([k_val.astype(np.float32)], verbose=0)
        elif mode=="theory":
            val_pred = model.predict([feff_val.astype(np.float32)], verbose=0)
        else:
            val_pred = model.predict([k_val.astype(np.float32), feff_val.astype(np.float32)], verbose=0)
        val_loss = np.mean((y_val - val_pred)**2)

        history["loss"].append(epoch_loss)
        history["val_loss"].append(val_loss) 
        history["phys"].append(epoch_phys)
        history["val_phys"].append(phys_val)

        if epoch % 100 == 0 or epoch == 1:
            logging.info(f"Epoch {epoch:4d}  "
                         f"loss={epoch_loss:.4e}  "
                         f"val={val_loss:.4e}  "
                         f"phys={epoch_phys:.2e}  "
                         f"val_phys={phys_val:.2e}")
    return history

# ──────────────────────────────────────────────────────────────────────────
#  6 · Evaluation & plots
# ──────────────────────────────────────────────────────────────────────────
def evaluate(model, k_te, feff_te, y_te, scaler_y, mode):
    if mode=="blind":
        y_pred = model.predict([k_te.astype(np.float32)], verbose=0)
    elif mode=="theory":
        y_pred = model.predict([feff_te.astype(np.float32)], verbose=0)
    else:
        y_pred = model.predict([k_te.astype(np.float32), feff_te.astype(np.float32)], verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred.astype(np.float32))
    y_true = scaler_y.inverse_transform(y_te.astype(np.float32))
    mse = mean_squared_error(y_true, y_pred)
    logging.info(f"Test  MSE = {mse:.6e}")
    return y_true, y_pred, mse

def plot_prediction(k_vals, y_true, y_pred, save_html="EXAFS-prediction_plot.html"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_vals, y=y_true, name="Experimental EXAFS",
                             mode="lines", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=k_vals, y=y_pred, name="Predicted EXAFS",
                             mode="lines", line=dict(width=3)))
    fig.update_layout(template="plotly_dark",
                      xaxis_title="k (Å⁻¹)",
                      yaxis_title="k²χ(k) (Å⁻²)",
                      title="PINN prediction")
    fig.write_html(save_html)
    logging.info(f"Saved plot ➜ {save_html}")

# ──────────────────────────────────────────────────────────────────────────
#  7 · Save model output
# ──────────────────────────────────────────────────────────────────────────
def save_artifacts(model, scaler_k, scaler_feff, scaler_y, out_dir="artifacts"):
    Path(out_dir).mkdir(exist_ok=True)
    model.save(Path(out_dir)/"pinn_model.keras")
    joblib.dump({"k":scaler_k, "feff":scaler_feff, "y":scaler_y},
                Path(out_dir)/"scalers.pkl")
    logging.info(f"Saved model & scalers to {out_dir}")

# ──────────────────────────────────────────────────────────────────────────
#  8 · Test model on new unknown data
# ──────────────────────────────────────────────────────────────────────────
def test_on_unseen(model_path, scaler_path,
                   test_exp_folders, test_theo_folders, mode="hybrid",
                   output_html="unseen_test_EXAFS-prediction_plot.html", 
                   output_dat="unseen_test_EXAFS-prediction_output.dat"):

    k, feff, y = load_paired_data(test_exp_folders, test_theo_folders, mode=mode)
    scalers = joblib.load(scaler_path)
    sk, sf, sy = scalers["k"], scalers["feff"], scalers["y"]
    if k is not None: k = sk.transform(k).astype(np.float32)
    if feff is not None: feff = sf.transform(feff).astype(np.float32)
    y_scaled = sy.transform(y.reshape(-1, 1)).astype(np.float32)

    model = tf.keras.models.load_model(model_path, compile=False)

    if mode == "blind":
        y_pred = model.predict([k], verbose=0)
    elif mode == "theory":
        y_pred = model.predict([feff], verbose=0)
    else:
        y_pred = model.predict([k, feff], verbose=0)

    y_pred_unscaled = sy.inverse_transform(y_pred)
    y_true_unscaled = sy.inverse_transform(y_scaled)

    # save prediction
    k_unscaled = scalers["k"].inverse_transform(k) if k is not None else np.zeros_like(y).reshape(-1,1)
    np.savetxt(output_dat, np.column_stack([
    k_unscaled,
    y_true_unscaled, y_pred_unscaled]),
    header="k  y_true  y_pred")

    # Plot 
    k_plot = scalers["k"].inverse_transform(k).flatten() if k is not None else np.arange(len(y))
    plot_prediction(k_plot, y_true_unscaled.flatten(), y_pred_unscaled.flatten(),
                    save_html=output_html)

    logging.info(f"Saved unseen test results: {output_dat}")

# ──────────────────────────────────────────────────────────────────────────
#  9 · Plot training histroy & Save
# ──────────────────────────────────────────────────────────────────────────
# Plot
def plot_training_history(history, save_path="training_history_plot.html"):
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(y=history["loss"],     name="Train Loss",      line=dict(width=5)))
    fig.add_trace(go.Scatter(y=history["val_loss"], name="Val Loss",        line=dict(width=5)))
    fig.add_trace(go.Scatter(y=history["phys"],     name="Train Physics",   line=dict(width=5, dash="dot")))
    fig.add_trace(go.Scatter(y=history["val_phys"], name="Val Physics",     line=dict(width=5, dash="dot")))

    # Proper tick font and axis title formatting
    fig.update_layout(
        template="plotly_white",
        title=dict(text="PINNs Training History", font=dict(size=28)),
        xaxis=dict(
            title=dict(text="Epoch", font=dict(size=30)),
            tickfont=dict(size=28),
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text="Loss", font=dict(size=28)),
            tickfont=dict(size=28),
            showgrid=True
        ),
        font=dict(size=28),  # General font
        legend=dict(x=0, y=1.05, orientation="h", font=dict(size=28)),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    fig.write_html(save_path)
    logging.info(f"Saved training history plot ➜ {save_path}")


# Save
def save_training_history_txt(history, save_path="training_history.txt"):
    # Stack values into columns: epoch, loss, val_loss, phys, val_phys
    num_epochs = len(history["loss"])
    data = np.column_stack((
        np.arange(1, num_epochs + 1),
        history["loss"],
        history["val_loss"],
        history["phys"],
        history["val_phys"]
    ))
    
    # Define header
    header = "Epoch\tTrain_Loss\tVal_Loss\tTrain_Phys\tVal_Phys"
    
    # Save to text file
    np.savetxt(save_path, data, header=header, fmt="%.6e", delimiter="\t")  

    print(f"Saved training history data ➜ {save_path}")


# ──────────────────────────────────────────────────────────────────────────
#  10 · Main script
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hybrid PINN training or testing")
    parser.add_argument("--test", action="store_true", help="Run prediction on unseen data")
    args = parser.parse_args()
    # ------------------------
    MODE = "hybrid"         # "blind" | "theory" | "hybrid"
    EPOCHS = 1500
    # ------------------------

    if args.test:
        # Testing mode
        test_on_unseen(
            model_path="artifacts/pinn_model.keras",
            scaler_path="artifacts/scalers.pkl",
            test_exp_folders=["C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTest_Datasets/Unknown_labels"],
            test_theo_folders=["C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTest_Datasets/Unknown_input"],
            mode=MODE,
            output_html="unseen_test_EXAFS-prediction_plot.html",
            output_dat="unseen_test_EXAFS-prediction_output.dat"
        )
        return

    # ❶  Define your data folders  ── adjust to your paths
    experimental_folders = [
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/CuO",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/Cu",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/ZnS",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/CdS",
        #"C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/Y2O3",
        #"C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/LiCoO2",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/Pt",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/Zr",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/ZnO_exp",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/CuO_exp",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/ZnFoil_exp",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Labels/ZnSe_exp"
    ]
    theoretical_folders  = [
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/CuO",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/Cu",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/ZnS",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/CdS",
        #"C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/Y2O3",
        #"C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/LiCoO2",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/Pt",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/Zr",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/ZnO_exp",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/CuO_exp",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/ZnFoil_exp",
        "C:/Users/latifkhan/Documents\Files_Latif/machineLearning/modelTraining_Datasets/Input_data/ZnSe_exp"
    ]

    k, feff, y = load_paired_data(experimental_folders,
                                  theoretical_folders,
                                  mode=MODE)

    k, feff, y, sk, sf, sy = preprocess_hybrid(k, feff, y)
    k_tr, k_te, feff_tr, feff_te, y_tr, y_te = train_test_split(
        k, feff, y, test_size=0.15, random_state=42)
    
    # OTHER OPTIONS NOT RECOMMENDED: Train /‑ test split options with different modes
    #if MODE == "blind":              # k only
    #    k_tr, k_te, y_tr, y_te = train_test_split(
    #        k, y, test_size=0.15, random_state=42)
    #    feff_tr = feff_te = None
#
    #elif MODE == "theory":           # feff only
    #    feff_tr, feff_te, y_tr, y_te = train_test_split(
    #        feff, y, test_size=0.15, random_state=42)
    #    k_tr = k_te = None
#
    #else:                            # hybrid  (k + feff)
    #    k_tr, k_te, feff_tr, feff_te, y_tr, y_te = train_test_split(
    #        k, feff, y, test_size=0.15, random_state=42)


    model = create_hybrid_pinn(input_dim_k=None if k_tr is None else k_tr.shape[1],
                               input_dim_feff=None if feff_tr is None else feff_tr.shape[1],
                               mode=MODE)
    model.summary(print_fn=logging.info)

    history = train_pinn(model, k_tr, feff_tr, y_tr,
                         k_te, feff_te, y_te,
                         epochs=EPOCHS, mode=MODE)
    
    #   Plot and save training history
    plot_training_history(history, save_path="training_history_plot.html")
    save_training_history_txt(history, "training_history.txt")


    #   Evaluation on held‑out test set
    y_true, y_pred, mse = evaluate(model, k_te, feff_te, y_te, sy, MODE)
    #   Plot
    k_plot = sk.inverse_transform(k_te).flatten() if k_te is not None else np.arange(len(y_true))
    plot_prediction(k_plot, y_true.flatten(), y_pred.flatten())

    #   Save
    save_artifacts(model, sk, sf, sy)

if __name__ == "__main__":
    main()
