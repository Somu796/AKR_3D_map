import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# didn't got synced well
# %% 1. Importing Libraries
import sys
import typing
from pathlib import Path

import gpflow
import joblib
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import tensorflow as tf
import tf_keras as keras
from gpflow.utilities import print_summary
from sklearn.preprocessing import StandardScaler

mpl.use("Agg")  # no display on HPC
from datetime import datetime

import matplotlib.pyplot as plt

# %% 2. Setting up Environment

# 2. Setting project paths
project_root = Path.cwd()
sys.path.append(str(project_root))

# Define standard data subdirectories for easy access later
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
ASSETS_DIR = project_root / "assets" / "3D_Objects_03"


def gaussian_processes() -> None:
    # 3. Reading the data

    processed_data_base_name = "04_prepared_data_for_classification"
    class_data = pd.read_parquet(
        f"{PROCESSED_DATA_DIR}/{processed_data_base_name}.parquet"
    )

    # 4. Feature selection and engineering
    features = ["x_gse", "y_gse", "z_gse"]
    X_raw = class_data[features].to_numpy().astype("float64")
    y_raw = class_data[["AKR_Observed"]].to_numpy().astype("float64")

    # 5. Scaling (Critical for GP Kernels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 6. Creating the TensorFlow Dataset
    # Using minibatches of 1024 allows us to process 885k points without RAM issues
    batch_size = 1024
    train_dataset = tf.data.Dataset.from_tensor_slices((X_scaled, y_raw))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).repeat()
    train_iter = iter(train_dataset)

    # 7. Inducing Variables
    # Pick 1000 random points from your data to represent the 3D space
    num_inducing = 1000
    inducing_variable = X_scaled[
        np.random.choice(X_scaled.shape[0], num_inducing, replace=False),
        :,
    ]

    # 8. Defining the Model
    kernel = gpflow.kernels.Matern52()  # Smooth but physically realistic
    likelihood = gpflow.likelihoods.Bernoulli()  # For 0/1 probability

    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        num_data=X_scaled.shape[0],
    )
    # 9. Setup Checkpointing (MUST BE HERE)
    optimizer = tf.optimizers.Adam(learning_rate=0.0005)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_dir = project_root / "assets" / "gaussian_processes_model_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manager = tf.train.CheckpointManager(ckpt, str(checkpoint_dir), max_to_keep=1)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    # 10. Optimization Step Function
    @tf.function
    def optimization_step(
        model: gpflow.models.SVGP,
        optimizer: tf.optimizers.Optimizer,
        train_iter: typing.Iterator,
    ) -> tf.Tensor:
        """
        Optimizes the SVGP model using a single minibatch from the iterator.

        All inputs are passed explicitly to avoid S6911 (global variable dependency).
        """
        with tf.GradientTape() as tape:
            loss = model.training_loss(next(train_iter))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # 11. Optimization Loop (Updated for Plotting)

    elbo_log = []
    steps_log = []

    print("Starting Optimization...")
    for step in range(15000):
        loss = optimization_step(model, optimizer, train_iter)

        # We log every 100 steps to keep the plot smooth but the list small
        if step % 100 == 0:
            # loss is negative ELBO, so we append the negative to see the actual ELBO
            elbo_log.append(-loss.numpy())
            steps_log.append(step)

        if step % 500 == 0:
            print(f"Step {step} - ELBO: {-loss.numpy():.4f}")

        if step % 2000 == 0:
            manager.save()

    # 12. Final Save
    manager.save()
    joblib.dump(scaler, str(checkpoint_dir / "scaler.pkl"))
    print_message = f"Training Complete. Model and Scaler saved to {checkpoint_dir}"
    print(print_message)

    # 13. performance metrics
    # Create a DataFrame for easier plotting
    df_elbo = pd.DataFrame({"Step": steps_log, "ELBO": elbo_log})

    # Create the figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_elbo["Step"],
            y=df_elbo["ELBO"],
            mode="lines",
            name="ELBO",
            line={"color": "#1f77b4", "width": 2},
            hovertemplate="<b>Step:</b> %{x}<br><b>ELBO:</b> %{y:.2f}<extra></extra>",
        ),
    )

    # Update layout for a professional dashboard look
    fig.update_layout(
        title="SVGP Model Convergence (ELBO)",
        xaxis_title="Iteration Step",
        yaxis_title="Evidence Lower Bound (ELBO)",
        template="plotly_white",
        hovermode="x unified",
    )

    # Save as interactive HTML for your dashboard
    plot_html_path = checkpoint_dir / "elbo_plot.html"
    fig.write_html(str(plot_html_path))

    # Also save as a static image for reports
    plot_png_path = checkpoint_dir / "elbo_plot.png"
    fig.write_image(str(plot_png_path))

    print(f"Interactive plot saved to: {plot_html_path}")

    # This prints the optimized lengthscales and variance
    print_summary(model)

    # To load later
    # import joblib

    # # Load the "key" to your coordinate system
    # scaler = joblib.load("./assets/gaussian_processes_model_checkpoints/scaler.pkl")

    # # Use .transform() (NOT fit_transform) on new grid points
    # # raw_grid_coords shape should be (N, 3)
    # X_scaled = scaler.transform(raw_grid_coords)

    # import gpflow
    # import tensorflow as tf

    # # A. Rebuild the shell (Must match your training script exactly)
    # kernel = gpflow.kernels.Matern52()
    # likelihood = gpflow.likelihoods.Bernoulli()

    # # Note: inducing_variable shape must match your training (1000, 3)
    # # You can use dummy zeros here; the restore step will fill them with the real values.
    # model = gpflow.models.SVGP(
    #     kernel=kernel,
    #     likelihood=likelihood,
    #     inducing_variable=np.zeros((1000, 3)),
    #     num_data=1,  # Doesn't matter for prediction
    # )

    # # B. Restore the weights
    # ckpt = tf.train.Checkpoint(model=model)
    # manager = tf.train.CheckpointManager(ckpt, "./assets/model_checkpoints", max_to_keep=1)

    # if manager.latest_checkpoint:
    #     ckpt.restore(manager.latest_checkpoint).expect_partial()
    #     print("Model weights successfully loaded.")
    # else:
    #     print("No checkpoint found! Check your paths.")

    # # Predict Y (the binary outcome)
    # # p_mean: Probability of AKR (0.0 to 1.0)
    # # p_var: The uncertainty of the model
    # p_mean, p_var = model.predict_y(X_scaled)

    # # Convert to numpy for plotting or saving to JSON
    # probabilities = p_mean.numpy().flatten()


def bionomial_modeling():
    # 0. LOAD YOUR DATA
    # Replace these paths with your actual data files
    ltrmlat_data = pd.read_parquet(
        f"{ASSETS_DIR}/ltrmlat_akr_grid.parquet",
    )
    checkpoint_dir = project_root / "assets" / "binomial_model_checkpoints"
    # ← your path here

    print(f"Loaded ltrmlat_data: {ltrmlat_data.shape}")

    # ── 1. Residence time threshold ───────────────────────────────────────────────
    # ltrmlat_data has ~97,000 grid cells but most were never visited by Wind.
    # We only want to model cells where Wind spent meaningful time.
    # First isolate cells that were visited at all (residence_time > 0),
    # then take the 75th percentile of those - which corresponds to ~1 full orbit
    # (~23 hrs). Cells below this threshold have too few observations to give
    # reliable burst probability estimates.

    visited = ltrmlat_data[ltrmlat_data["residence_time"] > 0]
    # → ~3,700 cells out of 97,000 were visited

    min_residence = visited["residence_time"].quantile(0.75)
    # → ~82,000 seconds ≈ 23 hours ≈ 1 Wind orbit

    df_model = ltrmlat_data[ltrmlat_data["residence_time"] >= min_residence].copy()
    # → keeps only the top 25% best-sampled cells (~925 cells)
    # .copy() prevents pandas SettingWithCopyWarning on later assignments

    print(
        f"Cells kept: {len(df_model)} / {len(ltrmlat_data)} ({100 * len(df_model) / len(ltrmlat_data):.1f}%)"
    )

    # ── 2. Circular local time encoding ──────────────────────────────────────────
    # Local time (LT) runs 0-24 hrs and is circular: LT=0 and LT=24 are the same.
    # A linear model doesn't know this - it would treat 0 and 23 as far apart.
    # Solution: encode LT as (sin, cos) on a unit circle so the model sees
    # 23:00 and 01:00 as neighbours, which they physically are.
    #
    #   LT=0  → sin=0,  cos=1   (noon)
    #   LT=6  → sin=1,  cos=0   (dusk)
    #   LT=12 → sin=0,  cos=-1  (midnight)
    #   LT=18 → sin=-1, cos=0   (dawn)

    df_model["lt_sin"] = np.sin(2 * np.pi * df_model["lt"] / 24)
    df_model["lt_cos"] = np.cos(2 * np.pi * df_model["lt"] / 24)

    # ── 3. Zone indicator flags ───────────────────────────────────────────────────
    # These are binary (0.0 / 1.0) flags that tell the model whether a cell
    # is in a physically important region. They allow the model to learn a
    # different intercept for each zone, rather than assuming a smooth global trend.

    # Nightside: LT between 18:00-24:00 or 00:00-06:00
    # AKR is known to be strongest on the nightside (away from the Sun)
    lt_night_mask = (df_model["lt"] >= 18) | (df_model["lt"] <= 6)
    df_model["is_nightside"] = lt_night_mask.astype(float)
    # 1.0 = nightside, 0.0 = dayside

    # Auroral zone: |mlat| between 65° and 80°
    # AKR is generated along auroral field lines in this latitude band
    mlat_auroral_mask = df_model["mlat"].abs().between(65, 80)
    df_model["is_auroral"] = mlat_auroral_mask.astype(float)
    # 1.0 = auroral zone, 0.0 = outside auroral zone

    # Inner magnetosphere: radial distance between 2 and 8 Earth radii
    # Wind's orbit keeps it mostly here; this is where AKR source regions map to
    r_inner_mask = df_model["r"].between(2, 8)
    df_model["is_inner_mag"] = r_inner_mask.astype(float)
    # 1.0 = inner magnetosphere, 0.0 = further out

    # ── 4. Radial distance from peak ─────────────────────────────────────────────
    # AKR probability doesn't increase linearly with r - it peaks at a specific
    # shell and drops off on either side. Rather than assume where the peak is,
    # we find it from the data: the r value where normalised_observation_time
    # is highest (i.e. where Wind observed AKR most relative to how long it was there).
    # We then compute how far each cell is from that peak radius.
    # This lets the model fit a parabolic-like shape in r without needing r².

    r_peak = ltrmlat_data.loc[ltrmlat_data["normalised_observation_time"].idxmax(), "r"]
    # idxmax() → row index of the maximum value
    # .loc[index, "r"] → the r value at that row
    # Computed on full ltrmlat_data (not filtered df_model) so the reference
    # point isn't biased by the residence time filter

    df_model["r_dist_from_peak"] = (df_model["r"] - r_peak).abs()
    # e.g. r_peak=5.2 Re, cell at r=3.0 → dist=2.2 Re
    # e.g. r_peak=5.2 Re, cell at r=7.0 → dist=1.8 Re
    print(f"R peak: {r_peak:.2f} Re")

    # ── 5. MLat features - hemisphere-aware ──────────────────────────────────────
    # Magnetic latitude has three distinct pieces of information:
    #   a) How far from the equator (mlat_abs) - magnitude of auroral activity
    #   b) Which hemisphere (mlat_north) - north/south can be asymmetric
    #   c) How far from the active zone peak (mlat_dist_from_peak) - proximity
    #      to where AKR is most commonly observed
    #
    # We compute the peak mlat separately for north and south because the
    # magnetosphere is not perfectly symmetric - the auroral oval can sit at
    # different latitudes in each hemisphere.

    df_model["mlat_abs"] = df_model["mlat"].abs()
    df_model["mlat_north"] = (df_model["mlat"] > 0).astype(float)
    # mlat_abs:   e.g. mlat=−72° → 72°
    # mlat_north: 1.0 = northern hemisphere, 0.0 = southern hemisphere

    # Find peak separately in each hemisphere from the full unfiltered dataset
    north_data = ltrmlat_data[ltrmlat_data["mlat"] > 0]
    south_data = ltrmlat_data[ltrmlat_data["mlat"] < 0]

    mlat_peak_north = north_data.loc[
        north_data["normalised_observation_time"].idxmax(), "mlat"
    ]  # positive scalar, e.g. +71.0°

    mlat_peak_south = abs(
        south_data.loc[south_data["normalised_observation_time"].idxmax(), "mlat"]
    )  # converted to positive for comparison with mlat_abs, e.g. 68.0°

    print(f"MLat peak - North: {mlat_peak_north:.1f}°   South: {mlat_peak_south:.1f}°")

    # Signed distance from hemisphere-specific peak:
    #   negative → equatorward of the peak  (too close to equator)
    #   positive → poleward of the peak     (too close to pole)
    #   zero     → right at the peak latitude
    df_model["mlat_dist_from_peak"] = df_model.apply(
        lambda row: row["mlat_abs"]
        - (mlat_peak_north if row["mlat_north"] == 1.0 else mlat_peak_south),
        axis=1,
    )
    # e.g. north cell at mlat=65°, peak=71° → 65−71 = −6° (equatorward)
    # e.g. north cell at mlat=75°, peak=71° → 75−71 = +4° (poleward)

    # ── 6. Interaction terms ──────────────────────────────────────────────────────
    # A linear model with separate features can't capture "the effect of mlat
    # depends on what LT you're at". Interaction terms multiply two features
    # together, letting the model learn joint effects.

    # LT × MLat: the auroral oval shifts in latitude depending on local time
    # (e.g. it sits higher at midnight than at noon)
    df_model["lt_sin_x_mlat"] = df_model["lt_sin"] * df_model["mlat_abs"]
    df_model["lt_cos_x_mlat"] = df_model["lt_cos"] * df_model["mlat_abs"]

    # Nightside AND auroral: captures the known AKR hotspot
    # (being nightside alone or auroral alone is less predictive than both)
    df_model["night_x_auroral"] = df_model["is_nightside"] * df_model["is_auroral"]
    # 1.0 only when both is_nightside=1 AND is_auroral=1

    # All three together: the most specific AKR source region
    df_model["night_x_auroral_x_inner"] = (
        df_model["is_nightside"] * df_model["is_auroral"] * df_model["is_inner_mag"]
    )
    # 1.0 only when nightside + auroral zone + inner magnetosphere all true

    # ── 7. Trials and successes ───────────────────────────────────────────────────
    # The binomial GLM needs integer counts, not continuous times.
    # We divide by 183 seconds (≈ one Wind spin period / observation window)
    # to convert time into discrete "how many chances did Wind have".
    #
    #   n_trials  = how many observation windows fit in the residence time
    #             = how many times Wind *could* have detected AKR
    #   n_success = how many windows actually contained AKR
    #             = how many times Wind *did* detect AKR
    #
    # The model then learns: P(AKR detected | location features)

    df_model["n_trials"] = (df_model["residence_time"] / 183).astype(int)
    df_model["n_success"] = (df_model["observation_time"] / 183).astype(int)

    # Safety: n_success can never exceed n_trials (guards against rounding errors)
    df_model["n_success"] = df_model["n_success"].clip(upper=df_model["n_trials"])

    # Drop any remaining cells with zero trials (no information content)
    df_model = df_model[df_model["n_trials"] > 0]

    # ── 8. Feature list ───────────────────────────────────────────────────────────
    # These are the columns passed to the GLM.
    # Grouped by what physical question each feature answers:

    features = [
        # Where is Wind in local time? (circular, split into two components)
        "lt_sin",
        "lt_cos",
        # Where is Wind in magnetic latitude?
        "mlat_abs",  # how far from equator
        "mlat_north",  # which hemisphere
        "mlat_dist_from_peak",  # how far from the active auroral zone
        # Where is Wind radially?
        "r",  # raw distance in Earth radii
        "r_dist_from_peak",  # distance from the most-observed shell
        # Is Wind in a known active region?
        "is_nightside",  # away from Sun
        "is_auroral",  # in the auroral latitude band
        "is_inner_mag",  # within 2-8 Re
        # Joint effects
        "lt_sin_x_mlat",  # LT shape × latitude magnitude
        "lt_cos_x_mlat",  # LT shape × latitude magnitude
        "night_x_auroral",  # nightside + auroral combined
        "night_x_auroral_x_inner",  # all three source region flags combined
    ]

    X = sm.add_constant(df_model[features])
    # add_constant adds a column of 1s → allows the model to fit an intercept
    # (the baseline log-odds when all features are zero)

    # ── 9. Fit binomial GLM ───────────────────────────────────────────────────────
    # Binomial GLM with logit link models:
    #   log(p / 1−p) = β₀ + β₁·lt_sin + β₂·lt_cos + ...
    # where p = probability that AKR is active in a given cell.
    # Passing [n_success, n_trials] as the response tells statsmodels each row
    # is already aggregated (not individual Bernoulli trials).

    print("\nFitting binomial GLM...")
    model = sm.GLM(
        df_model[["n_success", "n_trials"]],
        X,
        family=sm.families.Binomial(),
    ).fit()
    print("Done.")

    # 10. OUTPUTS

    OUT = Path(checkpoint_dir)
    OUT.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Text summary ──────────────────────────────────────────────────────────────
    null_deviance = model.null_deviance
    model_deviance = model.deviance
    pseudo_r2 = 1 - (model_deviance / null_deviance)

    summary_path = OUT / f"model_summary_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BINOMIAL GLM - AKR BURST PROBABILITY\n")
        f.write(f"Run timestamp : {timestamp}\n")
        f.write(f"Cells in model: {len(df_model)}\n")
        f.write(f"R peak        : {r_peak:.2f} Re\n")
        f.write(f"MLat peak N   : {mlat_peak_north:.1f}°\n")
        f.write(f"MLat peak S   : {mlat_peak_south:.1f}°\n")
        f.write(
            f"Min residence : {min_residence:.0f} s  ({min_residence / 3600:.1f} hrs)\n"
        )
        f.write("=" * 80 + "\n\n")
        f.write(model.summary().as_text())
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("ODDS RATIOS\n")
        f.write("=" * 80 + "\n")
        coef_df = pd.DataFrame(
            {
                "coef": model.params,
                "std_err": model.bse,
                "z": model.tvalues,
                "p_value": model.pvalues,
                "odds_ratio": np.exp(model.params),
                "OR_ci_low": np.exp(model.params - 1.96 * model.bse),
                "OR_ci_high": np.exp(model.params + 1.96 * model.bse),
            }
        )
        f.write(coef_df.round(4).to_string())
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("GOODNESS OF FIT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Null deviance      : {null_deviance:.2f}\n")
        f.write(f"Model deviance     : {model_deviance:.2f}\n")
        f.write(
            f"Deviance reduction : {null_deviance - model_deviance:.2f} "
            f"({100 * (null_deviance - model_deviance) / null_deviance:.1f}%)\n"
        )
        f.write(f"McFadden pseudo-R² : {pseudo_r2:.4f}\n")
        f.write(f"AIC                : {model.aic:.2f}\n")
        f.write(f"BIC                : {model.bic:.2f}\n")

    # ── Predictions CSV ───────────────────────────────────────────────────────────
    df_results = df_model[["lt", "r", "mlat", "n_trials", "n_success"]].copy()
    df_results["p_predicted"] = model.predict(X)
    df_results["p_observed"] = df_model["n_success"] / df_model["n_trials"]
    df_results["pearson_resid"] = model.resid_pearson
    df_results["deviance_resid"] = model.resid_deviance
    df_results.to_csv(OUT / f"predictions_{timestamp}.csv", index=False)

    # ── Coefficients CSV ──────────────────────────────────────────────────────────
    coef_df.to_csv(OUT / f"coefficients_{timestamp}.csv")

    # ── Feature importance CSV ────────────────────────────────────────────────────
    importance = pd.DataFrame(
        {
            "feature": model.params.index,
            "coef": model.params.values,
            "abs_z": np.abs(model.tvalues.values),
            "p_value": model.pvalues.values,
            "significant": (model.pvalues.values < 0.05),
        }
    ).sort_values("abs_z", ascending=False)
    importance.to_csv(OUT / f"feature_importance_{timestamp}.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_results["p_observed"], df_results["p_predicted"], alpha=0.4, s=15)
    lims = [0, max(df_results["p_observed"].max(), df_results["p_predicted"].max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Observed probability")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Observed vs Predicted")
    fig.tight_layout()
    fig.savefig(OUT / f"obs_vs_pred_{timestamp}.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df_results["p_predicted"], df_results["pearson_resid"], alpha=0.4, s=15)
    ax.axhline(0, color="red", linestyle="--")
    ax.axhline(2, color="orange", linestyle=":", linewidth=0.8)
    ax.axhline(-2, color="orange", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Pearson residual")
    ax.set_title("Residuals vs Fitted")
    fig.tight_layout()
    fig.savefig(OUT / f"residuals_{timestamp}.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, label in zip(
        axes, ["lt", "r", "mlat"], ["LT (hrs)", "R (Re)", "MLat (°)"]
    ):
        ax.scatter(df_results[col], df_results["pearson_resid"], alpha=0.4, s=12)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel(label)
        ax.set_ylabel("Pearson residual")
        ax.set_title(f"Residuals vs {label}")
    fig.suptitle("Spatial Structure in Residuals")
    fig.tight_layout()
    fig.savefig(OUT / f"spatial_residuals_{timestamp}.png", dpi=300)
    plt.close(fig)

    coefs = model.params.drop("const")
    errors = 1.96 * model.bse.drop("const")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(coefs)), coefs.values, xerr=errors.values, alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels(coefs.index, fontsize=9)
    ax.set_xlabel("Coefficient (log-odds)")
    ax.set_title("GLM Coefficients ± 95% CI")
    fig.tight_layout()
    fig.savefig(OUT / f"coefficients_{timestamp}.png", dpi=300)
    plt.close(fig)

    # ── Final stdout (goes into HPC job log) ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Outputs → {OUT.resolve()}")
    print(f"  {summary_path.name}")
    print(f"  predictions_{timestamp}.csv")
    print(f"  coefficients_{timestamp}.csv")
    print(f"  feature_importance_{timestamp}.csv")
    print(f"  4 x .png plots")
    print(f"\nMcFadden R² : {pseudo_r2:.4f}")
    print(f"AIC         : {model.aic:.2f}")
    print(f"n cells     : {len(df_model)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    bionomial_modeling()
