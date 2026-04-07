import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# didn't got synced well
# %% 1. Importing Libraries
import sys
import typing
from pathlib import Path

import gpflow
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
import tf_keras as keras
from gpflow.utilities import print_summary
from sklearn.preprocessing import StandardScaler

# %% 2. Setting up Environment

# 2. Setting project paths
project_root = Path.cwd()
sys.path.append(str(project_root))

# Define standard data subdirectories for easy access later
RAW_DATA_DIR = project_root / "data" / "raw"
PROCESSED_DATA_DIR = project_root / "data" / "processed"
ASSETS_DIR = project_root / "assets" / "3D_Objects"

# %% 3. Reading the data

processed_data_base_name = "04_prepared_data_for_classification"
class_data = pd.read_parquet(f"{PROCESSED_DATA_DIR}/{processed_data_base_name}.parquet")

# %% 4. Feature selection and engineering
features = ["x_gse", "y_gse", "z_gse"]
X_raw = class_data[features].to_numpy().astype("float64")
y_raw = class_data[["AKR_Observed"]].to_numpy().astype("float64")


# %% 5. Scaling (Critical for GP Kernels)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# %% 6. Creating the TensorFlow Dataset
# Using minibatches of 1024 allows us to process 885k points without RAM issues
batch_size = 1024
train_dataset = tf.data.Dataset.from_tensor_slices((X_scaled, y_raw))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).repeat()
train_iter = iter(train_dataset)


# %% 7. Inducing Variables
# Pick 1000 random points from your data to represent the 3D space
num_inducing = 1000
inducing_variable = X_scaled[
    np.random.choice(X_scaled.shape[0], num_inducing, replace=False),
    :,
]

# %% 8. Defining the Model
kernel = gpflow.kernels.Matern52()  # Smooth but physically realistic
likelihood = gpflow.likelihoods.Bernoulli()  # For 0/1 probability

model = gpflow.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=inducing_variable,
    num_data=X_scaled.shape[0],
)
# %% 9. Setup Checkpointing (MUST BE HERE)
optimizer = tf.optimizers.Adam(learning_rate=0.0005)
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_dir = project_root / "assets" / "model_checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

manager = tf.train.CheckpointManager(ckpt, str(checkpoint_dir), max_to_keep=1)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print(f"Restored from {manager.latest_checkpoint}")
else:
    print("Initializing from scratch.")


# %% 10. Optimization Step Function
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


# %% 11. Optimization Loop (Updated for Plotting)


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

# %% 12. Final Save
manager.save()
joblib.dump(scaler, str(checkpoint_dir / "scaler.pkl"))
print_message = f"Training Complete. Model and Scaler saved to {checkpoint_dir}"
print(print_message)

# %% 13. performance metrics
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
plot_html_path = project_root / "assets" / "elbo_plot.html"
fig.write_html(str(plot_html_path))

# Also save as a static image for reports
plot_png_path = project_root / "assets" / "elbo_plot.png"
fig.write_image(str(plot_png_path))

print(f"Interactive plot saved to: {plot_html_path}")

# This prints the optimized lengthscales and variance
print_summary(model)

# %% To load later
# import joblib

# # Load the "key" to your coordinate system
# scaler = joblib.load("./assets/model_checkpoints/scaler.pkl")

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
