import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# 1. Setup Paths
project_root = Path.cwd()
PROCESSED_DATA_DIR = project_root / "data" / "processed"
RESULTS_DIR = project_root / "assets" / "tree_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load Data
print("Loading data for Tree Inference...")
processed_base = "04_prepared_data_for_classification"
df = pd.read_parquet(PROCESSED_DATA_DIR / f"{processed_base}.parquet")

features = ["x_gse", "y_gse", "z_gse"]
X_raw = df[features].values.astype("float64")
y_raw = df["AKR_Observed"].values.astype("int") # Tree likes int for classification

# 3. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 4. Train the "Pattern" Tree
# We use max_depth=5 so the image is actually readable
print("Fitting Decision Tree...")
tree_model = DecisionTreeClassifier(
            max_depth=5, 
                class_weight="balanced", # Good if you have more 'No AKR' than 'AKR'
                    random_state=42
                    )
tree_model.fit(X_scaled, y_raw)

# 5. Export Feature Importance to Log
print("\n--- Feature Importance ---")
for name, importance in zip(features, tree_model.feature_importances_):
        print(f"{name}: {importance:.4f}")

        # 6. Save the Tree Visualization (HPC Safe)
        print("\nGenerating Tree Diagram...")
        plt.figure(figsize=(25, 12))
        plot_tree(
                    tree_model, 
                        feature_names=features, 
                            class_names=['No AKR', 'AKR'], 
                                filled=True, 
                                    rounded=True, 
                                        fontsize=10
                                        )

        # Save as PNG instead of showing
        plot_path = RESULTS_DIR / "decision_tree_logic.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close() # Frees up memory on the HPC
        print(f"Tree logic diagram saved to: {plot_path}")

        # 7. Pattern Inference (Summary of the 3D Probability)
        # Let's see what the tree thinks of a simple Z-axis slice at X=0, Y=0
        print("\nRunning quick Z-axis Inference check...")
        z_check = np.linspace(-15, 15, 31)
        test_coords = np.zeros((len(z_check), 3))
        test_coords[:, 2] = z_check # Set the Z values
        test_coords_scaled = scaler.transform(test_coords)

        probs = tree_model.predict_proba(test_coords_scaled)[:, 1]

        # Print a small table of the result
        print("Z_GSE | Prob(AKR)")
        for z, p in zip(z_check, probs):
                print(f"{z:5.1f} | {p:.4f}")

                print("\nAll tasks complete.")
