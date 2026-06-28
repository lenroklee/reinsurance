# src/models/actuarial_audit.py

import os
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Absolute path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reinsurance_core.duckdb")

def run_permutation_importance() -> None:
    """Executes Step 5: Permutation Importance for Risk Driver Analysis."""
    print("[INFO] Step 5: Calculating Permutation Importance (Risk Drivers)...")
    conn = duckdb.connect(DB_PATH)
    
    # Extract features and target for the evaluation model
    query = """
    SELECT 
        p.Area, p.VehPower, p.VehAge, p.DrivAge, p.BonusMalus, p.Region,
        base.ReportedClaimCount
    FROM Dim_Policies p
    JOIN policy_portfolio base ON p.IDpol = base.IDpol
    WHERE p.Exposure > 0
    """
    df = conn.execute(query).df()
    conn.close()

    # Format categoricals for XGBoost
    df['Area'] = df['Area'].astype('category')
    df['Region'] = df['Region'].astype('category')
    
    features = ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Region']
    X = df[features]
    y = df['ReportedClaimCount']

    # Train a fast evaluation model using Scikit-Learn API
    print("       Training evaluation XGBoost model...")
    model = xgb.XGBRegressor(
        objective='count:poisson', 
        n_estimators=50, 
        enable_categorical=True, 
        random_state=42
    )
    model.fit(X, y)

    print("       Running permutations (this will take a few seconds)...")
    # We use negative mean Poisson deviance as the scoring metric for actuarial accuracy
    result = permutation_importance(
        model, X, y, 
        scoring='neg_mean_poisson_deviance', 
        n_repeats=5, 
        random_state=42
    )

    # Sort and plot
    sorted_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        tick_labels=np.array(features)[sorted_idx]
    )
    plt.title("Actuarial Risk Drivers (Permutation Importance)")
    plt.xlabel("Decrease in Model Accuracy when Feature is Shuffled")
    plt.tight_layout()
    
    save_path = os.path.join(PROJECT_ROOT, "data", "risk_importance.png")
    plt.savefig(save_path)
    print(f"[SUCCESS] Risk Driver plot saved to: {save_path}\n")

def run_underwriting_audit() -> None:
    """Executes Step 6: DuckDB SQL Profiling for Underwriting Leakage."""
    print("[INFO] Step 6: Executing SQL Underwriting Audit (Leakage Clusters)...")
    conn = duckdb.connect(DB_PATH)
    
    # Aggregate toxic policies by demographic clusters
    query = """
    SELECT 
        p.Region, 
        p.VehPower, 
        ROUND(AVG(m.Technical_Loss_Ratio), 2) AS Avg_TLR,
        COUNT(*) AS Toxic_Policy_Count,
        ROUND(SUM(m.Expected_Pure_Premium), 2) AS Total_Expected_Losses
    FROM ML_Underwriting_Insights m
    JOIN Dim_Policies p ON m.IDpol = p.IDpol
    WHERE m.Technical_Loss_Ratio > 150
    GROUP BY p.Region, p.VehPower
    HAVING COUNT(*) > 50  -- Filter out statistical noise
    ORDER BY Avg_TLR DESC
    LIMIT 10;
    """
    
    toxic_clusters = conn.execute(query).df()
    conn.close()
    
    print("🚨 TOP 10 TOXIC UNDERWRITING CLUSTERS (TLR > 150%) 🚨")
    print("="*85)
    print(toxic_clusters.to_string(index=False))
    print("="*85)
    print("[SUCCESS] Strategic Underwriting Audit completed.")

def main() -> None:
    print("[INFO] Starting Phase 2 Finalization: Actuarial Audit...\n")
    run_permutation_importance()
    run_underwriting_audit()

if __name__ == "__main__":
    main()