import os
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Absolute path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reinsurance_core.duckdb")

def load_modeling_data() -> pd.DataFrame:
    """Extracts the joined feature space from DuckDB for quantitative modeling."""
    print("[INFO] Extracting analytical dataset from DuckDB...")
    conn = duckdb.connect(DB_PATH)
    
    # We join Dimensions (features), Facts (targets), and the base portfolio (raw counts)
    query = """
    SELECT 
        p.IDpol, p.Exposure, p.Area, p.VehPower, p.VehAge, 
        p.DrivAge, p.BonusMalus, p.Region, 
        base.ReportedClaimCount,
        f.UltimateLosses
    FROM Dim_Policies p
    JOIN Fact_Financials f ON p.IDpol = f.IDpol
    JOIN policy_portfolio base ON p.IDpol = base.IDpol
    WHERE p.Exposure > 0  -- Filter out zero-exposure policies to prevent log(0) errors
    """
    df = conn.execute(query).df()
    conn.close()
    return df

def train_glm_baseline(df: pd.DataFrame) -> None:
    """
    Trains traditional Actuarial Generalized Linear Models (GLMs).
    - Model 1: Poisson GLM for Claim Frequency
    - Model 2: Gamma GLM for Claim Severity
    """
    # -------------------------------------------------------------------
    # 1. ACTUARIAL DATA TRANSFORMATIONS
    # -------------------------------------------------------------------
    print("[INFO] Applying actuarial transformations (Offsets and Targets)...")
    
    # Frequency Target & Offset
    df['Log_Exposure'] = np.log(df['Exposure'])
    
    # Severity Target (Only calculated for policies that actually had a claim)
    # Average Severity = Total Cost / Number of Claims
    df_claims = df[df['ReportedClaimCount'] > 0].copy()
    df_claims['Avg_Severity'] = df_claims['UltimateLosses'] / df_claims['ReportedClaimCount']
    
    # Cap extreme severity outliers to stabilize the Gamma GLM (99th percentile)
    cap_threshold = df_claims['Avg_Severity'].quantile(0.99)
    df_claims['Avg_Severity'] = np.minimum(df_claims['Avg_Severity'], cap_threshold)

    # -------------------------------------------------------------------
    # 2. TRAIN FREQUENCY GLM (POISSON)
    # -------------------------------------------------------------------
    print("\n[INFO] Training Frequency GLM (Poisson)...")
    # We use a formula string, similar to R. 'C()' treats variables as Categorical.
    freq_formula = "ReportedClaimCount ~ VehPower + VehAge + DrivAge + BonusMalus + C(Area)"
    
    # Family=Poisson, Link=Log (default for Poisson)
    freq_model = smf.glm(
        formula=freq_formula, 
        data=df, 
        offset=df['Log_Exposure'], 
        family=sm.families.Poisson()
    ).fit()
    
    print("--- FREQUENCY MODEL SUMMARY (Top Variables) ---")
    # We print just a slice of the summary to keep the terminal clean
    print(freq_model.summary().tables[1])

    # -------------------------------------------------------------------
    # 3. TRAIN SEVERITY GLM (GAMMA)
    # -------------------------------------------------------------------
    print("\n[INFO] Training Severity GLM (Gamma)...")
    sev_formula = "Avg_Severity ~ VehPower + VehAge + DrivAge + BonusMalus + C(Area)"
    
    # Family=Gamma, Link=Log (forces multiplicative factors)
    sev_model = smf.glm(
        formula=sev_formula, 
        data=df_claims, 
        family=sm.families.Gamma(link=sm.families.links.log())
    ).fit()
    
    print("--- SEVERITY MODEL SUMMARY (Top Variables) ---")
    print(sev_model.summary().tables[1])
    
    print("\n[SUCCESS] Actuarial GLM Baselines successfully trained.")

def main() -> None:
    print("[INFO] Starting Phase 2: Actuarial Baseline Modeling...")
    df = load_modeling_data()
    train_glm_baseline(df)

if __name__ == "__main__":
    main()