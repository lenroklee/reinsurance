import os
import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

# Absolute path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reinsurance_core.duckdb")

def load_and_prepare_data() -> pd.DataFrame:
    """Extracts data from DuckDB and prepares variables for Machine Learning."""
    print("[INFO] Extracting analytical dataset from DuckDB...")
    conn = duckdb.connect(DB_PATH)
    
    query = """
    SELECT 
        p.IDpol, p.Exposure, p.Area, p.VehPower, p.VehAge, 
        p.DrivAge, p.BonusMalus, p.Region, 
        base.ReportedClaimCount,
        f.UltimateLosses,
        f.EarnedPremium
    FROM Dim_Policies p
    JOIN Fact_Financials f ON p.IDpol = f.IDpol
    JOIN policy_portfolio base ON p.IDpol = base.IDpol
    WHERE p.Exposure > 0
    """
    df = conn.execute(query).df()
    conn.close()
    
    # 2026 Standard: Native categorical variable handling for XGBoost
    df['Area'] = df['Area'].astype('category')
    df['Region'] = df['Region'].astype('category')
    
    return df

def train_actuarial_xgboost() -> None:
    """Trains the XGBoost pipeline for Frequency, Severity, and Tail Risk."""
    df = load_and_prepare_data()
    print("[INFO] Dataset loaded. Portfolio size:", len(df))
    
    # Defining input features
    feature_cols = ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Region']
    X = df[feature_cols]
    
    # -------------------------------------------------------------------
    # 1. FREQUENCY MODEL (POISSON)
    # -------------------------------------------------------------------
    print("\n[INFO] Training XGBoost Frequency Model (Poisson Objective)...")
    y_freq = df['ReportedClaimCount']
    
    # DMatrix configuration with base_margin (Actuarial Offset)
    dtrain_freq = xgb.DMatrix(X, label=y_freq, enable_categorical=True)
    dtrain_freq.set_base_margin(np.log(df['Exposure']))
    
    freq_params = {
        'objective': 'count:poisson',
        'eval_metric': 'poisson-nloglik',
        'max_depth': 5,
        'learning_rate': 0.05,
        'seed': 42
    }
    
    # Sequential tree model training
    freq_model = xgb.train(freq_params, dtrain_freq, num_boost_round=150)
    
    # Annualized frequency prediction
    dprod_freq = xgb.DMatrix(X, enable_categorical=True)
    dprod_freq.set_base_margin(np.log(df['Exposure']))
    df['Pred_Frequency_Count'] = freq_model.predict(dprod_freq)

    # -------------------------------------------------------------------
    # 2. SEVERITY MODEL (GAMMA)
    # -------------------------------------------------------------------
    print("\n[INFO] Training XGBoost Severity Model (Gamma Objective)...")
    # We will train the model ONLY on those who actually registered at least one claim
    # AND generated a positive financial loss (Gamma requires strictly > 0)
    df_claims = df[(df['ReportedClaimCount'] > 0) & (df['UltimateLosses'] > 0)].copy()
    
    X_claims = df_claims[feature_cols]
    y_sev = df_claims['UltimateLosses'] / df_claims['ReportedClaimCount']
    
    dtrain_sev = xgb.DMatrix(X_claims, label=y_sev, enable_categorical=True)
    
    sev_params = {
        'objective': 'reg:gamma',
        'eval_metric': 'gamma-nloglik',
        'max_depth': 5,
        'learning_rate': 0.05,
        'seed': 42
    }
    
    sev_model = xgb.train(sev_params, dtrain_sev, num_boost_round=120)
    
    # Average severity prediction across the entire portfolio
    df['Pred_Severity_Cost'] = sev_model.predict(xgb.DMatrix(X, enable_categorical=True))

    # -------------------------------------------------------------------
    # 3. PURE PREMIUM AND EXPECTED LOSS RATIO
    # -------------------------------------------------------------------
    # Calculation of the final expected loss (Pure Premium)
    df['Expected_Pure_Premium'] = df['Pred_Frequency_Count'] * df['Pred_Severity_Cost']
    
    # Calculation of the Technical Loss Ratio (Underwriting Leakage Detection)
    df['Technical_Loss_Ratio'] = (df['Expected_Pure_Premium'] / df['EarnedPremium']) * 100

    # -------------------------------------------------------------------
    # 4. REINSURANCE LAYER: TAIL RISK CLASSIFIER (XoL)
    # -------------------------------------------------------------------
    # Identify the 95th percentile threshold for severe claims
    large_loss_threshold = y_sev.quantile(0.95)
    print(f"\n[INFO] Configuring Excess of Loss (XoL) Reinsurance Layer...")
    print(f"       - Actuarial Retention Threshold (95th Percentile): EUR {large_loss_threshold:,.2f}")
    
    # The target is 1 if the policy generated a claim exceeding the retention, otherwise 0
    df['Is_Large_Loss'] = np.where((df['ReportedClaimCount'] > 0) & 
                                   ((df['UltimateLosses'] / df['ReportedClaimCount']) > large_loss_threshold), 1, 0)
    
    y_xol = df['Is_Large_Loss']
    
    # Calculate class balance to handle the severe imbalance of large claims
    ratio = (len(y_xol) - sum(y_xol)) / sum(y_xol)
    
    dtrain_xol = xgb.DMatrix(X, label=y_xol, enable_categorical=True)
    xol_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': ratio, # Stochastic balancing of tail risk
        'max_depth': 6,
        'learning_rate': 0.05,
        'seed': 42
    }
    
    xol_model = xgb.train(xol_params, dtrain_xol, num_boost_round=100)
    df['XoL_Trigger_Probability'] = xol_model.predict(xgb.DMatrix(X, enable_categorical=True))

    # -------------------------------------------------------------------
    # 5. WRITING INSIGHTS TO DATA WAREHOUSE (DUCKDB)
    # -------------------------------------------------------------------
    print("\n[INFO] Saving predictive metrics to DuckDB...")
    conn = duckdb.connect(DB_PATH)
    
    conn.execute("DROP TABLE IF EXISTS ML_Underwriting_Insights;")
    
    # Select only key columns to avoid duplicating demographics
    insights_df = df[[
        'IDpol', 'Pred_Frequency_Count', 'Pred_Severity_Cost', 
        'Expected_Pure_Premium', 'Technical_Loss_Ratio', 
        'XoL_Trigger_Probability', 'Is_Large_Loss'
    ]]
    
    conn.execute("CREATE TABLE ML_Underwriting_Insights AS SELECT * FROM insights_df")
    
    # Final aggregate validation
    avg_tech_lr = conn.execute("SELECT AVG(Technical_Loss_Ratio) FROM ML_Underwriting_Insights").fetchone()[0]
    high_risk_count = conn.execute("SELECT COUNT(*) FROM ML_Underwriting_Insights WHERE Technical_Loss_Ratio > 150").fetchone()[0]
    
    print("[SUCCESS] Predictive Analytics pipeline completed.")
    print(f"          - Average Technical Loss Ratio of the Portfolio: {avg_tech_lr:.2f}%")
    print(f"          - High-Risk Policies Detected (Underwriting Leakage > 150%): {high_risk_count:,}")
    
    conn.close()

def main() -> None:
    print("[INFO] Starting Phase 2: Advanced Actuarial Machine Learning...")
    train_actuarial_xgboost()

if __name__ == "__main__":
    main()