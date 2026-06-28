import os
import duckdb
import pandas as pd
import numpy as np

# Force absolute path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reinsurance_core.duckdb")

def generate_reserving_layer() -> None:
    """
    Simulates the actuarial claims development process (Reporting Delays and Payment Delays)
    to calculate IBNR (Incurred But Not Reported) and Case Reserves.
    """
    print("[INFO] Connecting to DuckDB to generate Claims Reserving Layer...")
    conn = duckdb.connect(DB_PATH)
    
    try:
        print("[INFO] Extracting claim events for simulation...")
        claims_df = conn.execute("""
            SELECT IDpol, TotalClaimAmount 
            FROM policy_portfolio 
            WHERE TotalClaimAmount > 0
        """).df()
        
        np.random.seed(42)  # For reproducibility
        num_claims = len(claims_df)
        
        # Stochastic simulation of reporting and payment delays (in days)
        reporting_delay = np.random.lognormal(mean=1.5, sigma=1.2, size=num_claims)
        payment_delay = np.random.lognormal(mean=3.0, sigma=1.5, size=num_claims)
        
        claims_df['ReportingDelayDays'] = np.round(reporting_delay)
        claims_df['PaymentDelayDays'] = np.round(payment_delay)
        claims_df['TotalDelayDays'] = claims_df['ReportingDelayDays'] + claims_df['PaymentDelayDays']
        
        # Financial Accounting Logic (Valuation Date set at day 90)
        valuation_day = 90
        
        claims_df['Status'] = np.where(
            claims_df['ReportingDelayDays'] > valuation_day, 'IBNR',
            np.where(claims_df['TotalDelayDays'] > valuation_day, 'Reported_Open', 'Paid_Closed')
        )
        
        claims_df['PaidAmount'] = np.where(claims_df['Status'] == 'Paid_Closed', claims_df['TotalClaimAmount'], 0.0)
        claims_df['CaseReserve'] = np.where(claims_df['Status'] == 'Reported_Open', claims_df['TotalClaimAmount'], 0.0)
        claims_df['IBNRReserve'] = np.where(claims_df['Status'] == 'IBNR', claims_df['TotalClaimAmount'], 0.0)
        claims_df['UltimateLoss'] = claims_df['TotalClaimAmount']
        
        print("[INFO] Writing fact_claims table into DuckDB...")
        conn.execute("DROP TABLE IF EXISTS fact_claims;")
        conn.execute("CREATE TABLE fact_claims AS SELECT * FROM claims_df")
        
        ibnr_total = conn.execute("SELECT SUM(IBNRReserve) FROM fact_claims").fetchone()[0]
        case_total = conn.execute("SELECT SUM(CaseReserve) FROM fact_claims").fetchone()[0]
        paid_total = conn.execute("SELECT SUM(PaidAmount) FROM fact_claims").fetchone()[0]
        
        print("[SUCCESS] Claims Reserving Layer created (fact_claims).")
        print(f"          - Total Paid Claims: EUR {paid_total:,.2f}")
        print(f"          - Total Case Reserves (Open): EUR {case_total:,.2f}")
        print(f"          - Total IBNR Reserves (Blind): EUR {ibnr_total:,.2f}")
        
    except Exception as e:
        print(f"[ERROR] Claims Reserving generator failed: {e}")
    finally:
        conn.close()
        print("[INFO] Database connection closed.")

def main() -> None:
    print("[INFO] Starting Flusso 2: Claims Reserving & IBNR Generator...")
    generate_reserving_layer()

if __name__ == "__main__":
    main()