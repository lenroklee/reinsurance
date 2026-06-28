import os
import duckdb

# Force absolute path resolution relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Clean directory and file mapping using absolute paths
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DB_PATH = os.path.join(PROCESSED_DIR, "reinsurance_core.duckdb")

FREQ_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "freMTPL2freq.csv")
SEV_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "freMTPL2sev.csv")

def setup_environment() -> None:
    """Ensures directories exist and removes old DB files to allow clean re-runs."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if os.path.exists(DB_PATH):
        print(f"[INFO] Existing database found at {DB_PATH}. Overwriting...")
        os.remove(DB_PATH)

def build_data_warehouse() -> None:
    """
    Executes the ETL: reads raw CSVs, aggregates claims per policy,
    performs a LEFT JOIN, and saves the results into the local DuckDB database.
    Configured to handle malformed rows in production datasets.
    """
    print("[INFO] Connecting to DuckDB...")
    conn = duckdb.connect(DB_PATH)
    
    try:
        # Added 'ignore_errors=true' and 'null_padding=true' to handle malformed rows cleanly
        query = f"""
        CREATE TABLE policy_portfolio AS
        WITH aggregated_claims AS (
            -- Step A: Sum claim amounts for policies with multiple accidents
            SELECT 
                IDpol,
                SUM(ClaimAmount) AS TotalClaimAmount,
                COUNT(ClaimAmount) AS ActualClaimCount
            FROM read_csv_auto('{SEV_FILE}', ignore_errors=true, null_padding=true)
            GROUP BY IDpol
        )
        -- Step B: Join the policy register with the aggregated claims
        SELECT 
            f.IDpol,
            f.Exposure,
            f.Area,
            f.VehPower,
            f.VehAge,
            f.DrivAge,
            f.BonusMalus,
            f.VehBrand,
            f.VehGas,
            f.Density,
            f.Region,
            f.ClaimNb AS ReportedClaimCount,
            -- NULL Handling: If there is no claim, the cost is 0.0
            COALESCE(c.TotalClaimAmount, 0.0) AS TotalClaimAmount,
            COALESCE(c.ActualClaimCount, 0) AS PaidClaimCount
        FROM read_csv_auto('{FREQ_FILE}', ignore_errors=true, null_padding=true) f
        LEFT JOIN aggregated_claims c ON f.IDpol = c.IDpol;
        """
        
        print("[INFO] Executing SQL transformation with error tolerance...")
        conn.execute(query)
        
        # Data validation
        total_rows = conn.execute("SELECT COUNT(*) FROM policy_portfolio").fetchone()[0]
        zero_claims = conn.execute("SELECT COUNT(*) FROM policy_portfolio WHERE TotalClaimAmount = 0").fetchone()[0]
        
        print("[SUCCESS] Table 'policy_portfolio' successfully created.")
        print(f"          - Total policies processed: {total_rows:,}")
        print(f"          - Policies with zero claims: {zero_claims:,}")
        
    except Exception as e:
        print(f"[ERROR] Database execution failed: {e}")
    finally:
        conn.close()
        print(f"[INFO] Connection closed. Database saved at {DB_PATH}")

def main() -> None:
    print("[INFO] Starting Data Transformation Pipeline...")
    setup_environment()
    build_data_warehouse()
    print("[INFO] Pipeline completed.")

if __name__ == "__main__":
    main()