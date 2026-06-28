import os
import duckdb

# Absolute path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reinsurance_core.duckdb")

def execute_final_modeling() -> None:
    """
    Finalizes the Data Warehouse by renaming tables according to roadmap standards
    (Dim_Policies, Fact_Financials), simulates underwriting dates (up to mid-2026), 
    and creates advanced SQL Views for Time-Series Analytics (Delta Loss Ratio).
    """
    print("[INFO] Connecting to DuckDB for final Data Modeling...")
    conn = duckdb.connect(DB_PATH)
    
    try:
        # Clean up any previous executions to allow safe re-runs
        conn.execute("DROP VIEW IF EXISTS vw_monthly_regional_summary;")
        conn.execute("DROP TABLE IF EXISTS Dim_Policies;")
        conn.execute("DROP TABLE IF EXISTS Fact_Financials;")
        
        # 1. Create Dim_Policies (The Demographic Dimension)
        # Here we add Claim Frequency and simulate dates up to June 2026
        print("[INFO] Creating Dim_Policies table (with 2024-2026 time simulation)...")
        conn.execute("""
        CREATE TABLE Dim_Policies AS
        SELECT 
            IDpol,
            Exposure,
            Region,
            Area,
            VehPower,
            VehAge,
            DrivAge,
            BonusMalus,
            -- Metric: Claim Frequency (Claims / Exposure)
            ROUND(ReportedClaimCount / NULLIF(Exposure, 0.0), 4) AS Claim_Frequency,
            -- Stochastic Generation of Underwriting Date (Between Jan 2024 and Jun 2026)
            DATE '2024-01-01' + CAST(FLOOR(RANDOM() * 880) AS INTEGER) AS Underwriting_Date,
            -- Month extraction for cohorts (YYYY-MM)
            strftime(DATE '2024-01-01' + CAST(FLOOR(RANDOM() * 880) AS INTEGER), '%Y-%m') AS Underwriting_Month
        FROM policy_portfolio;
        """)

        # 2. Create Fact_Financials (The Financial Fact Table)
        print("[INFO] Creating Fact_Financials table...")
        conn.execute("""
        CREATE TABLE Fact_Financials AS
        SELECT * FROM financial_reporting_mart;
        """)

        # 3. Create Analytical SQL View (Temporal and delta feature engineering)
        # We use Window Functions (LAG) to calculate the difference with the previous month
        print("[INFO] Creating SQL View: vw_monthly_regional_summary (with Delta Loss Ratio)...")
        conn.execute("""
        CREATE VIEW vw_monthly_regional_summary AS
        WITH monthly_agg AS (
            SELECT 
                p.Underwriting_Month,
                p.Region,
                SUM(p.Exposure) AS Total_Exposure,
                SUM(f.EarnedPremium) AS Total_EarnedPremium,
                SUM(f.UltimateLosses) AS Total_UltimateLosses,
                -- Calculation of the regional monthly Loss Ratio
                ROUND((SUM(f.UltimateLosses) / NULLIF(SUM(f.EarnedPremium), 0)) * 100, 2) AS Monthly_LR
            FROM Dim_Policies p
            JOIN Fact_Financials f ON p.IDpol = f.IDpol
            GROUP BY p.Underwriting_Month, p.Region
        )
        SELECT 
            Underwriting_Month,
            Region,
            Total_EarnedPremium,
            Total_UltimateLosses,
            Monthly_LR,
            -- Retrieve the previous month's LR for the same region
            LAG(Monthly_LR) OVER(PARTITION BY Region ORDER BY Underwriting_Month) AS Prev_Month_LR,
            -- Calculate the Delta LR
            ROUND(Monthly_LR - LAG(Monthly_LR) OVER(PARTITION BY Region ORDER BY Underwriting_Month), 2) AS Delta_LR
        FROM monthly_agg;
        """)
        
        # View validation
        sample_view = conn.execute("""
            SELECT Region, Underwriting_Month, Monthly_LR, Delta_LR 
            FROM vw_monthly_regional_summary 
            WHERE Region = 'R82' AND Underwriting_Month >= '2026-01'
            ORDER BY Underwriting_Month LIMIT 3
        """).fetchall()
        
        print("[SUCCESS] Data Modeling successfully completed.")
        print("          Sample data from the View (Region R82 in 2026):")
        for row in sample_view:
            print(f"          - Month: {row[1]} | LR: {row[2]}% | Delta vs Prev Month: {row[3]}%")

    except Exception as e:
        print(f"[ERROR] Data modeling failed: {e}")
    finally:
        # 4. Export to Parquet for Power BI Ingestion
        print("[INFO] Exporting Star Schema to Parquet files...")
        
        # Define the export directory
        parquet_dir = os.path.join(PROJECT_ROOT, "data", "processed", "powerbi_export")
        os.makedirs(parquet_dir, exist_ok=True)

        # Export Tables and Views using ZSTD compression
        conn.execute(f"COPY Dim_Policies TO '{parquet_dir}/dim_policies.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);")
        conn.execute(f"COPY Fact_Financials TO '{parquet_dir}/fact_financials.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);")
        conn.execute(f"COPY (SELECT * FROM vw_monthly_regional_summary) TO '{parquet_dir}/vw_monthly_summary.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);")
        
        print(f"[SUCCESS] Parquet files exported to: {parquet_dir}")
        conn.close()
        print("[INFO] Database connection closed.")

def main() -> None:
    print("[INFO] Starting Step 3 & 4: Final Data Modeling...")
    execute_final_modeling()

if __name__ == "__main__":
    main()