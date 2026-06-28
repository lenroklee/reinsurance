import os
import duckdb

# Force absolute path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reinsurance_core.duckdb")

def calculate_financial_layers() -> None:
    """
    Enriches the database with professional P&C financial metrics, combining
    earned premiums, underwriting expenses, ultimate losses, and reinsurance structures.
    """
    print("[INFO] Connecting to DuckDB for financial metrics calculation...")
    conn = duckdb.connect(DB_PATH)
    
    try:
        conn.execute("DROP TABLE IF EXISTS financial_reporting_mart;")
        
        # Advanced SQL joining the portfolio data with our simulated reserving layer
        query = """
        CREATE TABLE financial_reporting_mart AS
        WITH premium_baseline AS (
            SELECT 
                p.IDpol,
                p.Exposure,
                p.Region,
                p.TotalClaimAmount AS ReportedLosses,
                COALESCE(c.PaidAmount, 0.0) AS PaidLosses,
                COALESCE(c.CaseReserve, 0.0) AS CaseReserves,
                COALESCE(c.IBNRReserve, 0.0) AS IBNRReserves,
                -- Ultimate Losses = Reported + IBNR
                p.TotalClaimAmount + COALESCE(c.IBNRReserve, 0.0) AS UltimateLosses,
                -- Commercial tariff formulation based on risk variables
                (250.0 + (p.VehPower * 15.0) - (CASE WHEN p.DrivAge > 25 THEN 50 ELSE 0 END) + (p.BonusMalus * 2.5)) * p.Exposure AS EarnedPremium,
                -- Operational expenses (OPEX) allocation
                (30.0 + (p.TotalClaimAmount * 0.05)) * p.Exposure AS UnderwritingExpenses
            FROM policy_portfolio p
            LEFT JOIN fact_claims c ON p.IDpol = c.IDpol
        )
        SELECT 
            IDpol,
            Exposure,
            Region,
            EarnedPremium,
            PaidLosses,
            CaseReserves,
            IBNRReserves,
            UltimateLosses,
            UnderwritingExpenses,
            -- Financial KPI Formulations (Using NULLIF to guarantee safety against zero division)
            ROUND((UltimateLosses / NULLIF(EarnedPremium, 0)) * 100, 2) AS LossRatio,
            ROUND((UnderwritingExpenses / NULLIF(EarnedPremium, 0)) * 100, 2) AS ExpenseRatio,
            ROUND(((UltimateLosses + UnderwritingExpenses) / NULLIF(EarnedPremium, 0)) * 100, 2) AS CombinedRatio,
            -- Proportional Reinsurance Layer: 50% Quota Share Treaty
            ROUND(EarnedPremium * 0.5, 2) AS ReinsurerCededPremium,
            ROUND(UltimateLosses * 0.5, 2) AS ReinsurerCededLosses,
            ROUND(UnderwritingExpenses * 0.15, 2) AS ReinsuranceCedingCommission
        FROM premium_baseline;
        """
        
        print("[INFO] Executing financial metrics database build...")
        conn.execute(query)
        
        # Aggregate validation for high-level visibility
        metrics = conn.execute("""
            SELECT 
                ROUND((SUM(UltimateLosses) / SUM(EarnedPremium)) * 100, 2),
                ROUND((SUM(UnderwritingExpenses) / SUM(EarnedPremium)) * 100, 2),
                ROUND((SUM(UltimateLosses + UnderwritingExpenses) / SUM(EarnedPremium)) * 100, 2)
            FROM financial_reporting_mart
        """).fetchone()
        
        print(f"[SUCCESS] Financial Reporting Mart successfully created.")
        print(f"          - Portfolio Loss Ratio (incl. IBNR): {metrics[0]}%")
        print(f"          - Portfolio Expense Ratio: {metrics[1]}%")
        print(f"          - Portfolio Combined Ratio: {metrics[2]}%")
        
    except Exception as e:
        print(f"[ERROR] Financial metrics pipeline failed: {e}")
    finally:
        conn.close()
        print("[INFO] Database connection closed.")

def main() -> None:
    print("[INFO] Starting Financial Data Mart Generation...")
    calculate_financial_layers()

if __name__ == "__main__":
    main()