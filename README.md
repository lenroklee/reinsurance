# reinsurance ds project

# Automated Actuarial Pricing & Reinsurance Analytics Pipeline

## Overview
This repository contains an end-to-end data engineering and predictive analytics pipeline designed for Property & Casualty (P&C) Reinsurance Finance. The project demonstrates how to transition from traditional actuarial pricing to an advanced machine learning architecture, focusing on portfolio profitability, tail-risk identification, and underwriting leakage.

The objective is to automate the extraction of actionable business insights from raw insurance data, simulating the infrastructure of a modern Reinsurance Finance department.

## Architecture & Workflow

The pipeline is structured into two core operational phases:

1. **Data Engineering (ETL):** Processing raw exposure and claims data into a local relational data warehouse using DuckDB.
2. **Predictive Analytics:** Benchmarking traditional Generalized Linear Models (GLMs) against non-linear XGBoost algorithms to model Frequency, Severity, and Reinsurance Tail-Risk.

```mermaid
graph TD;
    A[Raw CSV Data] --> B[DuckDB ETL Pipeline];
    B --> C[Reserving & Financial Metrics];
    C --> D[Actuarial Data Warehouse];
    D --> E[GLM Baseline];
    D --> F[XGBoost Frequency & Severity];
    F --> G[Excess of Loss Classifier];
    G --> H[Underwriting Audit & Leakage Detection];
