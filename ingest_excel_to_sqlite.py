#!/usr/bin/env python3
"""
Enhanced Excel to SQLite Ingestion Script for EPCL VEHS Data

This script converts the EPCL VEHS Excel file into a normalized SQLite database
with core columns for fast querying and extra_json for complete data retention.

Features:
- Normalized schema with core columns + extra_json pattern
- Date parsing and validation
- Proper indexing for performance
- Data quality checks and logging
- Summary tables for common queries

Requirements:
- pandas>=1.5.0
- openpyxl>=3.0.0
- sqlite3 (built-in)

Usage:
    python ingest_excel_to_sqlite.py
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
EXCEL_PATH = "EPCL VEHS Data (Mar23 - Mar24).xlsx"
SQLITE_PATH = "epcl_vehs.db"

# Core columns for Incident and Hazard ID tables (most queried fields)
CORE_INCIDENT_COLS = [
    "incident_number",
    "date_of_occurrence", 
    "incident_type",
    "section",
    "title",
    "status",
    "category",
    "description",
    "group_company",
    "location",
    "sub_location",
    "department",
    "sub_department",
    "injury_potential",
    "location_tag",
    "specific_location_of_occurrence",
    "repeated_incident",
    "incident_time",
    "person_involved",
    "contracting_company",
    "reported_by",
    "date_reported",
    "date_entered",
    "responsible_for_investigation",
    "task_or_activity_at_time_of_incident",
    "injury_classification",
    "restricted_days",
    "lost_days",
    "investigation_team_leader",
    "target_completion_date",
    "completion_date",
    "injury_illness_type",
    "body_part",
    "accident_type",
    "accident_agent",
    "job_title",
    "hire_date",
    "years_in_present_job",
    "total_years_experience",
    "ppe_worn",
    "chemicals_released",
    "quantity_released",
    "equipment_failure_type",
    "equipment",
    "equipment_id",
    "asset_damage_type",
    "pse_category",
    "relevant_consequence_incident",
    "worst_case_consequence_incident", 
    "actual_consequence_incident",
    "root_cause",
    "cost_type",
    "total_cost",
    "investigation_type",
    "corrective_actions",
    "violation_type_incident"
]

# Core columns for Audit-related tables
CORE_AUDIT_COLS = [
    "audit_number",
    "audit_location", 
    "audit_title",
    "auditor",
    "start_date",
    "audit_status",
    "location_tag",
    "audit_category",
    "auditing_body",
    "audit_rating",
    "group_company",
    "audit_type_epcl",
    "template",
    "template_version",
    "created_by",
    "audit_team",
    "supervisor",
    "responsible_for_action_plan",
    "checklist_category",
    "question",
    "regulatory_reference",
    "answer",
    "recommendation",
    "response",
    "finding",
    "finding_location",
    "worst_case_consequence",
    "action_item_number",
    "action_item_title",
    "action_item_description",
    "action_item_responsible",
    "action_item_priority",
    "action_item_due_date",
    "action_item_status"
]


def normalize_column_name(col_name: str) -> str:
    """
    Normalize column names to database-friendly format.
    
    Args:
        col_name: Original column name
        
    Returns:
        Normalized column name
    """
    if pd.isna(col_name):
        return "unnamed_column"
    
    # Convert to string and clean
    col_name = str(col_name).strip()
    
    # Remove newlines and extra spaces
    col_name = re.sub(r'\s+', ' ', col_name)
    
    # Convert to lowercase
    col_name = col_name.lower()
    
    # Replace special characters and spaces with underscores
    col_name = re.sub(r'[^\w\s]', '', col_name)
    col_name = re.sub(r'\s+', '_', col_name)
    
    # Remove duplicate underscores
    col_name = re.sub(r'_+', '_', col_name)
    
    # Remove leading/trailing underscores
    col_name = col_name.strip('_')
    
    # Handle duplicates (like .1 suffix)
    col_name = col_name.replace('_1', '_alt')
    
    return col_name


def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and standardize date columns.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with parsed dates
    """
    date_keywords = ['date', 'time', 'entered', 'completion', 'due', 'hire']
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            try:
                # Try to parse as datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Convert to ISO format string for SQLite compatibility
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                df[col] = df[col].fillna(None)
                
                logger.info(f"Parsed date column: {col}")
            except Exception as e:
                logger.warning(f"Could not parse date column {col}: {e}")
                
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize numeric columns.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with cleaned numeric data
    """
    numeric_keywords = ['cost', 'days', 'quantity', 'years', 'number', 'rating']
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in numeric_keywords):
            try:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"Cleaned numeric column: {col}")
            except Exception as e:
                logger.warning(f"Could not clean numeric column {col}: {e}")
                
    return df


def create_database_schema(conn: sqlite3.Connection) -> None:
    """
    Create the database schema with proper tables and indexes.
    
    Args:
        conn: SQLite database connection
    """
    logger.info("Creating database schema...")
    
    # Create metadata table for query assistance
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_metadata (
            table_name TEXT,
            column_name TEXT,
            column_type TEXT,
            description TEXT,
            is_core_column BOOLEAN,
            sample_values TEXT
        )
    """)
    
    # Create ingestion log table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            table_name TEXT,
            records_processed INTEGER,
            records_inserted INTEGER,
            status TEXT,
            notes TEXT
        )
    """)
    
    conn.commit()
    logger.info("Database schema created successfully")


def ingest_sheet_to_table(
    conn: sqlite3.Connection, 
    sheet_name: str, 
    df: pd.DataFrame, 
    core_cols: List[str],
    table_name: str
) -> Dict[str, Any]:
    """
    Ingest a sheet into a database table with core columns + extra_json pattern.
    
    Args:
        conn: Database connection
        sheet_name: Original sheet name
        df: DataFrame to ingest
        core_cols: List of core column names
        table_name: Target table name
        
    Returns:
        Dictionary with ingestion statistics
    """
    logger.info(f"Processing sheet '{sheet_name}' -> table '{table_name}'")
    
    if df.empty:
        logger.warning(f"Sheet '{sheet_name}' is empty, skipping...")
        return {"records_processed": 0, "records_inserted": 0, "status": "skipped"}
    
    # Normalize column names
    original_cols = df.columns.tolist()
    df.columns = [normalize_column_name(col) for col in df.columns]
    
    # Parse dates and clean numeric data
    df = parse_date_columns(df)
    df = clean_numeric_columns(df)
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    records_processed = len(df)
    
    # Create table with core columns + extra_json
    core_cols_sql = ",\n    ".join([f"{col} TEXT" for col in core_cols])
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {core_cols_sql},
            extra_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    
    conn.execute(create_sql)
    
    # Prepare data for insertion
    rows_to_insert = []
    
    for _, row in df.iterrows():
        # Extract core column values
        core_values = {}
        for col in core_cols:
            value = row.get(col)
            if pd.isna(value):
                core_values[col] = None
            else:
                core_values[col] = str(value) if value is not None else None
        
        # Pack remaining columns into extra_json
        extra_data = {}
        for col in df.columns:
            if col not in core_cols:
                value = row.get(col)
                if not pd.isna(value):
                    extra_data[col] = value
        
        # Add original column mapping to extra_json
        extra_data['_original_columns'] = dict(zip(df.columns, original_cols))
        
        # Serialize extra_json
        extra_json = json.dumps(extra_data, default=str, ensure_ascii=False)
        
        # Prepare row for insertion
        row_values = [core_values.get(col) for col in core_cols] + [extra_json]
        rows_to_insert.append(row_values)
    
    # Insert data
    placeholders = ", ".join(["?"] * (len(core_cols) + 1))
    columns = ", ".join(core_cols + ["extra_json"])
    insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    try:
        with conn:
            conn.executemany(insert_sql, rows_to_insert)
        
        records_inserted = len(rows_to_insert)
        logger.info(f"Inserted {records_inserted} records into {table_name}")
        
        # Create indexes for performance
        create_indexes(conn, table_name, core_cols)
        
        # Log metadata
        log_table_metadata(conn, table_name, core_cols, df.columns.tolist())
        
        return {
            "records_processed": records_processed,
            "records_inserted": records_inserted,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error inserting data into {table_name}: {e}")
        return {
            "records_processed": records_processed,
            "records_inserted": 0,
            "status": "error",
            "error": str(e)
        }


def create_indexes(conn: sqlite3.Connection, table_name: str, core_cols: List[str]) -> None:
    """
    Create indexes on important columns for query performance.
    
    Args:
        conn: Database connection
        table_name: Table to index
        core_cols: Core columns list
    """
    # Index commonly queried columns
    index_cols = [
        "date_of_occurrence", "incident_number", "audit_number", 
        "location", "department", "category", "status", 
        "incident_type", "audit_status", "start_date"
    ]
    
    for col in index_cols:
        if col in core_cols:
            try:
                index_name = f"idx_{table_name}_{col}"
                conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({col})")
                logger.info(f"Created index: {index_name}")
            except Exception as e:
                logger.warning(f"Could not create index on {table_name}.{col}: {e}")


def log_table_metadata(
    conn: sqlite3.Connection, 
    table_name: str, 
    core_cols: List[str], 
    all_cols: List[str]
) -> None:
    """
    Log table metadata for query assistance.
    
    Args:
        conn: Database connection
        table_name: Table name
        core_cols: Core columns
        all_cols: All columns
    """
    # Clear existing metadata for this table
    conn.execute("DELETE FROM query_metadata WHERE table_name = ?", (table_name,))
    
    # Insert core column metadata
    for col in core_cols:
        conn.execute("""
            INSERT INTO query_metadata 
            (table_name, column_name, column_type, description, is_core_column) 
            VALUES (?, ?, ?, ?, ?)
        """, (table_name, col, "TEXT", f"Core column: {col}", True))
    
    # Insert extra columns info
    extra_cols = [col for col in all_cols if col not in core_cols]
    if extra_cols:
        conn.execute("""
            INSERT INTO query_metadata 
            (table_name, column_name, column_type, description, is_core_column) 
            VALUES (?, ?, ?, ?, ?)
        """, (table_name, "extra_json", "JSON", f"Contains: {', '.join(extra_cols[:10])}{'...' if len(extra_cols) > 10 else ''}", False))


def create_summary_tables(conn: sqlite3.Connection) -> None:
    """
    Create summary tables for common analytical queries.
    
    Args:
        conn: Database connection
    """
    logger.info("Creating summary tables...")
    
    # Monthly incident summary
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incident_monthly_summary AS
        SELECT 
            substr(date_of_occurrence, 1, 7) as year_month,
            COUNT(*) as incident_count,
            COUNT(CASE WHEN status = 'Closed' THEN 1 END) as closed_count,
            COUNT(CASE WHEN injury_potential = 'High' THEN 1 END) as high_risk_count,
            SUM(CAST(total_cost AS REAL)) as total_cost_sum
        FROM incident 
        WHERE date_of_occurrence IS NOT NULL
        GROUP BY substr(date_of_occurrence, 1, 7)
        ORDER BY year_month
    """)
    
    # Incident by category summary
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incident_category_summary AS
        SELECT 
            category,
            COUNT(*) as incident_count,
            COUNT(CASE WHEN status = 'Closed' THEN 1 END) as closed_count,
            AVG(CAST(total_cost AS REAL)) as avg_cost
        FROM incident 
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY incident_count DESC
    """)
    
    # Location-based summary
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incident_location_summary AS
        SELECT 
            location,
            department,
            COUNT(*) as incident_count,
            COUNT(CASE WHEN injury_potential = 'High' THEN 1 END) as high_risk_count
        FROM incident 
        WHERE location IS NOT NULL
        GROUP BY location, department
        ORDER BY incident_count DESC
    """)
    
    conn.commit()
    logger.info("Summary tables created successfully")


def log_ingestion_result(
    conn: sqlite3.Connection, 
    table_name: str, 
    result: Dict[str, Any]
) -> None:
    """
    Log ingestion results for audit trail.
    
    Args:
        conn: Database connection
        table_name: Table name
        result: Ingestion result dictionary
    """
    conn.execute("""
        INSERT INTO ingestion_log 
        (timestamp, table_name, records_processed, records_inserted, status, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        table_name,
        result.get("records_processed", 0),
        result.get("records_inserted", 0),
        result.get("status", "unknown"),
        result.get("error", "")
    ))


def main():
    """Main ingestion function."""
    logger.info("Starting EPCL VEHS data ingestion...")
    
    # Check if Excel file exists
    if not Path(EXCEL_PATH).exists():
        logger.error(f"Excel file not found: {EXCEL_PATH}")
        return
    
    try:
        # Read Excel file
        logger.info(f"Reading Excel file: {EXCEL_PATH}")
        excel_data = pd.read_excel(EXCEL_PATH, sheet_name=None)
        logger.info(f"Found {len(excel_data)} sheets")
        
        # Create database connection
        conn = sqlite3.connect(SQLITE_PATH)
        
        # Create schema
        create_database_schema(conn)
        
        # Sheet to table mapping
        sheet_mapping = {
            "Incident": ("incident", CORE_INCIDENT_COLS),
            "Hazard ID": ("hazard_id", CORE_INCIDENT_COLS),
            "Audit": ("audit", CORE_AUDIT_COLS),
            "Audit Findings": ("audit_findings", CORE_AUDIT_COLS),
            "Inspection": ("inspection", CORE_AUDIT_COLS),
            "Inspection Findings": ("inspection_findings", CORE_AUDIT_COLS)
        }
        
        # Process each sheet
        total_records = 0
        for sheet_name, df in excel_data.items():
            if sheet_name in sheet_mapping:
                table_name, core_cols = sheet_mapping[sheet_name]
                
                result = ingest_sheet_to_table(conn, sheet_name, df, core_cols, table_name)
                log_ingestion_result(conn, table_name, result)
                
                total_records += result.get("records_inserted", 0)
                
                logger.info(f"Sheet '{sheet_name}' -> Table '{table_name}': {result}")
            else:
                logger.warning(f"Unknown sheet: {sheet_name}, skipping...")
        
        # Create summary tables
        create_summary_tables(conn)
        
        # Final commit and close
        conn.commit()
        conn.close()
        
        logger.info(f"Ingestion completed successfully!")
        logger.info(f"Total records inserted: {total_records}")
        logger.info(f"Database created: {SQLITE_PATH}")
        
        # Print summary
        print("\n" + "="*60)
        print("EPCL VEHS DATA INGESTION COMPLETE")
        print("="*60)
        print(f"Database: {SQLITE_PATH}")
        print(f"Total records: {total_records}")
        print(f"Tables created: {', '.join([mapping[0] for mapping in sheet_mapping.values()])}")
        print("Summary tables: incident_monthly_summary, incident_category_summary, incident_location_summary")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
