#!/usr/bin/env python3
"""
Excel Sheet and Column Extractor

This script extracts all sheet names and their corresponding column names
from an Excel file and displays them in a structured format.

Requirements:
- pandas
- openpyxl

Usage:
    python extract_excel_info.py
"""

import pandas as pd
import os
from typing import Dict, List


def extract_excel_info(file_path: str) -> Dict[str, List[str]]:
    """
    Extract sheet names and column names from an Excel file.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        Dict[str, List[str]]: Dictionary with sheet names as keys and column names as values
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    # Dictionary to store sheet names and their columns
    excel_info = {}
    
    try:
        # Get all sheet names
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        print(f"Found {len(sheet_names)} sheets in the Excel file:")
        print("=" * 60)
        
        # Extract column names for each sheet
        for sheet_name in sheet_names:
            try:
                # Read only the first row to get column names
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
                column_names = df.columns.tolist()
                excel_info[sheet_name] = column_names
                
                print(f"\nSheet: '{sheet_name}'")
                print(f"Number of columns: {len(column_names)}")
                print("Columns:")
                for i, col in enumerate(column_names, 1):
                    print(f"  {i:2d}. {col}")
                    
            except Exception as e:
                print(f"\nError reading sheet '{sheet_name}': {str(e)}")
                excel_info[sheet_name] = []
                
    except Exception as e:
        raise Exception(f"Error processing Excel file: {str(e)}")
    
    return excel_info


def save_to_text_file(excel_info: Dict[str, List[str]], output_file: str):
    """
    Save the extracted information to a text file.
    
    Args:
        excel_info (Dict[str, List[str]]): Dictionary with sheet and column information
        output_file (str): Path to the output text file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Excel File Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        for sheet_name, columns in excel_info.items():
            f.write(f"Sheet: {sheet_name}\n")
            f.write(f"Number of columns: {len(columns)}\n")
            f.write("Columns:\n")
            
            if columns:
                for i, col in enumerate(columns, 1):
                    f.write(f"  {i:2d}. {col}\n")
            else:
                f.write("  No columns found or error reading sheet\n")
            
            f.write("\n" + "-" * 40 + "\n\n")


def main():
    """Main function to execute the script."""
    # Excel file path
    excel_file_path = "EPCL VEHS Data (Mar23 - Mar24).xlsx"
    
    # Output file for saving results
    output_file_path = "excel_analysis_report.txt"
    
    try:
        print("Extracting Excel file information...")
        print(f"File: {excel_file_path}")
        print("=" * 60)
        
        # Extract information
        excel_info = extract_excel_info(excel_file_path)
        
        # Save to text file
        save_to_text_file(excel_info, output_file_path)
        
        print("\n" + "=" * 60)
        print(f"Analysis complete!")
        print(f"Results saved to: {output_file_path}")
        print(f"Total sheets processed: {len(excel_info)}")
        
        # Summary
        total_columns = sum(len(columns) for columns in excel_info.values())
        print(f"Total columns across all sheets: {total_columns}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the Excel file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
