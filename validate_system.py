#!/usr/bin/env python3
"""
EPCL VEHS System Validation Script

This script performs comprehensive validation of the entire EPCL VEHS SQL Agent system
to ensure everything is working correctly before deployment or after updates.

Validation Categories:
- File structure and dependencies
- Database integrity and schema
- SQL security and validation
- API endpoints and authentication
- LangChain agent functionality
- Performance benchmarks
- Security compliance

Usage:
    python validate_system.py [--quick] [--full] [--security] [--performance]
"""

import os
import sys
import json
import time
import sqlite3
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(message: str):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def print_header(message: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")


class SystemValidator:
    """Comprehensive system validator for EPCL VEHS SQL Agent."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "start_time": datetime.now(),
            "categories": {}
        }
    
    def validate_file_structure(self) -> Dict[str, Any]:
        """Validate project file structure and dependencies."""
        print_header("File Structure & Dependencies")
        
        category_results = {"passed": 0, "failed": 0, "warnings": 0}
        
        # Required files
        required_files = [
            "requirements.txt",
            "main.py",
            "safe_sql_executor.py",
            "langchain_sql_agent.py",
            "ingest_excel_to_sqlite.py",
            "run_system.py",
            "test_suite.py",
            "README.md",
            "Dockerfile",
            "docker-compose.yml",
            ".env.example"
        ]
        
        print_info("Checking required files...")
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print_success(f"Found: {file_path}")
                category_results["passed"] += 1
            else:
                print_error(f"Missing: {file_path}")
                category_results["failed"] += 1
        
        # Check Excel data file
        excel_file = self.project_root / "EPCL VEHS Data (Mar23 - Mar24).xlsx"
        if excel_file.exists():
            size_mb = excel_file.stat().st_size / (1024 * 1024)
            print_success(f"Excel data file found ({size_mb:.1f} MB)")
            category_results["passed"] += 1
        else:
            print_warning("Excel data file not found - required for data ingestion")
            category_results["warnings"] += 1
        
        # Check Python dependencies
        print_info("Validating Python dependencies...")
        try:
            import pandas
            import openpyxl
            import fastapi
            import langchain
            import openai
            print_success("Core dependencies available")
            category_results["passed"] += 1
        except ImportError as e:
            print_error(f"Missing dependency: {e}")
            category_results["failed"] += 1
        
        self.results["categories"]["file_structure"] = category_results
        return category_results
    
    def validate_database(self) -> Dict[str, Any]:
        """Validate database structure and integrity."""
        print_header("Database Validation")
        
        category_results = {"passed": 0, "failed": 0, "warnings": 0}
        
        db_path = self.project_root / "epcl_vehs.db"
        
        if not db_path.exists():
            print_warning("Database not found - run ingestion first")
            category_results["warnings"] += 1
            self.results["categories"]["database"] = category_results
            return category_results
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check required tables
            expected_tables = [
                "incident", "hazard_id", "audit", "audit_findings", 
                "inspection", "inspection_findings", "query_metadata"
            ]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            print_info("Checking database tables...")
            for table in expected_tables:
                if table in existing_tables:
                    # Check table structure
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print_success(f"Table '{table}' exists with {count} records")
                    category_results["passed"] += 1
                else:
                    print_error(f"Missing table: {table}")
                    category_results["failed"] += 1
            
            # Check indexes
            print_info("Checking database indexes...")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            
            if len(indexes) > 0:
                print_success(f"Found {len(indexes)} database indexes")
                category_results["passed"] += 1
            else:
                print_warning("No database indexes found")
                category_results["warnings"] += 1
            
            # Check data quality
            if "incident" in existing_tables:
                cursor.execute("SELECT COUNT(*) FROM incident WHERE date_of_occurrence IS NOT NULL")
                valid_dates = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM incident")
                total_records = cursor.fetchone()[0]
                
                if total_records > 0:
                    date_quality = (valid_dates / total_records) * 100
                    if date_quality > 80:
                        print_success(f"Date quality: {date_quality:.1f}% valid dates")
                        category_results["passed"] += 1
                    else:
                        print_warning(f"Date quality: {date_quality:.1f}% valid dates")
                        category_results["warnings"] += 1
            
            conn.close()
            
        except Exception as e:
            print_error(f"Database validation failed: {e}")
            category_results["failed"] += 1
        
        self.results["categories"]["database"] = category_results
        return category_results
    
    def validate_sql_security(self) -> Dict[str, Any]:
        """Validate SQL security and injection prevention."""
        print_header("SQL Security Validation")
        
        category_results = {"passed": 0, "failed": 0, "warnings": 0}
        
        try:
            from safe_sql_executor import SafeSQLExecutor, SQLValidator
            
            # Test SQL validator
            validator = SQLValidator()
            
            # Test valid queries
            valid_queries = [
                "SELECT COUNT(*) FROM incident",
                "SELECT category, COUNT(*) FROM incident GROUP BY category LIMIT 10"
            ]
            
            print_info("Testing valid SQL queries...")
            for query in valid_queries:
                result = validator.validate_query(query)
                if result.is_valid:
                    print_success(f"Valid query accepted: {query[:50]}...")
                    category_results["passed"] += 1
                else:
                    print_error(f"Valid query rejected: {query[:50]}...")
                    category_results["failed"] += 1
            
            # Test malicious queries
            malicious_queries = [
                "DROP TABLE incident",
                "DELETE FROM incident WHERE id = 1",
                "SELECT * FROM incident; DROP TABLE incident; --",
                "INSERT INTO incident VALUES (999, 'hacked')"
            ]
            
            print_info("Testing malicious SQL queries...")
            for query in malicious_queries:
                result = validator.validate_query(query)
                if not result.is_valid:
                    print_success(f"Malicious query blocked: {query[:50]}...")
                    category_results["passed"] += 1
                else:
                    print_error(f"Malicious query allowed: {query[:50]}...")
                    category_results["failed"] += 1
            
        except Exception as e:
            print_error(f"SQL security validation failed: {e}")
            category_results["failed"] += 1
        
        self.results["categories"]["sql_security"] = category_results
        return category_results
    
    def validate_api_endpoints(self) -> Dict[str, Any]:
        """Validate API endpoints and authentication."""
        print_header("API Endpoints Validation")
        
        category_results = {"passed": 0, "failed": 0, "warnings": 0}
        
        # Check if server is running
        base_url = "http://127.0.0.1:8000"
        
        try:
            # Test health endpoint
            print_info("Testing health endpoint...")
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code == 200:
                print_success("Health endpoint responding")
                category_results["passed"] += 1
            else:
                print_error(f"Health endpoint failed: {response.status_code}")
                category_results["failed"] += 1
            
            # Test metrics endpoint
            print_info("Testing metrics endpoint...")
            response = requests.get(f"{base_url}/metrics", timeout=10)
            if response.status_code == 200:
                print_success("Metrics endpoint responding")
                category_results["passed"] += 1
            else:
                print_error(f"Metrics endpoint failed: {response.status_code}")
                category_results["failed"] += 1
            
            # Test authentication
            print_info("Testing API authentication...")
            
            # Test without API key
            response = requests.post(f"{base_url}/query", 
                                   json={"query": "SELECT COUNT(*) FROM incident"}, 
                                   timeout=10)
            if response.status_code == 403:
                print_success("Authentication required (no API key rejected)")
                category_results["passed"] += 1
            else:
                print_error("Authentication bypass detected")
                category_results["failed"] += 1
            
            # Test with valid API key
            headers = {"Authorization": "Bearer epcl-demo-key-2024"}
            response = requests.post(f"{base_url}/query",
                                   json={"query": "SELECT COUNT(*) FROM incident"},
                                   headers=headers,
                                   timeout=30)
            if response.status_code == 200:
                print_success("Valid API key accepted")
                category_results["passed"] += 1
            else:
                print_warning(f"Query endpoint issue: {response.status_code}")
                category_results["warnings"] += 1
            
        except requests.exceptions.ConnectionError:
            print_warning("API server not running - start with 'python run_system.py --dev'")
            category_results["warnings"] += 1
        except Exception as e:
            print_error(f"API validation failed: {e}")
            category_results["failed"] += 1
        
        self.results["categories"]["api_endpoints"] = category_results
        return category_results
    
    def validate_langchain_agent(self) -> Dict[str, Any]:
        """Validate LangChain agent functionality."""
        print_header("LangChain Agent Validation")
        
        category_results = {"passed": 0, "failed": 0, "warnings": 0}
        
        # Check OpenAI API key
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your_openai_api_key_here":
            print_warning("OpenAI API key not configured - agent tests skipped")
            category_results["warnings"] += 1
            self.results["categories"]["langchain_agent"] = category_results
            return category_results
        
        try:
            from langchain_sql_agent import EPCLSQLAgent
            
            # Test agent initialization
            print_info("Testing agent initialization...")
            db_path = self.project_root / "epcl_vehs.db"
            if db_path.exists():
                agent = EPCLSQLAgent(str(db_path), openai_key)
                print_success("LangChain agent initialized successfully")
                category_results["passed"] += 1
                
                # Test schema retrieval
                print_info("Testing schema information...")
                schema_info = agent.get_schema_info()
                if schema_info and len(schema_info) > 0:
                    print_success(f"Schema information retrieved for {len(schema_info)} tables")
                    category_results["passed"] += 1
                else:
                    print_error("Schema information retrieval failed")
                    category_results["failed"] += 1
                
                # Test query suggestions
                print_info("Testing query suggestions...")
                suggestions = agent.suggest_queries()
                if suggestions and len(suggestions) > 0:
                    print_success(f"Generated {len(suggestions)} query suggestion categories")
                    category_results["passed"] += 1
                else:
                    print_error("Query suggestions generation failed")
                    category_results["failed"] += 1
            else:
                print_warning("Database not found - agent tests skipped")
                category_results["warnings"] += 1
            
        except Exception as e:
            print_error(f"LangChain agent validation failed: {e}")
            category_results["failed"] += 1
        
        self.results["categories"]["langchain_agent"] = category_results
        return category_results
    
    def validate_performance(self) -> Dict[str, Any]:
        """Validate system performance benchmarks."""
        print_header("Performance Validation")
        
        category_results = {"passed": 0, "failed": 0, "warnings": 0}
        
        try:
            from safe_sql_executor import SafeSQLExecutor
            
            db_path = self.project_root / "epcl_vehs.db"
            if not db_path.exists():
                print_warning("Database not found - performance tests skipped")
                category_results["warnings"] += 1
                self.results["categories"]["performance"] = category_results
                return category_results
            
            executor = SafeSQLExecutor(str(db_path))
            
            # Test query performance
            test_queries = [
                "SELECT COUNT(*) FROM incident",
                "SELECT category, COUNT(*) FROM incident GROUP BY category LIMIT 10",
                "SELECT * FROM incident LIMIT 100"
            ]
            
            print_info("Testing query performance...")
            for query in test_queries:
                start_time = time.time()
                result = executor.execute_query(query)
                execution_time = time.time() - start_time
                
                if result.success and execution_time < 5.0:
                    print_success(f"Query executed in {execution_time:.2f}s: {query[:50]}...")
                    category_results["passed"] += 1
                elif result.success:
                    print_warning(f"Slow query ({execution_time:.2f}s): {query[:50]}...")
                    category_results["warnings"] += 1
                else:
                    print_error(f"Query failed: {query[:50]}...")
                    category_results["failed"] += 1
            
            # Test cache performance
            print_info("Testing query caching...")
            cache_query = "SELECT COUNT(*) FROM incident"
            
            # First execution
            start_time = time.time()
            result1 = executor.execute_query(cache_query, use_cache=True)
            first_time = time.time() - start_time
            
            # Second execution (cached)
            start_time = time.time()
            result2 = executor.execute_query(cache_query, use_cache=True)
            cached_time = time.time() - start_time
            
            if result1.success and result2.success:
                print_success(f"Cache test passed: {first_time:.3f}s -> {cached_time:.3f}s")
                category_results["passed"] += 1
            else:
                print_error("Cache test failed")
                category_results["failed"] += 1
            
        except Exception as e:
            print_error(f"Performance validation failed: {e}")
            category_results["failed"] += 1
        
        self.results["categories"]["performance"] = category_results
        return category_results
    
    def run_validation(self, categories: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive system validation."""
        print_header("EPCL VEHS System Validation")
        print_info(f"Starting validation at {self.results['start_time']}")
        
        # Default categories
        if categories is None:
            categories = [
                "file_structure", "database", "sql_security", 
                "api_endpoints", "langchain_agent", "performance"
            ]
        
        # Run validation categories
        validation_methods = {
            "file_structure": self.validate_file_structure,
            "database": self.validate_database,
            "sql_security": self.validate_sql_security,
            "api_endpoints": self.validate_api_endpoints,
            "langchain_agent": self.validate_langchain_agent,
            "performance": self.validate_performance
        }
        
        for category in categories:
            if category in validation_methods:
                try:
                    category_result = validation_methods[category]()
                    self.results["passed"] += category_result["passed"]
                    self.results["failed"] += category_result["failed"]
                    self.results["warnings"] += category_result["warnings"]
                    self.results["total_checks"] += (
                        category_result["passed"] + 
                        category_result["failed"] + 
                        category_result["warnings"]
                    )
                except Exception as e:
                    print_error(f"Category '{category}' validation failed: {e}")
                    self.results["failed"] += 1
                    self.results["total_checks"] += 1
        
        # Generate final report
        self.generate_report()
        
        return self.results
    
    def generate_report(self):
        """Generate final validation report."""
        print_header("Validation Summary")
        
        end_time = datetime.now()
        duration = (end_time - self.results["start_time"]).total_seconds()
        
        print_info(f"Validation completed in {duration:.1f} seconds")
        print_info(f"Total checks: {self.results['total_checks']}")
        
        if self.results["passed"] > 0:
            print_success(f"Passed: {self.results['passed']}")
        
        if self.results["warnings"] > 0:
            print_warning(f"Warnings: {self.results['warnings']}")
        
        if self.results["failed"] > 0:
            print_error(f"Failed: {self.results['failed']}")
        
        # Overall status
        if self.results["failed"] == 0:
            if self.results["warnings"] == 0:
                print_success("🎉 All validations passed! System is ready for deployment.")
            else:
                print_warning("⚠️  System is functional but has warnings. Review above.")
        else:
            print_error("❌ System validation failed. Fix issues before deployment.")
        
        # Category breakdown
        print_info("\nCategory Results:")
        for category, results in self.results["categories"].items():
            status = "✅" if results["failed"] == 0 else "❌"
            print(f"  {status} {category}: {results['passed']} passed, {results['failed']} failed, {results['warnings']} warnings")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EPCL VEHS System Validator")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (file structure and database)")
    parser.add_argument("--full", action="store_true", help="Run full validation (all categories)")
    parser.add_argument("--security", action="store_true", help="Run security-focused validation")
    parser.add_argument("--performance", action="store_true", help="Run performance validation only")
    parser.add_argument("--api", action="store_true", help="Run API validation only")
    
    args = parser.parse_args()
    
    validator = SystemValidator()
    
    # Determine categories to run
    categories = None
    
    if args.quick:
        categories = ["file_structure", "database"]
    elif args.security:
        categories = ["file_structure", "sql_security", "api_endpoints"]
    elif args.performance:
        categories = ["performance"]
    elif args.api:
        categories = ["api_endpoints"]
    elif args.full:
        categories = None  # Run all categories
    else:
        # Default: run essential categories
        categories = ["file_structure", "database", "sql_security"]
    
    # Run validation
    results = validator.run_validation(categories)
    
    # Exit with appropriate code
    exit_code = 0 if results["failed"] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
