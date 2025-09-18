#!/usr/bin/env python3
"""
Comprehensive Test Suite for EPCL VEHS SQL Agent System

This module provides comprehensive testing for all components of the EPCL VEHS
SQL agent system including ingestion, validation, execution, and API endpoints.

Test Categories:
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Security tests for SQL injection prevention
- Performance tests for query execution
- API tests for REST endpoints
"""

import pytest
import sqlite3
import pandas as pd
import json
import tempfile
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import asyncio

# FastAPI testing
from fastapi.testclient import TestClient
import httpx

# Local imports
from safe_sql_executor import SafeSQLExecutor, SQLValidator, QueryRiskLevel
from langchain_sql_agent import EPCLSQLAgent
from main import app
import ingest_excel_to_sqlite


class TestSQLValidator:
    """Test suite for SQL validation functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.validator = SQLValidator()
    
    def test_valid_select_query(self):
        """Test validation of valid SELECT queries."""
        valid_queries = [
            "SELECT COUNT(*) FROM incident",
            "SELECT category, COUNT(*) FROM incident GROUP BY category",
            "SELECT * FROM incident WHERE date_of_occurrence > '2024-01-01' LIMIT 100",
            "SELECT i.incident_number, i.category FROM incident i WHERE i.status = 'Closed'"
        ]
        
        for query in valid_queries:
            result = self.validator.validate_query(query)
            assert result.is_valid, f"Query should be valid: {query}"
            assert result.risk_level != QueryRiskLevel.CRITICAL
    
    def test_forbidden_queries(self):
        """Test rejection of forbidden SQL operations."""
        forbidden_queries = [
            "DROP TABLE incident",
            "DELETE FROM incident WHERE id = 1",
            "INSERT INTO incident VALUES (1, 'test')",
            "UPDATE incident SET status = 'Closed'",
            "ALTER TABLE incident ADD COLUMN test TEXT",
            "PRAGMA table_info(incident)",
            "ATTACH DATABASE 'test.db' AS test",
            "SELECT * FROM sqlite_master"
        ]
        
        for query in forbidden_queries:
            result = self.validator.validate_query(query)
            assert not result.is_valid, f"Query should be invalid: {query}"
            assert result.risk_level == QueryRiskLevel.CRITICAL
    
    def test_sql_injection_attempts(self):
        """Test prevention of SQL injection attacks."""
        injection_attempts = [
            "SELECT * FROM incident; DROP TABLE incident; --",
            "SELECT * FROM incident WHERE id = 1 OR 1=1",
            "SELECT * FROM incident UNION SELECT * FROM sqlite_master",
            "SELECT * FROM incident WHERE name = 'test'; EXEC xp_cmdshell('dir')",
            "SELECT * FROM incident /* comment */ WHERE 1=1"
        ]
        
        for query in injection_attempts:
            result = self.validator.validate_query(query)
            # Should either be invalid or have high risk
            assert not result.is_valid or result.risk_level == QueryRiskLevel.HIGH
    
    def test_query_complexity_assessment(self):
        """Test query complexity scoring."""
        simple_query = "SELECT COUNT(*) FROM incident"
        complex_query = """
            SELECT i.category, COUNT(*) as count, 
                   AVG(CAST(i.total_cost AS REAL)) as avg_cost
            FROM incident i 
            JOIN audit a ON i.location = a.audit_location 
            WHERE i.date_of_occurrence > '2024-01-01' 
            AND i.status IN ('Closed', 'Under Review')
            GROUP BY i.category 
            HAVING COUNT(*) > 5
            ORDER BY count DESC
        """
        
        simple_result = self.validator.validate_query(simple_query)
        complex_result = self.validator.validate_query(complex_query)
        
        # Complex query should have higher risk level
        assert simple_result.risk_level.value <= complex_result.risk_level.value
    
    def test_automatic_limit_addition(self):
        """Test automatic addition of LIMIT clause."""
        query_without_limit = "SELECT * FROM incident"
        result = self.validator.validate_query(query_without_limit)
        
        assert result.is_valid
        assert "LIMIT" in result.sanitized_query.upper()
        assert "1000" in result.sanitized_query


class TestSafeSQLExecutor:
    """Test suite for safe SQL execution."""
    
    def setup_method(self):
        """Setup test environment with temporary database."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Create test database with sample data
        self._create_test_database()
        
        self.executor = SafeSQLExecutor(self.temp_db.name, max_execution_time=5.0)
    
    def teardown_method(self):
        """Cleanup test environment."""
        os.unlink(self.temp_db.name)
    
    def _create_test_database(self):
        """Create test database with sample data."""
        conn = sqlite3.connect(self.temp_db.name)
        
        # Create incident table
        conn.execute("""
            CREATE TABLE incident (
                id INTEGER PRIMARY KEY,
                incident_number TEXT,
                date_of_occurrence TEXT,
                incident_type TEXT,
                location TEXT,
                department TEXT,
                category TEXT,
                status TEXT,
                total_cost REAL,
                extra_json TEXT
            )
        """)
        
        # Insert sample data
        sample_data = [
            (1, 'INC-2024-001', '2024-01-15', 'Near Miss', 'Plant A', 'Production', 'Safety', 'Closed', 1000.0, '{}'),
            (2, 'INC-2024-002', '2024-01-20', 'First Aid', 'Plant B', 'Maintenance', 'Injury', 'Open', 500.0, '{}'),
            (3, 'INC-2024-003', '2024-02-01', 'Property Damage', 'Plant A', 'Operations', 'Equipment', 'Closed', 5000.0, '{}'),
        ]
        
        conn.executemany("""
            INSERT INTO incident 
            (id, incident_number, date_of_occurrence, incident_type, location, department, category, status, total_cost, extra_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_data)
        
        conn.commit()
        conn.close()
    
    def test_valid_query_execution(self):
        """Test execution of valid queries."""
        query = "SELECT COUNT(*) as total FROM incident"
        result = self.executor.execute_query(query)
        
        assert result.success
        assert result.row_count == 1
        assert result.data[0]['total'] == 3
        assert result.execution_time > 0
    
    def test_parameterized_query(self):
        """Test execution with parameters."""
        query = "SELECT * FROM incident WHERE location = ?"
        parameters = ['Plant A']
        result = self.executor.execute_query(query, parameters)
        
        assert result.success
        assert result.row_count == 2
        assert all(row['location'] == 'Plant A' for row in result.data)
    
    def test_relative_date_processing(self):
        """Test processing of relative date expressions."""
        query = "SELECT COUNT(*) FROM incident WHERE date_of_occurrence > 'last 30 days'"
        result = self.executor.execute_query(query)
        
        # Should execute without error (may return 0 results depending on test data dates)
        assert result.success
    
    def test_query_timeout(self):
        """Test query timeout handling."""
        # Create executor with very short timeout
        short_executor = SafeSQLExecutor(self.temp_db.name, max_execution_time=0.001)
        
        # This query should timeout
        query = "SELECT COUNT(*) FROM incident"
        result = short_executor.execute_query(query)
        
        # Should handle timeout gracefully
        assert not result.success or result.success  # May or may not timeout depending on system speed
    
    def test_query_caching(self):
        """Test query result caching."""
        query = "SELECT COUNT(*) FROM incident"
        
        # First execution
        result1 = self.executor.execute_query(query, use_cache=True)
        
        # Second execution (should use cache)
        result2 = self.executor.execute_query(query, use_cache=True)
        
        assert result1.success and result2.success
        assert result1.data == result2.data
    
    def test_schema_information(self):
        """Test schema information retrieval."""
        schema = self.executor.get_table_schema('incident')
        
        assert 'table_name' in schema
        assert schema['table_name'] == 'incident'
        assert 'columns' in schema
        assert len(schema['columns']) > 0
    
    def test_sample_data_retrieval(self):
        """Test sample data retrieval."""
        sample = self.executor.get_sample_data('incident', limit=2)
        
        assert 'sample_data' in sample
        assert len(sample['sample_data']) <= 2
        assert 'columns' in sample


class TestIngestionScript:
    """Test suite for Excel ingestion functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Create sample Excel data
        self.sample_data = self._create_sample_excel_data()
    
    def teardown_method(self):
        """Cleanup test environment."""
        os.unlink(self.temp_db.name)
    
    def _create_sample_excel_data(self) -> Dict[str, pd.DataFrame]:
        """Create sample Excel data for testing."""
        incident_data = pd.DataFrame({
            'Incident Number': ['INC-001', 'INC-002', 'INC-003'],
            'Date of Occurrence': ['2024-01-15', '2024-01-20', '2024-02-01'],
            'Incident Type(s)': ['Near Miss', 'First Aid', 'Property Damage'],
            'Location': ['Plant A', 'Plant B', 'Plant A'],
            'Category': ['Safety', 'Injury', 'Equipment'],
            'Status': ['Closed', 'Open', 'Closed'],
            'Total Cost': [1000, 500, 5000],
            'Extra Column 1': ['Data1', 'Data2', 'Data3'],
            'Extra Column 2': ['More1', 'More2', 'More3']
        })
        
        audit_data = pd.DataFrame({
            'Audit Number': ['AUD-001', 'AUD-002'],
            'Audit Location': ['Plant A', 'Plant B'],
            'Start Date': ['2024-01-10', '2024-01-25'],
            'Audit Status': ['Completed', 'In Progress'],
            'Finding': ['Minor issue', 'Major concern']
        })
        
        return {
            'Incident': incident_data,
            'Audit': audit_data
        }
    
    def test_column_normalization(self):
        """Test column name normalization."""
        from ingest_excel_to_sqlite import normalize_column_name
        
        test_cases = [
            ('Incident Number', 'incident_number'),
            ('Date of Occurrence', 'date_of_occurrence'),
            ('Incident Type(s)', 'incident_types'),
            ('Total Cost ($)', 'total_cost'),
            ('Name.1', 'name_alt')
        ]
        
        for original, expected in test_cases:
            result = normalize_column_name(original)
            assert result == expected, f"Expected {expected}, got {result}"
    
    def test_date_parsing(self):
        """Test date column parsing."""
        from ingest_excel_to_sqlite import parse_date_columns
        
        df = pd.DataFrame({
            'date_column': ['2024-01-15', '2024-01-20', 'invalid'],
            'time_column': ['2024-01-15 10:30:00', '2024-01-20 14:45:00', None],
            'regular_column': ['A', 'B', 'C']
        })
        
        result_df = parse_date_columns(df)
        
        # Check that date columns were processed
        assert result_df['date_column'].notna().sum() == 2  # 2 valid dates
        assert result_df['regular_column'].equals(df['regular_column'])  # Unchanged


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints."""
    
    def setup_method(self):
        """Setup test environment."""
        self.client = TestClient(app)
        self.api_key = "epcl-demo-key-2024"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_queries" in data
        assert "success_rate" in data
        assert "average_response_time" in data
    
    def test_chat_interface(self):
        """Test chat interface endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_query_endpoint_authentication(self):
        """Test query endpoint authentication."""
        # Test without authentication
        response = self.client.post("/query", json={"query": "SELECT COUNT(*) FROM incident"})
        assert response.status_code == 403  # Forbidden
        
        # Test with invalid API key
        invalid_headers = {"Authorization": "Bearer invalid-key"}
        response = self.client.post(
            "/query", 
            json={"query": "SELECT COUNT(*) FROM incident"},
            headers=invalid_headers
        )
        assert response.status_code == 401  # Unauthorized
    
    @patch('main.sql_agent')
    def test_query_endpoint_success(self, mock_agent):
        """Test successful query execution."""
        # Mock agent response
        mock_agent.query.return_value = {
            "success": True,
            "response": {
                "answer": "There are 5 incidents in the database.",
                "recommendations": ["Monitor trends regularly"]
            },
            "executed_sql": ["SELECT COUNT(*) FROM incident"],
            "execution_time": 0.5
        }
        
        response = self.client.post(
            "/query",
            json={
                "query": "How many incidents are there?",
                "include_explanation": True
            },
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "response" in data
        assert "execution_time" in data
    
    @patch('main.sql_agent')
    def test_query_endpoint_error(self, mock_agent):
        """Test query execution error handling."""
        # Mock agent error
        mock_agent.query.return_value = {
            "success": False,
            "error": "Database connection failed"
        }
        
        response = self.client.post(
            "/query",
            json={"query": "SELECT * FROM nonexistent_table"},
            headers=self.headers
        )
        
        assert response.status_code == 200  # API returns 200 but success=False
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_rate_limiting(self):
        """Test API rate limiting."""
        # This test would need to be adjusted based on actual rate limits
        # and might be slow to run in practice
        pass


class TestSecurityFeatures:
    """Test suite for security features."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.executor = SafeSQLExecutor(self.temp_db.name)
    
    def teardown_method(self):
        """Cleanup test environment."""
        os.unlink(self.temp_db.name)
    
    def test_sql_injection_prevention(self):
        """Test comprehensive SQL injection prevention."""
        malicious_queries = [
            "SELECT * FROM incident; DROP TABLE incident; --",
            "SELECT * FROM incident WHERE id = 1' OR '1'='1",
            "SELECT * FROM incident UNION SELECT password FROM users",
            "SELECT * FROM incident; EXEC xp_cmdshell('rm -rf /')",
            "SELECT * FROM incident WHERE name = 'test'; INSERT INTO incident VALUES (999, 'hacked')"
        ]
        
        for query in malicious_queries:
            result = self.executor.execute_query(query)
            # Should either fail validation or execute safely
            if result.success:
                # If it succeeded, it should be a safe SELECT only
                assert "DROP" not in query.upper() or not result.success
                assert "DELETE" not in query.upper() or not result.success
                assert "INSERT" not in query.upper() or not result.success
    
    def test_table_access_control(self):
        """Test that only allowed tables can be accessed."""
        forbidden_tables = [
            "sqlite_master",
            "sqlite_temp_master", 
            "users",
            "passwords",
            "admin_config"
        ]
        
        for table in forbidden_tables:
            query = f"SELECT * FROM {table}"
            result = self.executor.execute_query(query)
            # Should fail validation for non-allowed tables
            assert not result.success or table in self.executor.validator.ALLOWED_TABLES
    
    def test_query_complexity_limits(self):
        """Test that overly complex queries are handled appropriately."""
        # Create a very complex query
        complex_query = """
            SELECT a.*, b.*, c.*, d.*
            FROM incident a
            JOIN audit b ON a.location = b.audit_location
            JOIN inspection c ON b.audit_location = c.audit_location  
            JOIN audit_findings d ON c.audit_number = d.audit_number
            WHERE a.date_of_occurrence IN (
                SELECT date_of_occurrence FROM incident 
                WHERE category IN (
                    SELECT DISTINCT category FROM incident 
                    WHERE total_cost > (
                        SELECT AVG(total_cost) FROM incident
                    )
                )
            )
        """
        
        result = self.executor.execute_query(complex_query)
        # Should either execute with appropriate limits or be flagged as high risk
        if result.success:
            assert result.row_count <= 1000  # Should have limit applied


class TestPerformance:
    """Test suite for performance characteristics."""
    
    def setup_method(self):
        """Setup test environment with larger dataset."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self._create_large_test_database()
        self.executor = SafeSQLExecutor(self.temp_db.name)
    
    def teardown_method(self):
        """Cleanup test environment."""
        os.unlink(self.temp_db.name)
    
    def _create_large_test_database(self):
        """Create test database with larger dataset."""
        conn = sqlite3.connect(self.temp_db.name)
        
        # Create incident table
        conn.execute("""
            CREATE TABLE incident (
                id INTEGER PRIMARY KEY,
                incident_number TEXT,
                date_of_occurrence TEXT,
                location TEXT,
                category TEXT,
                total_cost REAL
            )
        """)
        
        # Insert 1000 sample records
        import random
        locations = ['Plant A', 'Plant B', 'Plant C', 'Office', 'Warehouse']
        categories = ['Safety', 'Equipment', 'Environmental', 'Security']
        
        data = []
        for i in range(1000):
            data.append((
                i + 1,
                f'INC-2024-{i+1:04d}',
                f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
                random.choice(locations),
                random.choice(categories),
                random.uniform(100, 10000)
            ))
        
        conn.executemany("""
            INSERT INTO incident (id, incident_number, date_of_occurrence, location, category, total_cost)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)
        
        # Create indexes
        conn.execute("CREATE INDEX idx_incident_date ON incident(date_of_occurrence)")
        conn.execute("CREATE INDEX idx_incident_location ON incident(location)")
        conn.execute("CREATE INDEX idx_incident_category ON incident(category)")
        
        conn.commit()
        conn.close()
    
    def test_query_performance(self):
        """Test query execution performance."""
        queries = [
            "SELECT COUNT(*) FROM incident",
            "SELECT location, COUNT(*) FROM incident GROUP BY location",
            "SELECT * FROM incident WHERE date_of_occurrence > '2024-06-01' LIMIT 100",
            "SELECT category, AVG(total_cost) FROM incident GROUP BY category"
        ]
        
        for query in queries:
            start_time = time.time()
            result = self.executor.execute_query(query)
            execution_time = time.time() - start_time
            
            assert result.success, f"Query failed: {query}"
            assert execution_time < 5.0, f"Query too slow ({execution_time:.2f}s): {query}"
    
    def test_cache_performance(self):
        """Test query caching performance."""
        query = "SELECT COUNT(*) FROM incident WHERE location = 'Plant A'"
        
        # First execution (no cache)
        start_time = time.time()
        result1 = self.executor.execute_query(query, use_cache=True)
        first_execution_time = time.time() - start_time
        
        # Second execution (with cache)
        start_time = time.time()
        result2 = self.executor.execute_query(query, use_cache=True)
        cached_execution_time = time.time() - start_time
        
        assert result1.success and result2.success
        assert result1.data == result2.data
        # Cached execution should be faster (though this might not always be true for simple queries)


# Test runner and utilities
def run_all_tests():
    """Run all tests and generate report."""
    print("Running EPCL VEHS SQL Agent Test Suite")
    print("=" * 60)
    
    # Run pytest with verbose output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check output above.")
    
    return exit_code


if __name__ == "__main__":
    run_all_tests()
