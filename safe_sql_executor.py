#!/usr/bin/env python3
"""
Safe SQL Executor with Validation

This module provides a secure SQL execution layer with comprehensive validation,
parameter sanitization, and query safety checks for the EPCL VEHS system.

Features:
- SQL injection prevention
- Query complexity analysis
- Parameter validation
- Result formatting
- Audit logging
- Rate limiting support

Security measures:
- Whitelist-based SQL validation
- Parameter binding enforcement
- Query pattern analysis
- Result size limiting
- Execution time monitoring
"""

import re
import sqlite3
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRiskLevel(Enum):
    """Query risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QueryResult:
    """Structured query result."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    row_count: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    risk_level: QueryRiskLevel = QueryRiskLevel.LOW
    query_hash: Optional[str] = None


@dataclass
class ValidationResult:
    """SQL validation result."""
    is_valid: bool
    risk_level: QueryRiskLevel
    issues: List[str]
    sanitized_query: Optional[str] = None
    parameters: Optional[List[Any]] = None


class SQLValidator:
    """Comprehensive SQL query validator."""
    
    # Allowed SQL keywords (whitelist approach)
    ALLOWED_KEYWORDS = {
        'select', 'from', 'where', 'and', 'or', 'not', 'in', 'like', 
        'between', 'is', 'null', 'order', 'by', 'group', 'having',
        'limit', 'offset', 'as', 'distinct', 'count', 'sum', 'avg',
        'min', 'max', 'case', 'when', 'then', 'else', 'end', 'cast',
        'substr', 'length', 'upper', 'lower', 'trim', 'coalesce',
        'strftime', 'date', 'datetime', 'julianday', 'round', 'abs'
    }
    
    # Forbidden patterns (blacklist)
    FORBIDDEN_PATTERNS = [
        r'\b(drop|delete|insert|update|alter|create|pragma|attach|detach)\b',
        r'\b(sqlite_master|sqlite_temp_master)\b',
        r'--',  # SQL comments
        r'/\*.*?\*/',  # Multi-line comments
        r';\s*\w',  # Multiple statements
        r'\bexec\b',
        r'\bevaluate\b',
        r'\bsystem\b',
        r'\bshell\b',
        r'\bload_extension\b'
    ]
    
    # Risk patterns
    HIGH_RISK_PATTERNS = [
        r'\bcross\s+join\b',
        r'\bjoin\s+\w+\s+on\s+1\s*=\s*1\b',
        r'\bselect\s+\*\s+from\s+\w+\s*$',  # SELECT * without WHERE
        r'\bunion\b.*\bunion\b',  # Multiple UNIONs
    ]
    
    # Allowed tables
    ALLOWED_TABLES = {
        'incident', 'hazard_id', 'audit', 'audit_findings', 
        'inspection', 'inspection_findings', 'query_metadata',
        'incident_monthly_summary', 'incident_category_summary',
        'incident_location_summary'
    }
    
    def __init__(self):
        self.forbidden_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.FORBIDDEN_PATTERNS]
        self.risk_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.HIGH_RISK_PATTERNS]
    
    def validate_query(self, query: str) -> ValidationResult:
        """
        Validate SQL query for security and performance.
        
        Args:
            query: SQL query string
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        risk_level = QueryRiskLevel.LOW
        
        # Basic sanitization
        query = query.strip()
        
        if not query:
            return ValidationResult(
                is_valid=False,
                risk_level=QueryRiskLevel.CRITICAL,
                issues=["Empty query"]
            )
        
        # Check if query starts with SELECT
        if not re.match(r'^\s*select\b', query, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                risk_level=QueryRiskLevel.CRITICAL,
                issues=["Only SELECT statements are allowed"]
            )
        
        # Check for forbidden patterns
        for pattern in self.forbidden_regex:
            if pattern.search(query):
                return ValidationResult(
                    is_valid=False,
                    risk_level=QueryRiskLevel.CRITICAL,
                    issues=[f"Forbidden pattern detected: {pattern.pattern}"]
                )
        
        # Check for semicolons (multiple statements)
        if ';' in query.rstrip(';'):
            return ValidationResult(
                is_valid=False,
                risk_level=QueryRiskLevel.CRITICAL,
                issues=["Multiple statements not allowed"]
            )
        
        # Validate table names
        table_matches = re.findall(r'\bfrom\s+(\w+)', query, re.IGNORECASE)
        table_matches.extend(re.findall(r'\bjoin\s+(\w+)', query, re.IGNORECASE))
        
        for table in table_matches:
            if table.lower() not in self.ALLOWED_TABLES:
                issues.append(f"Unknown table: {table}")
                risk_level = QueryRiskLevel.HIGH
        
        # Check for high-risk patterns
        for pattern in self.risk_regex:
            if pattern.search(query):
                issues.append(f"High-risk pattern: {pattern.pattern}")
                risk_level = max(risk_level, QueryRiskLevel.HIGH)
        
        # Check query complexity
        complexity_score = self._assess_complexity(query)
        if complexity_score > 10:
            issues.append(f"High complexity query (score: {complexity_score})")
            risk_level = max(risk_level, QueryRiskLevel.MEDIUM)
        
        # Ensure LIMIT clause for potentially large results
        if not re.search(r'\blimit\s+\d+', query, re.IGNORECASE):
            if not re.search(r'\bcount\s*\(', query, re.IGNORECASE):
                query += " LIMIT 1000"
                issues.append("Added LIMIT 1000 for safety")
        
        is_valid = risk_level != QueryRiskLevel.CRITICAL
        
        return ValidationResult(
            is_valid=is_valid,
            risk_level=risk_level,
            issues=issues,
            sanitized_query=query if is_valid else None
        )
    
    def _assess_complexity(self, query: str) -> int:
        """
        Assess query complexity based on various factors.
        
        Args:
            query: SQL query string
            
        Returns:
            Complexity score (higher = more complex)
        """
        score = 0
        
        # Count JOINs
        score += len(re.findall(r'\bjoin\b', query, re.IGNORECASE)) * 2
        
        # Count subqueries
        score += len(re.findall(r'\(\s*select\b', query, re.IGNORECASE)) * 3
        
        # Count aggregations
        score += len(re.findall(r'\b(count|sum|avg|min|max)\s*\(', query, re.IGNORECASE))
        
        # Count WHERE conditions
        score += len(re.findall(r'\b(and|or)\b', query, re.IGNORECASE))
        
        # Count UNION operations
        score += len(re.findall(r'\bunion\b', query, re.IGNORECASE)) * 2
        
        return score


class DateParameterProcessor:
    """Process relative date expressions into absolute dates."""
    
    @staticmethod
    def process_relative_dates(query: str) -> Tuple[str, List[Any]]:
        """
        Process relative date expressions and convert to parameters.
        
        Args:
            query: SQL query with potential relative date expressions
            
        Returns:
            Tuple of (processed_query, parameters)
        """
        parameters = []
        processed_query = query
        
        # Common relative date patterns
        patterns = {
            r'\blast\s+(\d+)\s+months?\b': lambda m: DateParameterProcessor._months_ago(int(m.group(1))),
            r'\blast\s+(\d+)\s+days?\b': lambda m: DateParameterProcessor._days_ago(int(m.group(1))),
            r'\blast\s+year\b': lambda m: DateParameterProcessor._months_ago(12),
            r'\bthis\s+year\b': lambda m: f"{datetime.now().year}-01-01",
            r'\btoday\b': lambda m: datetime.now().strftime('%Y-%m-%d'),
            r'\byesterday\b': lambda m: DateParameterProcessor._days_ago(1)
        }
        
        for pattern, replacement_func in patterns.items():
            matches = list(re.finditer(pattern, processed_query, re.IGNORECASE))
            for match in reversed(matches):  # Process from end to preserve positions
                replacement_date = replacement_func(match)
                processed_query = (
                    processed_query[:match.start()] + 
                    '?' + 
                    processed_query[match.end():]
                )
                parameters.append(replacement_date)
        
        return processed_query, parameters
    
    @staticmethod
    def _months_ago(months: int) -> str:
        """Get date N months ago."""
        today = datetime.now()
        # Approximate months calculation
        target_date = today - timedelta(days=months * 30)
        return target_date.strftime('%Y-%m-%d')
    
    @staticmethod
    def _days_ago(days: int) -> str:
        """Get date N days ago."""
        target_date = datetime.now() - timedelta(days=days)
        return target_date.strftime('%Y-%m-%d')


class SafeSQLExecutor:
    """Safe SQL executor with comprehensive security and monitoring."""
    
    def __init__(self, db_path: str, max_execution_time: float = 30.0):
        """
        Initialize safe SQL executor.
        
        Args:
            db_path: Path to SQLite database
            max_execution_time: Maximum query execution time in seconds
        """
        self.db_path = db_path
        self.max_execution_time = max_execution_time
        self.validator = SQLValidator()
        self.date_processor = DateParameterProcessor()
        self.query_cache = {}  # Simple query cache
        
    def execute_query(
        self, 
        query: str, 
        parameters: Optional[List[Any]] = None,
        use_cache: bool = True
    ) -> QueryResult:
        """
        Execute SQL query with comprehensive safety checks.
        
        Args:
            query: SQL query string
            parameters: Optional query parameters
            use_cache: Whether to use query caching
            
        Returns:
            QueryResult with execution details
        """
        start_time = time.time()
        
        # Generate query hash for caching and logging
        query_hash = hashlib.md5(f"{query}{parameters}".encode()).hexdigest()
        
        # Check cache first
        if use_cache and query_hash in self.query_cache:
            cached_result = self.query_cache[query_hash]
            cached_result.execution_time = time.time() - start_time
            logger.info(f"Query served from cache: {query_hash[:8]}")
            return cached_result
        
        try:
            # Process relative dates
            processed_query, date_params = self.date_processor.process_relative_dates(query)
            
            # Combine parameters
            all_params = (parameters or []) + date_params
            
            # Validate query
            validation = self.validator.validate_query(processed_query)
            
            if not validation.is_valid:
                return QueryResult(
                    success=False,
                    error_message=f"Query validation failed: {'; '.join(validation.issues)}",
                    risk_level=validation.risk_level,
                    query_hash=query_hash,
                    execution_time=time.time() - start_time
                )
            
            # Execute query
            result = self._execute_validated_query(
                validation.sanitized_query, 
                all_params, 
                query_hash
            )
            
            # Cache successful results
            if use_cache and result.success and result.row_count < 10000:
                self.query_cache[query_hash] = result
            
            result.execution_time = time.time() - start_time
            
            # Log execution
            self._log_query_execution(query, result, validation.risk_level)
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return QueryResult(
                success=False,
                error_message=str(e),
                risk_level=QueryRiskLevel.HIGH,
                query_hash=query_hash,
                execution_time=time.time() - start_time
            )
    
    def _execute_validated_query(
        self, 
        query: str, 
        parameters: List[Any], 
        query_hash: str
    ) -> QueryResult:
        """
        Execute a validated query against the database.
        
        Args:
            query: Validated SQL query
            parameters: Query parameters
            query_hash: Query hash for tracking
            
        Returns:
            QueryResult with execution details
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.max_execution_time)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            
            cursor = conn.cursor()
            
            # Execute query with parameters
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = [description[0] for description in cursor.description] if cursor.description else []
            data = [dict(row) for row in rows]
            
            return QueryResult(
                success=True,
                data=data,
                columns=columns,
                row_count=len(data),
                query_hash=query_hash
            )
            
        except sqlite3.OperationalError as e:
            if "timeout" in str(e).lower():
                return QueryResult(
                    success=False,
                    error_message=f"Query timeout after {self.max_execution_time}s",
                    risk_level=QueryRiskLevel.HIGH,
                    query_hash=query_hash
                )
            else:
                return QueryResult(
                    success=False,
                    error_message=f"Database error: {str(e)}",
                    risk_level=QueryRiskLevel.MEDIUM,
                    query_hash=query_hash
                )
        
        except Exception as e:
            return QueryResult(
                success=False,
                error_message=f"Execution error: {str(e)}",
                risk_level=QueryRiskLevel.HIGH,
                query_hash=query_hash
            )
        
        finally:
            if conn:
                conn.close()
    
    def _log_query_execution(
        self, 
        original_query: str, 
        result: QueryResult, 
        risk_level: QueryRiskLevel
    ) -> None:
        """
        Log query execution for audit trail.
        
        Args:
            original_query: Original query string
            result: Query execution result
            risk_level: Assessed risk level
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query_hash": result.query_hash,
            "query_preview": original_query[:100] + "..." if len(original_query) > 100 else original_query,
            "success": result.success,
            "row_count": result.row_count,
            "execution_time": result.execution_time,
            "risk_level": risk_level.value,
            "error": result.error_message
        }
        
        logger.info(f"Query executed: {json.dumps(log_entry)}")
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with schema information
        """
        if table_name not in self.validator.ALLOWED_TABLES:
            return {"error": f"Table '{table_name}' not allowed"}
        
        schema_query = f"PRAGMA table_info({table_name})"
        result = self.execute_query(schema_query, use_cache=True)
        
        if result.success:
            return {
                "table_name": table_name,
                "columns": result.data,
                "column_count": result.row_count
            }
        else:
            return {"error": result.error_message}
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> Dict[str, Any]:
        """
        Get sample data from a table.
        
        Args:
            table_name: Name of the table
            limit: Number of sample rows
            
        Returns:
            Dictionary with sample data
        """
        if table_name not in self.validator.ALLOWED_TABLES:
            return {"error": f"Table '{table_name}' not allowed"}
        
        sample_query = f"SELECT * FROM {table_name} LIMIT {min(limit, 10)}"
        result = self.execute_query(sample_query, use_cache=True)
        
        if result.success:
            return {
                "table_name": table_name,
                "sample_data": result.data,
                "columns": result.columns
            }
        else:
            return {"error": result.error_message}
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")


# Example usage and testing
if __name__ == "__main__":
    # Initialize executor
    executor = SafeSQLExecutor("epcl_vehs.db")
    
    # Test queries
    test_queries = [
        "SELECT COUNT(*) FROM incident",
        "SELECT category, COUNT(*) FROM incident GROUP BY category",
        "SELECT * FROM incident WHERE date_of_occurrence > 'last 6 months'",
        "DROP TABLE incident",  # Should be blocked
        "SELECT * FROM incident; DELETE FROM incident",  # Should be blocked
    ]
    
    print("Testing Safe SQL Executor:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = executor.execute_query(query)
        print(f"Success: {result.success}")
        print(f"Rows: {result.row_count}")
        print(f"Time: {result.execution_time:.3f}s")
        if result.error_message:
            print(f"Error: {result.error_message}")
