#!/usr/bin/env python3
"""
Simplified FastAPI Application for EPCL VEHS SQL Agent

This is a simplified version that focuses on core functionality
without complex LangChain configurations that might cause compatibility issues.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Local imports
from safe_sql_executor import SafeSQLExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "epcl_vehs.db"

# Global executor instance
sql_executor: Optional[SafeSQLExecutor] = None

# Metrics storage
metrics = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "start_time": datetime.now()
}

# Pydantic models
class QueryRequest(BaseModel):
    """Request model for SQL queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="SQL query")

class QueryResponse(BaseModel):
    """Response model for query results."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    row_count: int = 0
    execution_time: float = 0.0
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    database_connected: bool
    uptime_seconds: float

# No authentication required

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting EPCL VEHS Simple SQL API...")
    
    global sql_executor
    
    try:
        # Initialize SQL executor
        sql_executor = SafeSQLExecutor(DB_PATH, max_execution_time=30.0)
        logger.info("SQL Executor initialized successfully")
        
        # Test database connection
        test_result = sql_executor.execute_query("SELECT 1", use_cache=False)
        if test_result.success:
            logger.info("Database connection verified")
        else:
            logger.warning("Database connection test failed")
        
    except Exception as e:
        logger.error(f"Failed to initialize SQL executor: {e}")
        raise RuntimeError(f"Executor initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down EPCL VEHS Simple SQL API...")

# Create FastAPI app
app = FastAPI(
    title="EPCL VEHS Simple SQL API",
    description="Direct SQL API for EPCL VEHS Safety Data Analysis",
    version="1.0.0-simple",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def update_metrics(success: bool):
    """Update application metrics."""
    global metrics
    metrics["total_queries"] += 1
    if success:
        metrics["successful_queries"] += 1
    else:
        metrics["failed_queries"] += 1

# API Routes

@app.get("/", response_class=HTMLResponse)
async def simple_interface():
    """Serve a simple SQL interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EPCL VEHS Simple SQL Interface</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .query-section { margin: 20px 0; }
            textarea { width: 100%; height: 100px; padding: 10px; border: 2px solid #ddd; border-radius: 5px; font-family: monospace; }
            button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background: #2980b9; }
            .results { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 5px; }
            .error { background: #ffebee; color: #c62828; }
            .success { background: #e8f5e8; color: #2e7d32; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .examples { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .examples h3 { margin-top: 0; color: #1976d2; }
            .example-query { background: white; padding: 8px; margin: 5px 0; border-radius: 3px; cursor: pointer; font-family: monospace; font-size: 14px; }
            .example-query:hover { background: #f0f0f0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ EPCL VEHS Simple SQL Interface</h1>
            <p style="text-align: center; color: #666;">Direct SQL access to EPCL safety data with built-in security validation</p>
            
            <div class="examples">
                <h3>📝 Example Queries (Click to use):</h3>
                <div class="example-query" onclick="setQuery(this.textContent)">SELECT COUNT(*) FROM incident</div>
                <div class="example-query" onclick="setQuery(this.textContent)">SELECT category, COUNT(*) as count FROM incident GROUP BY category ORDER BY count DESC LIMIT 10</div>
                <div class="example-query" onclick="setQuery(this.textContent)">SELECT location, COUNT(*) as incidents FROM incident GROUP BY location ORDER BY incidents DESC LIMIT 5</div>
                <div class="example-query" onclick="setQuery(this.textContent)">SELECT incident_type, AVG(CAST(total_cost AS REAL)) as avg_cost FROM incident WHERE total_cost IS NOT NULL GROUP BY incident_type</div>
                <div class="example-query" onclick="setQuery(this.textContent)">SELECT status, COUNT(*) FROM incident GROUP BY status</div>
            </div>
            
            <div class="query-section">
                <label for="sqlQuery"><strong>SQL Query:</strong></label>
                <textarea id="sqlQuery" placeholder="Enter your SQL query here... (Only SELECT statements are allowed)"></textarea>
                <br><br>
                <button onclick="executeQuery()">Execute Query</button>
                <button onclick="clearResults()" style="background: #95a5a6;">Clear Results</button>
            </div>
            
            <div id="results" class="results" style="display: none;">
                <h3>Results:</h3>
                <div id="resultContent"></div>
            </div>
        </div>

        <script>
            // No API key required
            
            function setQuery(query) {
                document.getElementById('sqlQuery').value = query;
            }
            
            async function executeQuery() {
                const query = document.getElementById('sqlQuery').value.trim();
                const resultsDiv = document.getElementById('results');
                const resultContent = document.getElementById('resultContent');
                
                if (!query) {
                    alert('Please enter a SQL query');
                    return;
                }
                
                resultsDiv.style.display = 'block';
                resultContent.innerHTML = '<p>🔄 Executing query...</p>';
                
                try {
                    const response = await fetch('/execute', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ query: query })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        let html = `<div class="success">
                            <p><strong>✅ Query executed successfully!</strong></p>
                            <p>Rows returned: ${result.row_count} | Execution time: ${result.execution_time.toFixed(3)}s</p>
                        </div>`;
                        
                        if (result.data && result.data.length > 0) {
                            html += '<table><thead><tr>';
                            result.columns.forEach(col => {
                                html += `<th>${col}</th>`;
                            });
                            html += '</tr></thead><tbody>';
                            
                            result.data.forEach(row => {
                                html += '<tr>';
                                result.columns.forEach(col => {
                                    html += `<td>${row[col] || ''}</td>`;
                                });
                                html += '</tr>';
                            });
                            html += '</tbody></table>';
                        }
                        
                        resultContent.innerHTML = html;
                    } else {
                        resultContent.innerHTML = `<div class="error">
                            <p><strong>❌ Query failed:</strong></p>
                            <p>${result.error}</p>
                        </div>`;
                    }
                    
                } catch (error) {
                    resultContent.innerHTML = `<div class="error">
                        <p><strong>❌ Network error:</strong></p>
                        <p>${error.message}</p>
                    </div>`;
                }
            }
            
            function clearResults() {
                document.getElementById('results').style.display = 'none';
                document.getElementById('sqlQuery').value = '';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/execute", response_model=QueryResponse)
async def execute_sql(
    query_request: QueryRequest
):
    """Execute a SQL query with safety validation."""
    start_time = time.time()
    
    logger.info(f"Executing query: {query_request.query[:100]}...")
    
    try:
        if not sql_executor:
            raise HTTPException(status_code=503, detail="SQL executor not initialized")
        
        # Execute query through safe executor
        result = sql_executor.execute_query(query_request.query)
        
        execution_time = time.time() - start_time
        
        # Update metrics
        update_metrics(result.success)
        
        response = QueryResponse(
            success=result.success,
            data=result.data,
            columns=result.columns,
            row_count=result.row_count,
            execution_time=execution_time,
            error=result.error_message
        )
        
        logger.info(f"Query completed: success={result.success}, rows={result.row_count}, time={execution_time:.3f}s")
        
        return response
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Internal server error: {str(e)}"
        
        logger.error(f"Query execution failed: {error_msg}")
        update_metrics(False)
        
        return QueryResponse(
            success=False,
            error=error_msg,
            execution_time=execution_time
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - metrics["start_time"]).total_seconds()
    
    # Check database connection
    db_connected = False
    if sql_executor:
        try:
            test_result = sql_executor.execute_query("SELECT 1", use_cache=False)
            db_connected = test_result.success
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if db_connected else "unhealthy",
        timestamp=datetime.now().isoformat(),
        database_connected=db_connected,
        uptime_seconds=uptime
    )

@app.get("/metrics")
async def get_metrics():
    """Get application metrics."""
    uptime = (datetime.now() - metrics["start_time"]).total_seconds()
    success_rate = (metrics["successful_queries"] / max(metrics["total_queries"], 1)) * 100
    
    return {
        "total_queries": metrics["total_queries"],
        "successful_queries": metrics["successful_queries"],
        "failed_queries": metrics["failed_queries"],
        "success_rate": success_rate,
        "uptime_seconds": uptime
    }

@app.get("/schema")
async def get_schema():
    """Get database schema information."""
    try:
        if not sql_executor:
            raise HTTPException(status_code=503, detail="SQL executor not initialized")
        
        tables = ["incident", "hazard_id", "audit", "audit_findings", "inspection", "inspection_findings"]
        schema_info = {}
        
        for table in tables:
            schema = sql_executor.get_table_schema(table)
            sample = sql_executor.get_sample_data(table, limit=3)
            schema_info[table] = {"schema": schema, "sample": sample}
        
        return {"success": True, "schema": schema_info}
        
    except Exception as e:
        logger.error(f"Schema request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main execution
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting EPCL VEHS Simple API server on {host}:{port}")
    
    uvicorn.run(
        "main_simple:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
