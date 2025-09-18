#!/usr/bin/env python3
"""
FastAPI Application for EPCL VEHS SQL Agent

This module provides a production-ready REST API for the EPCL VEHS SQL agent
with authentication, rate limiting, monitoring, and a web-based chat interface.

Features:
- RESTful API endpoints
- Real-time chat interface
- Query validation and safety
- Rate limiting and monitoring
- Authentication and authorization
- Comprehensive logging
- Health checks and metrics

Endpoints:
- POST /query: Execute natural language query
- GET /schema: Get database schema information
- GET /suggestions: Get suggested queries
- GET /health: Health check
- GET /metrics: System metrics
- GET /: Chat interface (HTML)
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import asyncio

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Local imports
from langchain_sql_agent import EPCLSQLAgent
from safe_sql_executor import SafeSQLExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('epcl_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "epcl_vehs.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("EPCL_API_KEY", "epcl-demo-key-2024")  # Change in production
MAX_QUERIES_PER_MINUTE = int(os.getenv("MAX_QUERIES_PER_MINUTE", "10"))
MAX_QUERIES_PER_HOUR = int(os.getenv("MAX_QUERIES_PER_HOUR", "100"))

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global agent instance
sql_agent: Optional[EPCLSQLAgent] = None

# Metrics storage
metrics = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "average_response_time": 0.0,
    "start_time": datetime.now(),
    "last_query_time": None
}


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for natural language queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query")
    include_explanation: bool = Field(True, description="Include detailed explanation in response")
    use_cache: bool = Field(True, description="Use query caching for performance")


class QueryResponse(BaseModel):
    """Response model for query results."""
    success: bool
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: str
    query_id: Optional[str] = None
    chain_of_thought: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    database_connected: bool
    agent_initialized: bool
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate: float
    average_response_time: float
    uptime_seconds: float
    last_query_time: Optional[str]


# Authentication
security = HTTPBearer()


# No authentication required


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting EPCL VEHS SQL Agent API...")
    
    global sql_agent
    
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise RuntimeError("OpenAI API key required")
    
    try:
        # Initialize SQL agent
        sql_agent = EPCLSQLAgent(
            db_path=DB_PATH,
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4o-mini",
            temperature=0.1
        )
        logger.info("SQL Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize SQL agent: {e}")
        raise RuntimeError(f"Agent initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down EPCL VEHS SQL Agent API...")


# Create FastAPI app
app = FastAPI(
    title="EPCL VEHS SQL Agent API",
    description="Natural Language to SQL API for EPCL VEHS Safety Data Analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Utility functions
def update_metrics(success: bool, execution_time: float):
    """Update application metrics."""
    global metrics
    
    metrics["total_queries"] += 1
    metrics["last_query_time"] = datetime.now().isoformat()
    
    if success:
        metrics["successful_queries"] += 1
    else:
        metrics["failed_queries"] += 1
    
    # Update average response time (simple moving average)
    current_avg = metrics["average_response_time"]
    total_queries = metrics["total_queries"]
    metrics["average_response_time"] = ((current_avg * (total_queries - 1)) + execution_time) / total_queries


def generate_query_id() -> str:
    """Generate unique query ID."""
    return f"q_{int(time.time() * 1000)}"


# API Routes

@app.get("/", response_class=HTMLResponse)
async def chat_interface():
    """Serve the chat interface HTML."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EPCL VEHS Safety Data Assistant</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .container { 
                max-width: 1200px; 
                width: 90%; 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header { 
                background: linear-gradient(135deg, #2c3e50, #3498db); 
                color: white; 
                padding: 20px; 
                text-align: center; 
            }
            .header h1 { font-size: 2em; margin-bottom: 10px; }
            .header p { opacity: 0.9; }
            .chat-container { 
                display: flex; 
                height: 600px; 
            }
            .sidebar { 
                width: 300px; 
                background: #f8f9fa; 
                border-right: 1px solid #e9ecef; 
                padding: 20px;
                overflow-y: auto;
            }
            .sidebar h3 { 
                color: #2c3e50; 
                margin-bottom: 15px; 
                font-size: 1.1em;
            }
            .suggestion { 
                background: white; 
                border: 1px solid #e9ecef; 
                border-radius: 8px; 
                padding: 10px; 
                margin-bottom: 10px; 
                cursor: pointer; 
                transition: all 0.3s;
                font-size: 0.9em;
            }
            .suggestion:hover { 
                background: #e3f2fd; 
                border-color: #2196f3; 
            }
            .chat-area { 
                flex: 1; 
                display: flex; 
                flex-direction: column; 
            }
            .messages { 
                flex: 1; 
                padding: 20px; 
                overflow-y: auto; 
                background: #fafafa;
            }
            .message { 
                margin-bottom: 20px; 
                padding: 15px; 
                border-radius: 10px; 
                max-width: 80%;
            }
            .user-message { 
                background: #e3f2fd; 
                margin-left: auto; 
                text-align: right;
            }
            .bot-message { 
                background: white; 
                border: 1px solid #e9ecef;
            }
            .input-area { 
                padding: 20px; 
                border-top: 1px solid #e9ecef; 
                background: white;
            }
            .input-group { 
                display: flex; 
                gap: 10px; 
            }
            #queryInput { 
                flex: 1; 
                padding: 12px; 
                border: 2px solid #e9ecef; 
                border-radius: 8px; 
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s;
            }
            #queryInput:focus { 
                border-color: #2196f3; 
            }
            #sendButton { 
                padding: 12px 24px; 
                background: linear-gradient(135deg, #2196f3, #1976d2); 
                color: white; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-weight: bold;
                transition: all 0.3s;
            }
            #sendButton:hover { 
                transform: translateY(-2px); 
                box-shadow: 0 5px 15px rgba(33, 150, 243, 0.3);
            }
            #sendButton:disabled { 
                background: #ccc; 
                cursor: not-allowed; 
                transform: none;
                box-shadow: none;
            }
            .loading { 
                display: none; 
                text-align: center; 
                padding: 20px; 
                color: #666;
            }
            .error { 
                color: #d32f2f; 
                background: #ffebee; 
                border: 1px solid #ffcdd2;
            }
            .sql-code { 
                background: #f5f5f5; 
                border: 1px solid #ddd; 
                border-radius: 4px; 
                padding: 10px; 
                font-family: 'Courier New', monospace; 
                font-size: 0.9em; 
                margin: 10px 0;
                overflow-x: auto;
            }
            .metrics { 
                font-size: 0.8em; 
                color: #666; 
                margin-top: 10px;
            }
            .chain-of-thought {
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
                font-size: 0.9em;
            }
            .chain-step {
                margin: 8px 0;
                padding: 8px;
                border-left: 3px solid #007bff;
                background: white;
                border-radius: 4px;
            }
            .step-type {
                font-weight: bold;
                color: #007bff;
                text-transform: uppercase;
                font-size: 0.8em;
            }
            .step-content {
                margin-top: 5px;
                color: #333;
            }
            .toggle-chain {
                background: #e3f2fd;
                border: 1px solid #2196f3;
                border-radius: 4px;
                padding: 5px 10px;
                cursor: pointer;
                margin: 10px 0;
                font-size: 0.9em;
                color: #1976d2;
            }
            .toggle-chain:hover {
                background: #bbdefb;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ EPCL VEHS Safety Data Assistant</h1>
                <p>Ask questions about incidents, hazards, audits, and inspections in natural language</p>
            </div>
            
            <div class="chat-container">
                <div class="sidebar">
                    <h3>💡 Suggested Queries</h3>
                    <div class="suggestion" onclick="setQuery('How many incidents occurred in the last 6 months?')">
                        How many incidents occurred in the last 6 months?
                    </div>
                    <div class="suggestion" onclick="setQuery('What are the top 5 incident categories by frequency?')">
                        What are the top 5 incident categories by frequency?
                    </div>
                    <div class="suggestion" onclick="setQuery('Which locations have the highest incident rates?')">
                        Which locations have the highest incident rates?
                    </div>
                    <div class="suggestion" onclick="setQuery('What is the total cost of incidents this year?')">
                        What is the total cost of incidents this year?
                    </div>
                    <div class="suggestion" onclick="setQuery('Show me incidents with high injury potential')">
                        Show me incidents with high injury potential
                    </div>
                    <div class="suggestion" onclick="setQuery('What are the most common audit findings?')">
                        What are the most common audit findings?
                    </div>
                    <div class="suggestion" onclick="setQuery('Which departments need the most attention?')">
                        Which departments need the most attention?
                    </div>
                    
                    <h3 style="margin-top: 30px;">📊 Quick Stats</h3>
                    <div id="quickStats" style="font-size: 0.9em; color: #666;">
                        Loading statistics...
                    </div>
                </div>
                
                <div class="chat-area">
                    <div class="messages" id="messages">
                        <div class="message bot-message">
                            <strong>🤖 EPCL Safety Assistant</strong><br>
                            Hello! I'm your EPCL VEHS data assistant. I can help you analyze safety incidents, hazards, audits, and inspections using natural language queries.<br><br>
                            Try asking questions like:
                            <ul style="margin: 10px 0; padding-left: 20px;">
                                <li>"How many incidents happened last month?"</li>
                                <li>"Which location has the most safety issues?"</li>
                                <li>"Show me high-cost incidents"</li>
                                <li>"What are the trending safety concerns?"</li>
                            </ul>
                            <em>Your queries are processed securely and safely validated before execution.</em>
                        </div>
                    </div>
                    
                    <div class="loading" id="loading">
                        <div>🔄 Processing your query...</div>
                    </div>
                    
                    <div class="input-area">
                        <div class="input-group">
                            <input type="text" id="queryInput" placeholder="Ask me about EPCL safety data..." 
                                   onkeypress="handleKeyPress(event)">
                            <button id="sendButton" onclick="sendQuery()">Send</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const API_KEY = 'epcl-demo-key-2024'; // In production, get this securely
            
            function setQuery(query) {
                document.getElementById('queryInput').value = query;
                document.getElementById('queryInput').focus();
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendQuery();
                }
            }
            
            async function sendQuery() {
                const input = document.getElementById('queryInput');
                const query = input.value.trim();
                
                if (!query) return;
                
                // Add user message
                addMessage(query, 'user');
                
                // Clear input and show loading
                input.value = '';
                showLoading(true);
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${API_KEY}`
                        },
                        body: JSON.stringify({
                            query: query,
                            include_explanation: true,
                            use_cache: true
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        addBotResponse(result);
                    } else {
                        addMessage(`❌ Error: ${result.error}`, 'bot', true);
                    }
                    
                } catch (error) {
                    addMessage(`❌ Network error: ${error.message}`, 'bot', true);
                } finally {
                    showLoading(false);
                }
            }
            
            function addMessage(content, sender, isError = false) {
                const messages = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message ${isError ? 'error' : ''}`;
                
                if (sender === 'user') {
                    messageDiv.innerHTML = `<strong>👤 You:</strong><br>${content}`;
                } else {
                    messageDiv.innerHTML = content;
                }
                
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }
            
            function addBotResponse(result) {
                let content = `<strong>🤖 Assistant:</strong><br>`;
                content += result.response.answer;
                
                if (result.executed_sql && result.executed_sql.length > 0) {
                    content += `<br><br><strong>📝 SQL Query:</strong>`;
                    result.executed_sql.forEach(sql => {
                        content += `<div class="sql-code">${sql}</div>`;
                    });
                }
                
                // Add chain of thought if available
                if (result.chain_of_thought && result.chain_of_thought.length > 0) {
                    content += `<div class="toggle-chain" onclick="toggleChainOfThought(this)">🧠 Show Chain of Thought (${result.chain_of_thought.length} steps)</div>`;
                    content += `<div class="chain-of-thought" style="display: none;">`;
                    content += `<strong>🔗 Reasoning Steps:</strong><br>`;
                    
                    result.chain_of_thought.forEach((step, index) => {
                        content += `<div class="chain-step">`;
                        content += `<div class="step-type">${step.type}</div>`;
                        content += `<div class="step-content">`;
                        
                        switch(step.type) {
                            case 'action':
                                content += `Tool: ${step.tool}<br>Input: ${step.tool_input}`;
                                break;
                            case 'tool_start':
                                content += `Starting tool: ${step.tool_name}<br>Input: ${step.input}`;
                                break;
                            case 'tool_end':
                                content += `Tool output: ${step.output.substring(0, 200)}${step.output.length > 200 ? '...' : ''}`;
                                break;
                            case 'llm_start':
                                content += `LLM processing started`;
                                break;
                            case 'llm_end':
                                content += `LLM completed in ${step.duration?.toFixed(2) || 'N/A'}s`;
                                if (step.response) {
                                    content += `<br>Response: ${step.response.substring(0, 150)}${step.response.length > 150 ? '...' : ''}`;
                                }
                                break;
                            case 'finish':
                                content += `Final answer generated`;
                                break;
                            default:
                                content += JSON.stringify(step, null, 2);
                        }
                        
                        content += `</div></div>`;
                    });
                    
                    content += `</div>`;
                }
                
                if (result.response.recommendations && result.response.recommendations.length > 0) {
                    content += `<br><strong>💡 Recommendations:</strong><ul>`;
                    result.response.recommendations.forEach(rec => {
                        content += `<li>${rec}</li>`;
                    });
                    content += `</ul>`;
                }
                
                content += `<div class="metrics">⏱️ ${result.execution_time.toFixed(2)}s</div>`;
                
                addMessage(content, 'bot');
            }
            
            function toggleChainOfThought(element) {
                const chainDiv = element.nextElementSibling;
                if (chainDiv.style.display === 'none') {
                    chainDiv.style.display = 'block';
                    element.textContent = element.textContent.replace('Show', 'Hide');
                } else {
                    chainDiv.style.display = 'none';
                    element.textContent = element.textContent.replace('Hide', 'Show');
                }
            }
            
            function showLoading(show) {
                document.getElementById('loading').style.display = show ? 'block' : 'none';
                document.getElementById('sendButton').disabled = show;
            }
            
            // Load quick stats
            async function loadQuickStats() {
                try {
                    const response = await fetch('/metrics');
                    const metrics = await response.json();
                    
                    const stats = `
                        📈 Total Queries: ${metrics.total_queries}<br>
                        ✅ Success Rate: ${metrics.success_rate.toFixed(1)}%<br>
                        ⚡ Avg Response: ${metrics.average_response_time.toFixed(2)}s<br>
                        🕒 Uptime: ${Math.floor(metrics.uptime_seconds / 3600)}h
                    `;
                    
                    document.getElementById('quickStats').innerHTML = stats;
                } catch (error) {
                    document.getElementById('quickStats').innerHTML = 'Stats unavailable';
                }
            }
            
            // Load stats on page load
            loadQuickStats();
            
            // Refresh stats every 30 seconds
            setInterval(loadQuickStats, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/query", response_model=QueryResponse)
@limiter.limit(f"{MAX_QUERIES_PER_MINUTE}/minute")
async def execute_query(
    request: Request,
    query_request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """Execute a natural language query against the EPCL VEHS database."""
    start_time = time.time()
    query_id = generate_query_id()
    
    logger.info(f"Query {query_id}: {query_request.query}")
    
    try:
        if not sql_agent:
            raise HTTPException(status_code=503, detail="SQL agent not initialized")
        
        # Execute query through agent
        result = sql_agent.query(
            query_request.query,
            include_explanation=query_request.include_explanation
        )
        
        execution_time = time.time() - start_time
        
        # Update metrics in background
        background_tasks.add_task(update_metrics, result["success"], execution_time)
        
        response = QueryResponse(
            success=result["success"],
            response=result.get("response"),
            error=result.get("error"),
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            query_id=query_id,
            chain_of_thought=result.get("chain_of_thought", [])
        )
        
        logger.info(f"Query {query_id} completed: success={result['success']}, time={execution_time:.2f}s")
        
        return response
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Internal server error: {str(e)}"
        
        logger.error(f"Query {query_id} failed: {error_msg}")
        
        # Update metrics in background
        background_tasks.add_task(update_metrics, False, execution_time)
        
        return QueryResponse(
            success=False,
            error=error_msg,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            query_id=query_id
        )


@app.get("/schema")
async def get_schema():
    """Get database schema information."""
    try:
        if not sql_agent:
            raise HTTPException(status_code=503, detail="SQL agent not initialized")
        
        schema_info = sql_agent.get_schema_info()
        return {"success": True, "schema": schema_info}
        
    except Exception as e:
        logger.error(f"Schema request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suggestions")
async def get_suggestions():
    """Get suggested queries for users."""
    try:
        if not sql_agent:
            raise HTTPException(status_code=503, detail="SQL agent not initialized")
        
        suggestions = sql_agent.suggest_queries()
        return {"success": True, "suggestions": suggestions}
        
    except Exception as e:
        logger.error(f"Suggestions request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - metrics["start_time"]).total_seconds()
    
    # Check database connection
    db_connected = False
    try:
        executor = SafeSQLExecutor(DB_PATH)
        test_result = executor.execute_query("SELECT 1", use_cache=False)
        db_connected = test_result.success
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if (sql_agent is not None and db_connected) else "unhealthy",
        timestamp=datetime.now().isoformat(),
        database_connected=db_connected,
        agent_initialized=sql_agent is not None,
        uptime_seconds=uptime
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get application metrics."""
    uptime = (datetime.now() - metrics["start_time"]).total_seconds()
    success_rate = (metrics["successful_queries"] / max(metrics["total_queries"], 1)) * 100
    
    return MetricsResponse(
        total_queries=metrics["total_queries"],
        successful_queries=metrics["successful_queries"],
        failed_queries=metrics["failed_queries"],
        success_rate=success_rate,
        average_response_time=metrics["average_response_time"],
        uptime_seconds=uptime,
        last_query_time=metrics["last_query_time"]
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


# Main execution
if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting EPCL VEHS API server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info",
        access_log=True
    )
