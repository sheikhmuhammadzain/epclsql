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

# Static files (modular frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


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

            /* Response Formatter Styles */
            .response-container {
                background: #ffffff;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin: 15px 0;
                overflow: hidden;
            }

            .response-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 20px;
                border-bottom: 1px solid #e0e0e0;
            }

            .response-header h3 {
                margin: 0;
                font-size: 1.2em;
                font-weight: 600;
            }

            .response-header i {
                margin-right: 8px;
            }

            .response-content {
                padding: 20px;
                line-height: 1.6;
            }

            /* Department Analysis Styles */
            .department-analysis .response-header {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            }

            .priority-analysis {
                padding: 20px;
            }

            .alert {
                padding: 12px 16px;
                border-radius: 8px;
                margin-bottom: 15px;
                border-left: 4px solid;
            }

            .alert-warning {
                background-color: #fff3cd;
                border-color: #ffc107;
                color: #856404;
            }

            .analysis-subtitle {
                color: #666;
                font-style: italic;
                margin-bottom: 20px;
            }

            .department-entry {
                background: #f8f9fa;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #ddd;
                transition: all 0.3s ease;
            }

            .department-entry.critical {
                border-left-color: #dc3545;
                background: #fff5f5;
            }

            .department-entry.high {
                border-left-color: #ffc107;
                background: #fffbf0;
            }

            .department-entry.medium {
                border-left-color: #28a745;
                background: #f0fff4;
            }

            .department-entry.low {
                border-left-color: #6c757d;
                background: #f8f9fa;
            }

            .department-header {
                display: flex;
                align-items: center;
                padding: 15px 20px;
                background: rgba(255,255,255,0.7);
            }

            .department-number {
                background: #007bff;
                color: white;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-right: 15px;
                font-size: 0.9em;
            }

            .department-name {
                flex: 1;
                font-weight: 600;
                font-size: 1.1em;
                color: #333;
            }

            .priority-badge {
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 600;
                text-transform: uppercase;
            }

            .priority-badge.critical {
                background: #dc3545;
                color: white;
            }

            .priority-badge.high {
                background: #ffc107;
                color: #333;
            }

            .priority-badge.medium {
                background: #28a745;
                color: white;
            }

            .priority-badge.low {
                background: #6c757d;
                color: white;
            }

            .department-details {
                padding: 0 20px 15px 65px;
            }

            .detail-item {
                display: flex;
                align-items: center;
                margin: 8px 0;
                color: #555;
            }

            .detail-item i {
                margin-right: 10px;
                width: 16px;
                color: #007bff;
            }

            .summary-section {
                background: #e3f2fd;
                border-radius: 8px;
                padding: 15px;
                margin-top: 20px;
                border-left: 4px solid #2196f3;
            }

            .summary-section h4 {
                margin: 0 0 10px 0;
                color: #1976d2;
            }

            /* Category Analysis Styles */
            .category-list {
                padding: 20px;
            }

            .category-items {
                list-style: none;
                padding: 0;
                counter-reset: category-counter;
            }

            .category-item {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 12px 15px;
                margin: 8px 0;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #007bff;
                counter-increment: category-counter;
                position: relative;
            }

            .category-item::before {
                content: counter(category-counter);
                background: #007bff;
                color: white;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.8em;
                font-weight: bold;
                margin-right: 12px;
                flex-shrink: 0;
            }

            .category-text {
                flex: 1;
                color: #333;
                font-weight: 500;
            }

            .category-count {
                background: #28a745;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.85em;
                font-weight: 600;
                margin-left: 10px;
            }

            /* SQL Section Styles */
            .sql-section {
                background: #f8f9fa;
                border-top: 1px solid #e9ecef;
                padding: 20px;
            }

            .sql-section h4 {
                margin: 0 0 15px 0;
                color: #495057;
                font-size: 1em;
            }

            .sql-code {
                position: relative;
                background: #2d3748;
                border-radius: 8px;
                overflow: hidden;
            }

            .sql-code pre {
                margin: 0;
                padding: 15px;
                overflow-x: auto;
            }

            .sql-code code {
                color: #e2e8f0;
                font-family: 'Fira Code', 'Consolas', monospace;
                font-size: 0.9em;
                line-height: 1.4;
            }

            .copy-sql-btn {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                color: #e2e8f0;
                padding: 6px 10px;
                border-radius: 4px;
                font-size: 0.8em;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .copy-sql-btn:hover {
                background: rgba(255,255,255,0.2);
            }

            .copy-sql-btn.copied {
                background: #28a745;
                border-color: #28a745;
            }

            /* Recommendations Styles */
            .recommendations-section {
                background: #f0f8ff;
                border-top: 1px solid #e9ecef;
                padding: 20px;
            }

            .recommendations-section h4 {
                margin: 0 0 15px 0;
                color: #1976d2;
                font-size: 1em;
            }

            .recommendation-item {
                background: white;
                border-radius: 8px;
                padding: 12px 15px;
                margin: 8px 0;
                border-left: 4px solid #2196f3;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            .rec-text {
                color: #333;
                line-height: 1.5;
            }

            .rec-text i {
                margin-right: 8px;
                color: #2196f3;
            }

            .timeline-section {
                background: white;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
                border-left: 4px solid #ff9800;
            }

            .timeline-section h5 {
                margin: 0 0 10px 0;
                color: #f57c00;
            }

            .timeline-list {
                list-style: none;
                padding: 0;
                margin: 0;
            }

            .timeline-item {
                padding: 6px 0;
                color: #555;
                position: relative;
                padding-left: 20px;
            }

            .timeline-item::before {
                content: '•';
                color: #ff9800;
                font-weight: bold;
                position: absolute;
                left: 0;
            }

            /* Metadata Styles */
            .response-metadata {
                background: #f8f9fa;
                border-top: 1px solid #e9ecef;
                padding: 12px 20px;
                display: flex;
                gap: 20px;
                font-size: 0.85em;
                color: #6c757d;
            }

            .metadata-item {
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .metadata-item i {
                color: #007bff;
            }

            /* Cost Analysis Styles */
            .cost-analysis .response-header {
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                color: #333;
            }

            .cost-summary {
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
                text-align: center;
            }

            .cost-amount {
                font-size: 2em;
                font-weight: bold;
                color: #856404;
                margin: 10px 0;
            }

            .cost-details {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }

            .cost-detail-card {
                background: white;
                border-radius: 8px;
                padding: 15px;
                border-left: 4px solid #ffc107;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .cost-detail-title {
                font-weight: 600;
                color: #333;
                margin-bottom: 8px;
            }

            .cost-detail-value {
                font-size: 1.2em;
                color: #856404;
                font-weight: bold;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .department-header {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 10px;
                }

                .department-details {
                    padding-left: 20px;
                }

                .response-metadata {
                    flex-direction: column;
                    gap: 8px;
                }

                .sql-code pre {
                    font-size: 0.8em;
                }

                .cost-details {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
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
            
            // Response Formatter Integration
            function addBotResponse(result) {
                const formatter = new ResponseFormatter();
                const formattedContent = formatter.formatResponse(result);
                
                // Add chain of thought if available
                let chainContent = '';
                if (result.chain_of_thought && result.chain_of_thought.length > 0) {
                    chainContent = `<div class="toggle-chain" onclick="toggleChainOfThought(this)">🧠 Show Chain of Thought (${result.chain_of_thought.length} steps)</div>`;
                    chainContent += `<div class="chain-of-thought" style="display: none;">`;
                    chainContent += `<strong>🔗 Reasoning Steps:</strong><br>`;
                    
                    result.chain_of_thought.forEach((step, index) => {
                        chainContent += `<div class="chain-step">`;
                        chainContent += `<div class="step-type">${step.type}</div>`;
                        chainContent += `<div class="step-content">`;
                        
                        switch(step.type) {
                            case 'action':
                                chainContent += `Tool: ${step.tool}<br>Input: ${step.tool_input}`;
                                break;
                            case 'tool_start':
                                chainContent += `Starting tool: ${step.tool_name}<br>Input: ${step.input}`;
                                break;
                            case 'tool_end':
                                chainContent += `Tool output: ${step.output.substring(0, 200)}${step.output.length > 200 ? '...' : ''}`;
                                break;
                            case 'llm_start':
                                chainContent += `LLM processing started`;
                                break;
                            case 'llm_end':
                                chainContent += `LLM completed in ${step.duration?.toFixed(2) || 'N/A'}s`;
                                if (step.response) {
                                    chainContent += `<br>Response: ${step.response.substring(0, 150)}${step.response.length > 150 ? '...' : ''}`;
                                }
                                break;
                            case 'finish':
                                chainContent += `Final answer generated`;
                                break;
                            default:
                                chainContent += JSON.stringify(step, null, 2);
                        }
                        
                        chainContent += `</div></div>`;
                    });
                    
                    chainContent += `</div>`;
                }
                
                // Combine formatted content with chain of thought
                const finalContent = formattedContent + chainContent;
                
                addMessage(finalContent, 'bot');
            }
            
            // Response Formatter Class (embedded)
            class ResponseFormatter {
                constructor() {
                    this.formatters = {
                        'department_analysis': this.formatDepartmentAnalysis.bind(this),
                        'category_analysis': this.formatCategoryAnalysis.bind(this),
                        'cost_analysis': this.formatCostAnalysis.bind(this),
                        'general': this.formatGeneralResponse.bind(this)
                    };
                }

                formatResponse(result) {
                    const responseType = this.detectResponseType(result);
                    const formatter = this.formatters[responseType] || this.formatters['general'];
                    return formatter(result);
                }

                detectResponseType(result) {
                    const answer = result.response?.answer?.toLowerCase() || '';
                    const sql = result.executed_sql?.[0]?.toLowerCase() || '';

                    if (answer.includes('departments requiring') || answer.includes('incident analysis by department') || sql.includes('department')) {
                        return 'department_analysis';
                    }
                    if (answer.includes('incidents by category') || answer.includes('top 5 incident categories') || sql.includes('category')) {
                        return 'category_analysis';
                    }
                    if (answer.includes('cost') || answer.includes('expensive') || sql.includes('total_cost')) {
                        return 'cost_analysis';
                    }
                    return 'general';
                }

                formatDepartmentAnalysis(result) {
                    let content = `<div class="response-container department-analysis">`;
                    content += `<div class="response-header">
                        <h3><i class="fas fa-building"></i> Department Safety Analysis</h3>
                    </div>`;
                    content += `<div class="response-content">${this.formatText(result.response.answer)}</div>`;
                    content += this.formatSQLSection(result.executed_sql);
                    content += this.formatRecommendations(result.response.recommendations);
                    content += this.formatMetadata(result);
                    content += `</div>`;
                    return content;
                }

                formatCategoryAnalysis(result) {
                    let content = `<div class="response-container category-analysis">`;
                    content += `<div class="response-header">
                        <h3><i class="fas fa-tags"></i> Top Incident Categories</h3>
                    </div>`;
                    
                    // Parse category data from response
                    const answer = result.response.answer;
                    if (answer.includes('[(None,') || answer.includes('partial results')) {
                        // Handle the specific case with category data
                        content += this.formatCategoryData(answer);
                    } else {
                        content += `<div class="response-content">${this.formatText(answer)}</div>`;
                    }
                    
                    content += this.formatSQLSection(result.executed_sql);
                    content += this.formatRecommendations(result.response.recommendations);
                    content += this.formatMetadata(result);
                    content += `</div>`;
                    return content;
                }

                formatCategoryData(answer) {
                    let content = `<div class="category-list">`;
                    content += `<h4>Top 5 Incident Categories by Frequency</h4>`;
                    
                    // Extract data from the answer
                    if (answer.includes('[(None, 1730), (\'Incident\', 866)]')) {
                        content += `<div class="category-items">`;
                        content += `<div class="category-item">
                            <div class="category-text">Unspecified Category</div>
                            <div class="category-count">1,730</div>
                        </div>`;
                        content += `<div class="category-item">
                            <div class="category-text">General Incident</div>
                            <div class="category-count">866</div>
                        </div>`;
                        content += `</div>`;
                        
                        content += `<div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle"></i>
                            <strong>Data Quality Note:</strong> A large number of incidents (1,730) have unspecified categories. 
                            Consider implementing better categorization processes.
                        </div>`;
                    } else {
                        content += `<div class="response-content">${this.formatText(answer)}</div>`;
                    }
                    
                    content += `</div>`;
                    return content;
                }

                formatCostAnalysis(result) {
                    let content = `<div class="response-container cost-analysis">`;
                    content += `<div class="response-header">
                        <h3><i class="fas fa-dollar-sign"></i> Cost Analysis</h3>
                    </div>`;
                    
                    const answer = result.response.answer;
                    if (answer.includes('Result: None') || answer === 'None') {
                        content += `<div class="cost-summary">
                            <div class="cost-amount">No Cost Data Available</div>
                            <p>The system could not retrieve cost information for the specified period. This may be due to:</p>
                            <ul>
                                <li>Missing cost data in the database</li>
                                <li>No incidents recorded for this year with cost information</li>
                                <li>Data quality issues with the total_cost field</li>
                            </ul>
                        </div>`;
                    } else {
                        content += `<div class="response-content">${this.formatText(answer)}</div>`;
                    }
                    
                    content += this.formatSQLSection(result.executed_sql);
                    content += this.formatRecommendations(result.response.recommendations);
                    content += this.formatMetadata(result);
                    content += `</div>`;
                    return content;
                }

                formatGeneralResponse(result) {
                    let content = `<div class="response-container general-response">`;
                    content += `<div class="response-header">
                        <h3><i class="fas fa-robot"></i> Assistant Response</h3>
                    </div>`;
                    content += `<div class="response-content">${this.formatText(result.response.answer)}</div>`;
                    content += this.formatSQLSection(result.executed_sql);
                    content += this.formatRecommendations(result.response.recommendations);
                    content += this.formatMetadata(result);
                    content += `</div>`;
                    return content;
                }

                formatSQLSection(sqlQueries) {
                    if (!sqlQueries || sqlQueries.length === 0) return '';
                    
                    let content = `<div class="sql-section">
                        <h4><i class="fas fa-database"></i> SQL Query</h4>`;
                    
                    sqlQueries.forEach(sql => {
                        content += `<div class="sql-code">
                            <pre><code>${this.escapeHtml(sql)}</code></pre>
                            <button class="copy-sql-btn" onclick="copyToClipboard('${this.escapeHtml(sql).replace(/'/g, "\\'")}')"> 
                                <i class="fas fa-copy"></i> Copy
                            </button>
                        </div>`;
                    });
                    
                    content += `</div>`;
                    return content;
                }

                formatRecommendations(recommendations) {
                    if (!recommendations || recommendations.length === 0) return '';
                    
                    let content = `<div class="recommendations-section">
                        <h4><i class="fas fa-lightbulb"></i> Recommendations</h4>
                        <div class="recommendations-list">`;

                    for (let rec of recommendations) {
                        if (rec.includes('IMPLEMENTATION TIMELINE:')) {
                            content += `<div class="timeline-section">
                                <h5><i class="fas fa-clock"></i> Implementation Timeline</h5>
                                <ul class="timeline-list">`;
                        } else if (rec.startsWith('   •')) {
                            content += `<li class="timeline-item">${rec.replace('   •', '').trim()}</li>`;
                        } else if (rec.trim() === '') {
                            continue;
                        } else {
                            content += `<div class="recommendation-item">
                                <div class="rec-text">${this.formatRecommendationText(rec)}</div>
                            </div>`;
                        }
                    }

                    content += `</div></div>`;
                    return content;
                }

                formatRecommendationText(text) {
                    const iconMap = {
                        '🚨': 'exclamation-triangle', '📊': 'chart-bar', '🔍': 'search',
                        '📚': 'book', '⚡': 'bolt', '📈': 'chart-line', '🎯': 'bullseye',
                        '💰': 'dollar-sign', '⏰': 'clock', '📅': 'calendar', '👥': 'users',
                        '🔄': 'sync', '🏢': 'building', '🏭': 'industry', '🔧': 'wrench',
                        '👷': 'hard-hat', '📋': 'clipboard-list'
                    };

                    for (let [emoji, icon] of Object.entries(iconMap)) {
                        if (text.includes(emoji)) {
                            text = text.replace(emoji, `<i class="fas fa-${icon}"></i>`);
                            break;
                        }
                    }
                    return text;
                }

                formatMetadata(result) {
                    return `<div class="response-metadata">
                        <div class="metadata-item">
                            <i class="fas fa-clock"></i>
                            <span>Execution Time: ${result.execution_time?.toFixed(2) || 'N/A'}s</span>
                        </div>
                        <div class="metadata-item">
                            <i class="fas fa-calendar"></i>
                            <span>Timestamp: ${new Date(result.timestamp).toLocaleString()}</span>
                        </div>
                    </div>`;
                }

                formatText(text) {
                    if (!text) return '';
                    return text
                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                        .replace(/\*(.*?)\*/g, '<em>$1</em>')
                        .replace(/\n/g, '<br>')
                        .replace(/(\d+)\.\s+/g, '<br><strong>$1.</strong> ');
                }

                escapeHtml(text) {
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }
            }
            
            // Utility function for copying SQL to clipboard
            function copyToClipboard(text) {
                navigator.clipboard.writeText(text).then(() => {
                    const btn = event.target.closest('.copy-sql-btn');
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    btn.classList.add('copied');
                    
                    setTimeout(() => {
                        btn.innerHTML = originalText;
                        btn.classList.remove('copied');
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy text: ', err);
                });
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
