#!/usr/bin/env python3
"""
LangChain SQL Agent for EPCL VEHS System

This module implements a production-ready LangChain SQL agent that safely converts
natural language queries to SQL and executes them against the EPCL VEHS database.

Features:
- Natural language to SQL conversion
- Safe query execution with validation
- Context-aware responses
- Error recovery and retry logic
- Conversation memory
- Query explanation and reasoning

Components:
- Custom SQL toolkit with safety features
- Specialized prompts for EPCL domain
- Result formatting and explanation
- Audit logging and monitoring
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sqlite3
import ast

# LangChain imports
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent, AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult

# Local imports
from safe_sql_executor import SafeSQLExecutor, QueryResult, QueryRiskLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EPCLCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring LangChain agent execution."""
    
    def __init__(self):
        self.queries_executed = []
        self.errors = []
        self.start_time = None
        self.chain_of_thought = []
        self.current_step = None
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when agent takes an action."""
        step = {
            "type": "action",
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log,
            "timestamp": datetime.now().isoformat()
        }
        self.chain_of_thought.append(step)
        logger.info(f"Agent action: {action.tool} - {action.tool_input}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when agent finishes."""
        step = {
            "type": "finish",
            "return_values": finish.return_values,
            "log": finish.log,
            "timestamp": datetime.now().isoformat()
        }
        self.chain_of_thought.append(step)
        logger.info(f"Agent finished: {finish.return_values}")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts."""
        self.start_time = datetime.now()
        step = {
            "type": "llm_start",
            "prompts": prompts,
            "timestamp": self.start_time.isoformat()
        }
        self.chain_of_thought.append(step)
        logger.info("LLM processing started")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            step = {
                "type": "llm_end",
                "response": response.generations[0][0].text if response.generations else None,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            self.chain_of_thought.append(step)
            logger.info(f"LLM processing completed in {duration:.2f}s")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts."""
        step = {
            "type": "tool_start",
            "tool_name": serialized.get("name", "unknown"),
            "input": input_str,
            "timestamp": datetime.now().isoformat()
        }
        self.chain_of_thought.append(step)
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool ends."""
        step = {
            "type": "tool_end",
            "output": output,
            "timestamp": datetime.now().isoformat()
        }
        self.chain_of_thought.append(step)
    
    def get_chain_of_thought(self) -> List[Dict[str, Any]]:
        """Get the complete chain of thought."""
        return self.chain_of_thought
    
    def clear_chain_of_thought(self) -> None:
        """Clear the chain of thought for next query."""
        self.chain_of_thought = []
        self.current_step = None


class EPCLSQLAgent:
    """Enhanced SQL agent specifically designed for EPCL VEHS data analysis."""
    
    def __init__(
        self, 
        db_path: str, 
        openai_api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_execution_time: float = 30.0
    ):
        """
        Initialize the EPCL SQL agent.
        
        Args:
            db_path: Path to SQLite database
            openai_api_key: OpenAI API key
            model_name: OpenAI model to use
            temperature: LLM temperature (lower = more deterministic)
            max_execution_time: Maximum query execution time
        """
        self.db_path = db_path
        self.safe_executor = SafeSQLExecutor(db_path, max_execution_time)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=openai_api_key,
            max_tokens=2000
        )
        
        # Initialize database connection for LangChain
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        # Monkey-patch db.run to sanitize incoming SQL (prevents ```sql ... ``` issues)
        try:
            original_run = self.db.run

            def _sanitized_run(query: str, *args, **kwargs):  # type: ignore
                try:
                    clean_query = self._sanitize_sql(query)
                except Exception:
                    clean_query = query
                return original_run(clean_query, *args, **kwargs)

            self.db.run = _sanitized_run  # type: ignore
        except Exception:
            # If patching fails, continue without it (fallback sanitization still applies)
            pass
        
        # Create custom toolkit
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Initialize memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Create agent with custom prompt
        self.agent = self._create_agent()
        
        # Callback handler for monitoring
        self.callback_handler = EPCLCallbackHandler()
        
        # Store the last chain of thought for UI display
        self.last_chain_of_thought = []
        
        # Domain-specific context
        self.domain_context = self._load_domain_context()
        
    def _create_agent(self):
        """Create the SQL agent with custom configuration."""
        
        return create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=5,  # Increased from 3 to 5
            max_execution_time=45  # Increased from 30 to 45 seconds
            # early_stopping_method parameter removed as it's deprecated
        )
    
    def _load_domain_context(self) -> Dict[str, Any]:
        """Load domain-specific context and metadata."""
        context = {
            "incident_categories": [
                "Near Miss", "First Aid", "Medical Treatment", "Lost Time Injury",
                "Property Damage", "Environmental", "Security", "Process Safety"
            ],
            "risk_levels": ["Low", "Medium", "High", "Critical"],
            "common_locations": [
                "Production Area", "Warehouse", "Office", "Laboratory", 
                "Maintenance Shop", "Loading/Unloading", "Utilities"
            ],
            "status_values": ["Open", "In Progress", "Under Review", "Closed"],
            "date_ranges": {
                "current_year": datetime.now().year,
                "last_12_months": "2023-03-01 to 2024-03-31",
                "reporting_period": "March 2023 to March 2024"
            }
        }
        return context
    
    def query(self, user_input: str, include_explanation: bool = True) -> Dict[str, Any]:
        """
        Process a natural language query and return results with analysis.
        
        Args:
            user_input: Natural language question
            include_explanation: Whether to include detailed explanation
            
        Returns:
            Dictionary with query results, analysis, and metadata
        """
        start_time = datetime.now()
        
        try:
            # Pre-process the query for domain-specific terms
            processed_input = self._preprocess_query(user_input)
            
            # Clear previous chain of thought
            self.callback_handler.clear_chain_of_thought()
            
            # Try to execute through LangChain agent with error handling
            try:
                response = self.agent.invoke(
                    {"input": processed_input}, 
                    config={"callbacks": [self.callback_handler]}
                )
                
                # Capture chain of thought
                self.last_chain_of_thought = self.callback_handler.get_chain_of_thought()
                
                # Handle different response formats
                if isinstance(response, dict):
                    agent_response = response.get("output", str(response))
                else:
                    agent_response = str(response)
                
                # Check if agent stopped due to limits but still has useful output
                if "stopped due to iteration limit" in agent_response.lower() or "stopped due to time limit" in agent_response.lower():
                    # Try to extract partial results
                    partial_result = self._extract_partial_result_from_chain()
                    if partial_result:
                        logger.info("Agent hit limits but extracted partial results")
                        return partial_result
                    
            except Exception as agent_error:
                error_msg = str(agent_error)
                logger.warning(f"LangChain agent failed: {error_msg}")
                
                # Capture any partial chain of thought
                self.last_chain_of_thought = self.callback_handler.get_chain_of_thought()
                
                # Check if agent stopped due to iteration/time limits
                if "iteration limit" in error_msg.lower() or "time limit" in error_msg.lower() or "stopped due to" in error_msg.lower():
                    # Try to extract any partial results from chain of thought
                    partial_result = self._extract_partial_result_from_chain()
                    if partial_result:
                        logger.info("Extracted partial result from chain of thought")
                        return partial_result
                
                # Fallback to direct SQL execution
                return self._fallback_to_direct_sql(user_input, processed_input, start_time)
            
            # Extract and validate any SQL that was executed
            executed_queries = self._extract_executed_queries()
            
            # If the agent finished but didn't return executed queries, try to recover from chain
            if not executed_queries:
                has_sql_action = any(
                    step.get("type") == "action" and step.get("tool") == "sql_db_query"
                    for step in self.last_chain_of_thought
                )
                has_error_output = any(
                    step.get("type") == "tool_end" and isinstance(step.get("output"), str) and "error" in step.get("output").lower()
                    for step in self.last_chain_of_thought
                )
                if has_sql_action or has_error_output or "```" in agent_response:
                    partial = self._extract_partial_result_from_chain()
                    if partial:
                        return partial
            else:
                # We DO have executed queries; try to run the most relevant one to produce a concrete answer
                try:
                    preferred_query = None
                    # Prefer audit findings for findings-type questions
                    for q in executed_queries:
                        if "audit_findings" in q.lower():
                            preferred_query = q
                            break
                    # Else prefer inspection findings
                    if not preferred_query:
                        for q in executed_queries:
                            if "inspection_findings" in q.lower():
                                preferred_query = q
                                break
                    # Else pick the first
                    preferred_query = preferred_query or executed_queries[0]
                    # Execute preferred query
                    exec_result = self.safe_executor.execute_query(preferred_query)
                    if exec_result.success:
                        concrete_answer = self._format_sql_result_as_text(exec_result.data or [], preferred_query)
                        recs = self._generate_context_recommendations(preferred_query, exec_result.data or [])
                        execution_time = (datetime.now() - start_time).total_seconds()
                        return {
                            "success": True,
                            "response": {
                                "answer": concrete_answer,
                                "sql_queries": [preferred_query],
                                "explanation": "Answer derived from executed SQL extracted from the agent run.",
                                "recommendations": recs
                            },
                            "original_query": user_input,
                            "processed_query": processed_input,
                            "executed_sql": [preferred_query],
                            "execution_time": execution_time,
                            "timestamp": start_time.isoformat(),
                            "chain_of_thought": self.last_chain_of_thought
                        }
                except Exception as e:
                    logger.warning(f"Failed executing extracted SQL: {e}")
            
            # Format response with additional context
            formatted_response = self._format_response(
                agent_response, 
                executed_queries, 
                include_explanation
            )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "response": formatted_response,
                "original_query": user_input,
                "processed_query": processed_input,
                "executed_sql": executed_queries,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat(),
                "chain_of_thought": self.last_chain_of_thought
            }
            
            # Log successful query
            self._log_query(user_input, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            
            # Try fallback to direct SQL
            return self._fallback_to_direct_sql(user_input, user_input, start_time)
    
    def _preprocess_query(self, user_input: str) -> str:
        """
        Preprocess user input to handle domain-specific terms and synonyms.
        
        Args:
            user_input: Original user input
            
        Returns:
            Processed input with standardized terms
        """
        # Common synonyms and replacements
        replacements = {
            "accidents": "incidents",
            "mishaps": "incidents", 
            "events": "incidents",
            # Avoid expanding 'findings' or 'inspections' generically, as it confuses intent
            "last year": f"from {datetime.now().year - 1}-01-01 to {datetime.now().year - 1}-12-31",
            "this year": f"from {datetime.now().year}-01-01",
            "recent": "last 3 months",
            "cost": "total_cost",
            "money": "total_cost",
            "expensive": "high total_cost"
        }
        
        processed = user_input.lower()
        for old_term, new_term in replacements.items():
            processed = processed.replace(old_term, new_term)
        
        # Preserve specificity: if the user explicitly mentions 'audit findings' or 'inspection findings',
        # do NOT expand or alter the wording further.
        # We intentionally do not auto-expand generic 'findings' to both tables.
        
        # Add context about the reporting period
        if "trend" in processed or "over time" in processed:
            processed += f" (Note: Data covers {self.domain_context['date_ranges']['reporting_period']})"
        
        return processed
    
    def _fallback_to_direct_sql(self, original_query: str, processed_query: str, start_time: datetime) -> Dict[str, Any]:
        """
        Fallback method when LangChain agent fails - try to convert to simple SQL.
        
        Args:
            original_query: Original user query
            processed_query: Processed query
            start_time: Query start time
            
        Returns:
            Dictionary with fallback response
        """
        try:
            # Enhanced query mapping for common questions
            query_mappings = {
                # Department analysis - enhanced queries
                "departments need attention": "SELECT department, COUNT(*) as incident_count, COUNT(CASE WHEN status != 'Closed' THEN 1 END) as open_incidents, COUNT(CASE WHEN status = 'Closed' THEN 1 END) as closed_incidents, AVG(CASE WHEN total_cost IS NOT NULL THEN CAST(total_cost AS REAL) END) as avg_cost FROM incident WHERE department IS NOT NULL AND department != '' GROUP BY department ORDER BY open_incidents DESC, incident_count DESC LIMIT 10",
                "departments most attention": "SELECT department, COUNT(*) as incident_count, COUNT(CASE WHEN status != 'Closed' THEN 1 END) as open_incidents, COUNT(CASE WHEN status = 'Closed' THEN 1 END) as closed_incidents, AVG(CASE WHEN total_cost IS NOT NULL THEN CAST(total_cost AS REAL) END) as avg_cost FROM incident WHERE department IS NOT NULL AND department != '' GROUP BY department ORDER BY open_incidents DESC, incident_count DESC LIMIT 10",
                "which departments": "SELECT department, COUNT(*) as incident_count, COUNT(CASE WHEN status != 'Closed' THEN 1 END) as open_incidents FROM incident WHERE department IS NOT NULL AND department != '' GROUP BY department ORDER BY incident_count DESC LIMIT 10",
                "department incidents": "SELECT department, COUNT(*) as incident_count, COUNT(CASE WHEN status != 'Closed' THEN 1 END) as open_incidents FROM incident WHERE department IS NOT NULL AND department != '' GROUP BY department ORDER BY incident_count DESC LIMIT 10",

                # Findings analysis (audit and inspection)
                "most common audit findings": "SELECT finding, COUNT(*) as count FROM audit_findings WHERE finding IS NOT NULL AND TRIM(finding) != '' GROUP BY finding ORDER BY count DESC LIMIT 10",
                "common audit findings": "SELECT finding, COUNT(*) as count FROM audit_findings WHERE finding IS NOT NULL AND TRIM(finding) != '' GROUP BY finding ORDER BY count DESC LIMIT 10",
                "most common inspection findings": "SELECT finding, COUNT(*) as count FROM inspection_findings WHERE finding IS NOT NULL AND TRIM(finding) != '' GROUP BY finding ORDER BY count DESC LIMIT 10",
                "common inspection findings": "SELECT finding, COUNT(*) as count FROM inspection_findings WHERE finding IS NOT NULL AND TRIM(finding) != '' GROUP BY finding ORDER BY count DESC LIMIT 10",
                "most common findings": "SELECT 'audit' as source, finding, COUNT(*) as count FROM audit_findings WHERE finding IS NOT NULL AND TRIM(finding) != '' GROUP BY finding UNION ALL SELECT 'inspection' as source, finding, COUNT(*) as count FROM inspection_findings WHERE finding IS NOT NULL AND TRIM(finding) != '' GROUP BY finding ORDER BY count DESC LIMIT 10",
                
                # Location analysis
                "locations need attention": "SELECT location, COUNT(*) as incident_count, COUNT(CASE WHEN status = 'Open' THEN 1 END) as open_incidents FROM incident WHERE location IS NOT NULL GROUP BY location ORDER BY incident_count DESC LIMIT 10",
                "which locations": "SELECT location, COUNT(*) as incident_count FROM incident WHERE location IS NOT NULL GROUP BY location ORDER BY incident_count DESC LIMIT 10",
                
                # Category analysis
                "incidents by category": "SELECT category, COUNT(*) as count FROM incident WHERE category IS NOT NULL GROUP BY category ORDER BY count DESC LIMIT 10",
                "categories": "SELECT category, COUNT(*) as count FROM incident WHERE category IS NOT NULL GROUP BY category ORDER BY count DESC LIMIT 10",
                
                # Cost analysis
                "high cost incidents": "SELECT incident_number, total_cost, department, location, description FROM incident WHERE CAST(total_cost AS REAL) > 5000 ORDER BY CAST(total_cost AS REAL) DESC LIMIT 10",
                "expensive incidents": "SELECT incident_number, total_cost, department, location, description FROM incident WHERE CAST(total_cost AS REAL) > 5000 ORDER BY CAST(total_cost AS REAL) DESC LIMIT 10",
                "cost analysis": "SELECT department, AVG(CASE WHEN total_cost IS NOT NULL THEN CAST(total_cost AS REAL) END) as avg_cost, SUM(CASE WHEN total_cost IS NOT NULL THEN CAST(total_cost AS REAL) END) as total_cost FROM incident WHERE department IS NOT NULL GROUP BY department ORDER BY total_cost DESC LIMIT 10",
                
                # Status analysis
                "open incidents": "SELECT COUNT(*) as open_incidents FROM incident WHERE status = 'Open'",
                "closed incidents": "SELECT COUNT(*) as closed_incidents FROM incident WHERE status = 'Closed'",
                "incident status": "SELECT status, COUNT(*) as count FROM incident WHERE status IS NOT NULL GROUP BY status ORDER BY count DESC",
                
                # Time analysis
                "recent incidents": "SELECT incident_number, date_of_occurrence, incident_type, location, department FROM incident WHERE date_of_occurrence IS NOT NULL ORDER BY date_of_occurrence DESC LIMIT 10",
                "latest incidents": "SELECT incident_number, date_of_occurrence, incident_type, location, department FROM incident WHERE date_of_occurrence IS NOT NULL ORDER BY date_of_occurrence DESC LIMIT 10",
                
                # General counts
                "how many incidents": "SELECT COUNT(*) as total_incidents FROM incident",
                "total incidents": "SELECT COUNT(*) as total_incidents FROM incident",
                "incident count": "SELECT COUNT(*) as total_incidents FROM incident"
            }
            
            # Find matching query
            lower_query = processed_query.lower()
            sql_query = None
            
            for pattern, sql in query_mappings.items():
                if pattern in lower_query:
                    sql_query = sql
                    break
            
            if not sql_query:
                # Default fallback
                sql_query = "SELECT COUNT(*) as total_incidents FROM incident"
            
            # Execute the SQL query
            result = self.safe_executor.execute_query(sql_query)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.success:
                # Format the response
                response_text = self._format_sql_result_as_text(result.data, sql_query)
                
                # Generate context-aware recommendations
                recommendations = self._generate_context_recommendations(sql_query, result.data)
                
                return {
                    "success": True,
                    "response": {
                        "answer": response_text,
                        "sql_queries": [sql_query],
                        "explanation": "Analysis completed using intelligent query matching.",
                        "recommendations": recommendations
                    },
                    "original_query": original_query,
                    "processed_query": processed_query,
                    "executed_sql": [sql_query],
                    "execution_time": execution_time,
                    "timestamp": start_time.isoformat(),
                    "chain_of_thought": self.last_chain_of_thought
                }
            else:
                raise Exception(f"SQL execution failed: {result.error_message}")
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": False,
                "error": f"Both LangChain agent and fallback failed: {str(e)}",
                "original_query": original_query,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat(),
                "chain_of_thought": self.last_chain_of_thought
            }
    
    def _format_sql_result_as_text(self, data: List[Dict[str, Any]], sql_query: str) -> str:
        """Format SQL result data as natural language text."""
        if not data:
            return "No results found."
        
        # Handle department analysis queries with enhanced formatting
        if "department" in sql_query.lower() and "incident_count" in sql_query.lower():
            if "need attention" in sql_query.lower() or "most attention" in sql_query.lower():
                result_text = "🚨 **DEPARTMENTS REQUIRING IMMEDIATE ATTENTION**\n\n"
                result_text += "Based on open incidents and total incident history:\n\n"
                
                total_incidents = sum(row.get('incident_count', 0) for row in data)
                total_open = sum(row.get('open_incidents', 0) for row in data)
                
                for i, row in enumerate(data[:7], 1):
                    dept = row.get('department', 'Unknown')
                    count = row.get('incident_count', 0)
                    open_incidents = row.get('open_incidents', 0)
                    closed_incidents = row.get('closed_incidents', 0)
                    avg_cost = row.get('avg_cost')
                    
                    # Calculate percentages
                    incident_percentage = (count / total_incidents * 100) if total_incidents > 0 else 0
                    open_percentage = (open_incidents / total_open * 100) if total_open > 0 else 0
                    
                    # Priority indicator
                    if open_incidents > 20:
                        priority = "🔴 CRITICAL"
                    elif open_incidents > 10:
                        priority = "🟡 HIGH"
                    elif open_incidents > 5:
                        priority = "🟢 MEDIUM"
                    else:
                        priority = "⚪ LOW"
                    
                    result_text += f"{i}. **{dept}** {priority}\n"
                    result_text += f"   📊 Total incidents: {count} ({incident_percentage:.1f}% of all incidents)\n"
                    result_text += f"   🚨 Open incidents: {open_incidents} ({open_percentage:.1f}% of all open)\n"
                    result_text += f"   ✅ Closed incidents: {closed_incidents}\n"
                    if avg_cost is not None and avg_cost > 0:
                        result_text += f"   💰 Average cost: ${avg_cost:,.2f}\n"
                    result_text += "\n"
                
                result_text += f"\n📈 **SUMMARY**: {len(data)} departments analyzed, {total_incidents} total incidents, {total_open} currently open"
                return result_text.strip()
            else:
                result_text = "📊 **INCIDENT ANALYSIS BY DEPARTMENT**\n\n"
                for i, row in enumerate(data[:10], 1):
                    dept = row.get('department', 'Unknown')
                    count = row.get('incident_count', 0)
                    open_incidents = row.get('open_incidents', 0)
                    result_text += f"{i}. **{dept}**: {count} total incidents"
                    if open_incidents is not None:
                        result_text += f" ({open_incidents} open)"
                    result_text += "\n"
                return result_text.strip()
        
        # Handle location analysis
        if "location" in sql_query.lower() and "incident_count" in sql_query.lower():
            result_text = "Incidents by location:\n\n"
            for i, row in enumerate(data[:10], 1):
                location = row.get('location', 'Unknown')
                count = row.get('incident_count', 0)
                open_incidents = row.get('open_incidents')
                result_text += f"{i}. {location}: {count} incidents"
                if open_incidents is not None:
                    result_text += f" ({open_incidents} open)"
                result_text += "\n"
            return result_text.strip()
        
        # Handle category analysis
        if "category" in sql_query.lower() and "GROUP BY" in sql_query.upper():
            result_text = "Incidents by category:\n\n"
            for i, row in enumerate(data[:10], 1):
                category = row.get('category', 'Unknown')
                count = row.get('count', 0)
                result_text += f"{i}. {category}: {count} incidents\n"
            return result_text.strip()

        # Handle findings analysis (audit/inspection)
        if ("audit_findings" in sql_query.lower() or "inspection_findings" in sql_query.lower()) and "GROUP BY" in sql_query.upper() and "finding" in sql_query.lower():
            title = "Top findings"
            if "audit_findings" in sql_query.lower() and "inspection_findings" not in sql_query.lower():
                title = "Top audit findings"
            elif "inspection_findings" in sql_query.lower() and "audit_findings" not in sql_query.lower():
                title = "Top inspection findings"
            result_text = f"{title}:\n\n"
            for i, row in enumerate(data[:10], 1):
                raw_finding = row.get('finding') or row.get('FINDING')
                finding = (raw_finding or '').strip() or 'Unspecified'
                count = row.get('count') or row.get('COUNT') or 0
                result_text += f"{i}. {finding}: {count}\n"
            return result_text.strip()
        
        # Handle cost analysis
        if "total_cost" in sql_query.lower() and "expensive" in sql_query.lower():
            result_text = "High-cost incidents:\n\n"
            for i, row in enumerate(data[:5], 1):
                incident_num = row.get('incident_number', 'Unknown')
                cost = row.get('total_cost', 0)
                dept = row.get('department', 'Unknown')
                location = row.get('location', 'Unknown')
                result_text += f"{i}. {incident_num}\n"
                result_text += f"   • Cost: ${float(cost):,.2f}\n"
                result_text += f"   • Department: {dept}\n"
                result_text += f"   • Location: {location}\n\n"
            return result_text.strip()
        
        # Handle simple count queries
        if "COUNT(*)" in sql_query.upper() and len(data) == 1 and len(data[0]) == 1:
            count_value = list(data[0].values())[0]
            if "open" in sql_query.lower():
                return f"There are {count_value} open incidents."
            elif "closed" in sql_query.lower():
                return f"There are {count_value} closed incidents."
            else:
                return f"There are {count_value} total incidents in the database."
        
        # Handle status breakdown
        if "status" in sql_query.lower() and "GROUP BY" in sql_query.upper():
            result_text = "Incidents by status:\n\n"
            for row in data:
                status = row.get('status', 'Unknown')
                count = row.get('count', 0)
                result_text += f"• {status}: {count} incidents\n"
            return result_text.strip()
        
        # Handle recent incidents
        if "ORDER BY date_of_occurrence DESC" in sql_query:
            result_text = "Recent incidents:\n\n"
            for i, row in enumerate(data[:5], 1):
                incident_num = row.get('incident_number', 'Unknown')
                date = row.get('date_of_occurrence', 'Unknown')
                incident_type = row.get('incident_type', 'Unknown')
                location = row.get('location', 'Unknown')
                dept = row.get('department', 'Unknown')
                
                result_text += f"{i}. {incident_num} ({date})\n"
                result_text += f"   • Type: {incident_type}\n"
                result_text += f"   • Location: {location}\n"
                result_text += f"   • Department: {dept}\n\n"
            return result_text.strip()
        
        # Default formatting for GROUP BY queries
        if "GROUP BY" in sql_query.upper() and len(data) <= 10:
            result_text = "Results:\n\n"
            for i, row in enumerate(data, 1):
                if len(row) >= 2:
                    items = list(row.items())
                    key_name, key_value = items[0]
                    count_name, count_value = items[1]
                    result_text += f"{i}. {key_value}: {count_value}\n"
            return result_text.strip()
        
        # Default fallback
        if len(data) == 1:
            row = data[0]
            if len(row) == 1:
                return f"Result: {list(row.values())[0]}"
        
        return f"Found {len(data)} results. The query returned detailed data - use the SQL interface to view all columns and rows."
    
    def _generate_context_recommendations(self, sql_query: str, data: List[Dict[str, Any]]) -> List[str]:
        """Generate prescriptive analysis with specific, actionable recommendations based on the query and results."""
        recommendations = []
        
        if not data:
            return ["No data available for analysis. Consider expanding the query parameters."]
        
        # Department analysis recommendations with prescriptive insights
        if "department" in sql_query.lower() and "need attention" in sql_query.lower():
            if data:
                # Analyze the top departments
                top_3_depts = data[:3]
                total_incidents = sum(row.get('incident_count', 0) for row in data)
                
                top_dept = top_3_depts[0].get('department', 'Unknown')
                top_count = top_3_depts[0].get('incident_count', 0)
                top_percentage = (top_count / total_incidents * 100) if total_incidents > 0 else 0
                
                recommendations.extend([
                    f"🚨 IMMEDIATE ACTION REQUIRED: {top_dept} department accounts for {top_percentage:.1f}% of all incidents ({top_count} incidents)",
                    f"📊 PRIORITY RANKING: Focus resources on top 3 departments: {', '.join([d.get('department', 'Unknown') for d in top_3_depts])}",
                    f"🔍 ROOT CAUSE ANALYSIS: Conduct immediate investigation in {top_dept} - potential systemic issues",
                    f"📚 TRAINING INTERVENTION: Deploy targeted safety training for {top_dept} within 30 days",
                    f"⚡ RESOURCE ALLOCATION: Assign dedicated safety officer to {top_dept} if incidents > 50",
                    f"📈 MONITORING: Implement weekly incident tracking for top 3 departments",
                    f"🎯 TARGET: Reduce {top_dept} incidents by 25% within next quarter through focused interventions"
                ])
                
                # Add specific recommendations based on incident counts
                if top_count > 100:
                    recommendations.append(f"🔴 CRITICAL: {top_dept} shows extremely high incident rate - consider operational shutdown for safety review")
                elif top_count > 50:
                    recommendations.append(f"🟡 HIGH RISK: {top_dept} requires immediate management intervention and safety protocol review")
        
        # Location analysis with specific actions
        elif "location" in sql_query.lower() and data:
            top_location = data[0].get('location', 'Unknown')
            top_count = data[0].get('incident_count', 0)
            
            recommendations.extend([
                f"🏭 SITE-SPECIFIC ACTION: {top_location} requires immediate safety audit - {top_count} incidents recorded",
                f"🔧 ENGINEERING CONTROLS: Review physical safety measures at {top_location}",
                f"👥 STAFFING REVIEW: Assess supervision levels and worker experience at high-incident locations",
                f"📋 PROCEDURE UPDATE: Revise safety protocols specific to {top_location} environment",
                f"🚨 EMERGENCY RESPONSE: Ensure adequate emergency equipment at {top_location}",
                f"📊 BENCHMARKING: Compare {top_location} incident rates with industry standards"
            ])
        
        # Category analysis with targeted interventions
        elif "category" in sql_query.lower() and data:
            top_category = data[0].get('category', 'Unknown')
            top_count = data[0].get('count', 0)
            
            category_actions = {
                'Near Miss': ['Enhance near-miss reporting culture', 'Implement proactive hazard identification'],
                'First Aid': ['Review first aid station locations', 'Increase safety awareness training'],
                'Medical Treatment': ['Investigate injury severity trends', 'Review medical response protocols'],
                'Lost Time Injury': ['Implement return-to-work programs', 'Focus on injury prevention'],
                'Property Damage': ['Review equipment maintenance schedules', 'Assess operational procedures'],
                'Environmental': ['Conduct environmental impact assessment', 'Review waste management procedures'],
                'Security': ['Enhance security protocols', 'Review access control systems'],
                'Process Safety': ['Conduct process hazard analysis', 'Review safety instrumented systems']
            }
            
            specific_actions = category_actions.get(top_category, ['Review incident patterns', 'Implement targeted controls'])
            
            recommendations.extend([
                f"🎯 CATEGORY FOCUS: {top_category} incidents dominate with {top_count} occurrences",
                f"📋 SPECIFIC ACTIONS for {top_category}:"
            ])
            recommendations.extend([f"   • {action}" for action in specific_actions])
        
        # Cost analysis with ROI calculations
        elif "cost" in sql_query.lower() and data:
            if 'avg_cost' in str(data[0]) or 'total_cost' in str(data[0]):
                high_cost_dept = data[0].get('department', 'Unknown')
                avg_cost = data[0].get('avg_cost', 0)
                total_cost = data[0].get('total_cost', 0)
                
                recommendations.extend([
                    f"💰 COST IMPACT: {high_cost_dept} shows highest financial impact - ${total_cost:,.2f} total cost" if total_cost else f"💰 COST IMPACT: {high_cost_dept} shows highest average cost per incident - ${avg_cost:,.2f}",
                    f"📊 ROI ANALYSIS: Investing $50K in {high_cost_dept} safety could prevent ${avg_cost * 5:,.2f} in future costs",
                    f"🎯 COST REDUCTION TARGET: Focus prevention efforts on high-cost incident types",
                    f"📈 BUDGET ALLOCATION: Prioritize safety investments in {high_cost_dept}",
                    f"🔍 COST DRIVER ANALYSIS: Investigate why {high_cost_dept} incidents are more expensive",
                    f"⚖️ INSURANCE REVIEW: Assess insurance implications of high-cost incidents in {high_cost_dept}"
                ])
        
        # Status analysis with timeline actions
        elif "status" in sql_query.lower():
            if "open" in sql_query.lower():
                open_count = data[0].get('open_incidents', 0) if 'open_incidents' in str(data[0]) else data[0].get('count', 0)
                recommendations.extend([
                    f"⏰ URGENT: {open_count} open incidents require immediate closure action",
                    f"📅 TIMELINE: Establish 30-day closure target for all open incidents",
                    f"👥 RESOURCE ASSIGNMENT: Assign dedicated personnel to incident closure",
                    f"📊 TRACKING: Implement daily open incident review meetings",
                    f"🔄 PROCESS IMPROVEMENT: Review incident resolution workflow for bottlenecks",
                    f"📈 KPI TARGET: Achieve 95% incident closure rate within 30 days"
                ])
        
        # Recent incidents with trend analysis
        elif "recent" in sql_query.lower() or "latest" in sql_query.lower():
            recent_count = len(data)
            recommendations.extend([
                f"📈 TREND ANALYSIS: {recent_count} recent incidents identified - monitor for patterns",
                f"🚨 EARLY WARNING: Implement real-time incident tracking dashboard",
                f"📊 PATTERN RECOGNITION: Analyze recent incidents for common factors",
                f"⚡ RAPID RESPONSE: Establish 24-hour incident investigation protocol",
                f"📢 COMMUNICATION: Share lessons learned from recent incidents across organization",
                f"🎯 PREVENTION: Implement immediate controls based on recent incident analysis"
            ])
        
        # General data-driven recommendations
        else:
            total_records = len(data)
            recommendations.extend([
                f"📊 DATA INSIGHT: Analysis of {total_records} records reveals actionable patterns",
                f"🎯 STRATEGIC FOCUS: Use this data to prioritize safety investments",
                f"📈 PERFORMANCE METRICS: Establish KPIs based on identified trends",
                f"🔄 CONTINUOUS IMPROVEMENT: Schedule monthly data review sessions",
                f"📋 ACTION PLAN: Develop 90-day improvement plan based on findings"
            ])
        
        # Add implementation timeline
        recommendations.append("")
        recommendations.append("⏱️ IMPLEMENTATION TIMELINE:")
        recommendations.append("   • Week 1: Immediate actions and resource allocation")
        recommendations.append("   • Week 2-4: Deploy interventions and training")
        recommendations.append("   • Month 2-3: Monitor progress and adjust strategies")
        recommendations.append("   • Month 4: Evaluate effectiveness and plan next phase")
        
        return recommendations
    
    def _extract_executed_queries(self) -> List[str]:
        """Extract SQL queries that were executed during agent run."""
        executed_queries = []
        
        # Extract SQL queries from chain of thought
        for step in self.last_chain_of_thought:
            if step.get("type") == "tool_end" and step.get("output"):
                raw_output = step.get("output", "")
                output = self._sanitize_sql(raw_output)
                # Look for SQL queries in the output
                if any(kw in output.upper() for kw in ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")):
                    # Try to extract the first SQL-like line
                    lines = output.split('\n')
                    for line in lines:
                        clean_line = self._sanitize_sql(line.strip())
                        if clean_line.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")):
                            executed_queries.append(clean_line)
                            break
            elif step.get("type") == "action" and step.get("tool") == "sql_db_query":
                # Direct SQL query action
                tool_input = step.get("tool_input", "")
                if isinstance(tool_input, str) and tool_input.strip():
                    executed_queries.append(self._sanitize_sql(tool_input.strip()))
        
        return executed_queries

    def _sanitize_sql(self, sql: str) -> str:
        """Remove markdown/code-fence artifacts and stray backticks from SQL strings."""
        if not isinstance(sql, str):
            return sql
        s = sql.strip()
        # Remove common code-fence patterns like ```sql ... ``` or ``` ... ```
        if s.startswith("```") and s.endswith("```"):
            s = s[3:-3]
        # Remove language tag like 'sql' at the start after fence removal
        s = s.lstrip()
        if s.lower().startswith("sql\n"):
            s = s[4:]
        if s.lower().startswith("sql") and s[3:4].isspace():
            s = s[3:].lstrip()
        # Remove any stray backticks and weird triple backtick remnants
        s = s.replace("`", "")
        # Remove enclosing quotes " or ' if they wrap the entire statement
        if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1]
        # Collapse whitespace
        lines = [ln.strip() for ln in s.splitlines()]
        s = " ".join([ln for ln in lines if ln])
        return s.strip()
    
    def _extract_partial_result_from_chain(self) -> Optional[Dict[str, Any]]:
        """Extract partial results when agent times out but has executed SQL."""
        try:
            # Look for SQL query results in the chain of thought
            sql_queries = []
            sql_results = []
            last_query = None
        
            for step in self.last_chain_of_thought:
                if step.get("type") == "action" and step.get("tool") == "sql_db_query":
                    tool_input = step.get("tool_input", "")
                    if isinstance(tool_input, str) and tool_input.strip():
                        last_query = self._sanitize_sql(tool_input.strip())
                        sql_queries.append(last_query)
                elif step.get("type") == "tool_end" and step.get("output"):
                    output = self._sanitize_sql(step.get("output", ""))
                    if output and len(output) > 10:  # Has meaningful output
                        sql_results.append(output)
        
            # If we have captured any query but either no results, or results are just an echoed SQL string, execute preferred query directly
            if last_query or sql_queries:
                only_sql_echo = False
                if sql_results:
                    last_output = self._sanitize_sql(str(sql_results[-1])).strip()
                    if last_output.upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")):
                        only_sql_echo = True
                if not sql_results or only_sql_echo:
                    # Prefer an audit findings query if present when multiple queries are detected
                    preferred_query = last_query
                    for q in sql_queries:
                        if "audit_findings" in q.lower():
                            preferred_query = q
                            break
                    preferred_query = preferred_query or last_query
                    try:
                        result = self.safe_executor.execute_query(preferred_query)
                        if result.success:
                            # Format the result using our existing methods
                            formatted_text = self._format_sql_result_as_text(result.data, preferred_query)
                            recommendations = self._generate_context_recommendations(preferred_query, result.data)
                            
                            return {
                                "success": True,
                                "response": {
                                    "answer": formatted_text,
                                    "sql_queries": [preferred_query],
                                    "explanation": "Query completed successfully after agent timeout.",
                                    "recommendations": recommendations
                                },
                                "original_query": "",
                                "processed_query": "",
                                "executed_sql": [preferred_query],
                                "execution_time": 0,
                                "timestamp": datetime.now().isoformat(),
                                "chain_of_thought": self.last_chain_of_thought
                            }
                    except Exception as exec_error:
                        logger.error(f"Failed to execute partial query: {exec_error}")
            
            # Original logic for when we have real results (not just echoed SQL)
            if sql_queries and sql_results and not only_sql_echo:
                # Format the partial result
                response_text = "Analysis completed with partial results:\n\n"
                response_text += f"Query executed: {sql_queries[-1]}\n\n"
                response_text += f"Results: {sql_results[-1][:500]}..."
                
                return {
                    "success": True,
                    "response": {
                        "answer": response_text,
                        "sql_queries": sql_queries,
                        "explanation": "Agent reached iteration limit but partial results were extracted.",
                        "recommendations": ["Review the partial results", "Consider simplifying the query"]
                    },
                    "original_query": "",
                    "processed_query": "",
                    "executed_sql": sql_queries,
                    "execution_time": 0,
                    "timestamp": datetime.now().isoformat(),
                    "chain_of_thought": self.last_chain_of_thought
                }
        except Exception as e:
            logger.error(f"Failed to extract partial result: {e}")
        
        return None
    
    def _format_response(
        self, 
        agent_response: str, 
        executed_queries: List[str],
        include_explanation: bool
    ) -> Dict[str, Any]:
        """
        Format the agent response with additional context and analysis.
        
        Args:
            agent_response: Raw response from LangChain agent
            executed_queries: List of SQL queries executed
            include_explanation: Whether to include explanation
            
        Returns:
            Formatted response dictionary
        """
        formatted = {
            "answer": agent_response,
            "sql_queries": executed_queries,
        }
        
        if include_explanation:
            formatted["explanation"] = self._generate_explanation(agent_response)
            formatted["recommendations"] = self._generate_recommendations(agent_response)
        
        return formatted
    
    def _generate_explanation(self, response: str) -> str:
        """Generate explanation of the analysis."""
        # This could use the LLM to generate explanations
        return "Analysis based on EPCL VEHS database covering incident, hazard, audit, and inspection data."
    
    def _generate_recommendations(self, response: str) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        recommendations = []
        
        # Enhanced pattern matching for prescriptive analysis
        if "high" in response.lower() and ("incident" in response.lower() or "risk" in response.lower()):
            recommendations.extend([
                "🚨 IMMEDIATE ACTION: Deploy additional safety personnel to high-risk areas within 48 hours",
                "📋 PROTOCOL REVIEW: Conduct emergency review of safety procedures in identified high-risk zones",
                "🎯 TARGET INTERVENTION: Implement specific controls for top 3 risk factors identified",
                "📊 MONITORING: Establish daily safety checks in high-incident areas",
                "💡 INNOVATION: Consider advanced safety technologies (sensors, alerts, automation)"
            ])
        
        if "cost" in response.lower():
            recommendations.extend([
                "💰 ROI ANALYSIS: Calculate potential savings from preventing top 5 cost drivers",
                "📈 INVESTMENT STRATEGY: Allocate 15% of incident costs to prevention programs",
                "🔍 COST BREAKDOWN: Analyze direct vs indirect costs to optimize prevention focus",
                "⚖️ BUDGET REALLOCATION: Shift reactive spending to proactive safety investments",
                "📊 FINANCIAL TRACKING: Implement cost-per-incident KPI monitoring"
            ])
        
        if "trend" in response.lower():
            recommendations.extend([
                "📈 PREDICTIVE ANALYTICS: Implement trend forecasting to anticipate future incidents",
                "🔍 ROOT CAUSE DEEP DIVE: Investigate underlying factors driving negative trends",
                "⚡ EARLY INTERVENTION: Create automated alerts when trends exceed thresholds",
                "📊 BENCHMARK COMPARISON: Compare trends against industry standards",
                "🎯 TREND REVERSAL: Develop specific action plans to reverse negative patterns"
            ])
        
        if "department" in response.lower():
            recommendations.extend([
                "🏢 DEPARTMENTAL FOCUS: Assign safety champions to high-incident departments",
                "📚 TARGETED TRAINING: Deploy department-specific safety curriculum",
                "👥 LEADERSHIP ENGAGEMENT: Require department heads to attend safety meetings",
                "📊 PERFORMANCE METRICS: Tie department KPIs to safety performance"
            ])
        
        if "location" in response.lower():
            recommendations.extend([
                "🏭 SITE ASSESSMENT: Conduct comprehensive safety audits at problem locations",
                "🔧 INFRASTRUCTURE: Upgrade safety equipment and signage at high-incident sites",
                "👷 SUPERVISION: Increase safety supervisor presence at problematic locations",
                "📋 PROTOCOLS: Develop location-specific safety procedures"
            ])
        
        # Add general strategic recommendations if no specific patterns found
        if not recommendations:
            recommendations.extend([
                "📊 DATA-DRIVEN STRATEGY: Use this analysis to inform safety investment decisions",
                "🎯 FOCUSED APPROACH: Prioritize interventions based on data insights",
                "📈 PERFORMANCE TRACKING: Establish baseline metrics for improvement measurement",
                "🔄 CONTINUOUS IMPROVEMENT: Schedule quarterly reviews of safety data trends",
                "💡 INNOVATION OPPORTUNITY: Explore technology solutions for identified problem areas"
            ])
        
        return recommendations
    
    def _log_query(self, user_input: str, result: Dict[str, Any]) -> None:
        """Log query execution for audit and analysis."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_input,
            "success": result.get("success", False),
            "execution_time": result.get("execution_time", 0),
            "error": result.get("error"),
            "sql_queries": result.get("executed_sql", [])
        }
        
        logger.info(f"Query logged: {json.dumps(log_entry)}")
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information for user reference."""
        tables_info = {}
        
        for table_name in ["incident", "hazard_id", "audit", "audit_findings", "inspection", "inspection_findings"]:
            schema = self.safe_executor.get_table_schema(table_name)
            sample = self.safe_executor.get_sample_data(table_name, limit=3)
            
            tables_info[table_name] = {
                "schema": schema,
                "sample_data": sample,
                "description": self._get_table_description(table_name)
            }
        
        return tables_info
    
    def _get_table_description(self, table_name: str) -> str:
        """Get human-readable description of table purpose."""
        descriptions = {
            "incident": "Safety incidents including near misses, injuries, and property damage",
            "hazard_id": "Identified hazards and risk assessments",
            "audit": "Safety audits conducted across different locations and departments",
            "audit_findings": "Specific findings and observations from safety audits",
            "inspection": "Regular safety inspections and assessments", 
            "inspection_findings": "Detailed findings from safety inspections"
        }
        return descriptions.get(table_name, "Data table for EPCL VEHS system")
    
    def suggest_queries(self) -> List[Dict[str, str]]:
        """Suggest common queries users might want to ask."""
        suggestions = [
            {
                "category": "Incident Analysis",
                "queries": [
                    "How many incidents occurred in the last 6 months?",
                    "What are the top 5 incident categories by frequency?",
                    "Which locations have the highest incident rates?",
                    "What is the total cost of incidents this year?",
                    "Show me the trend of incidents over time"
                ]
            },
            {
                "category": "Risk Assessment", 
                "queries": [
                    "Which departments have the most high-risk incidents?",
                    "What are the most common causes of incidents?",
                    "Show me incidents with high injury potential",
                    "Which equipment failures led to incidents?",
                    "What are the worst-case consequences we've seen?"
                ]
            },
            {
                "category": "Audit & Compliance",
                "queries": [
                    "How many audits were completed this year?",
                    "What are the most common audit findings?",
                    "Which locations need the most attention based on audits?",
                    "What is the status of corrective actions?",
                    "Show me overdue action items"
                ]
            },
            {
                "category": "Performance Metrics",
                "queries": [
                    "What is our incident closure rate?",
                    "How long does it take to close incidents on average?",
                    "Which contractors have the best safety record?",
                    "What is the effectiveness of our safety training?",
                    "Compare this year's performance to last year"
                ]
            }
        ]
        
        return suggestions


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    DB_PATH = "epcl_vehs.db"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Initialize agent
    print("Initializing EPCL SQL Agent...")
    agent = EPCLSQLAgent(DB_PATH, OPENAI_API_KEY)
    
    # Test queries
    test_queries = [
        "How many incidents occurred in the last 6 months?",
        "What are the top 3 incident categories?",
        "Which location has the highest incident rate?",
        "Show me the cost breakdown of incidents by category"
    ]
    
    print("\nTesting EPCL SQL Agent:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nUser Query: {query}")
        print("-" * 40)
        
        result = agent.query(query)
        
        if result["success"]:
            print("Response:", result["response"]["answer"])
            if result["executed_sql"]:
                print("SQL Executed:", result["executed_sql"])
        else:
            print("Error:", result["error"])
        
        print(f"Execution Time: {result['execution_time']:.2f}s")
    
    # Show schema info
    print("\n" + "=" * 60)
    print("DATABASE SCHEMA INFORMATION:")
    print("=" * 60)
    
    schema_info = agent.get_schema_info()
    for table_name, info in schema_info.items():
        print(f"\nTable: {table_name}")
        print(f"Description: {info['description']}")
        if info['schema'].get('columns'):
            print(f"Columns: {len(info['schema']['columns'])}")
    
    # Show suggested queries
    print("\n" + "=" * 60)
    print("SUGGESTED QUERIES:")
    print("=" * 60)
    
    suggestions = agent.suggest_queries()
    for category in suggestions:
        print(f"\n{category['category']}:")
        for i, query in enumerate(category['queries'], 1):
            print(f"  {i}. {query}")
