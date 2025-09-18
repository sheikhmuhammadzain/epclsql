#!/usr/bin/env python3
"""
Test script to verify the LangChain SQL Agent fixes
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_sql_agent import EPCLSQLAgent

def test_agent_initialization():
    """Test that the agent can be initialized without errors."""
    print("Testing agent initialization...")
    
    # Mock API key for testing
    api_key = os.getenv("OPENAI_API_KEY", "test-key")
    db_path = "epcl_vehs.db"
    
    try:
        agent = EPCLSQLAgent(
            db_path=db_path,
            openai_api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0.1
        )
        print("✅ Agent initialized successfully")
        return agent
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return None

def test_callback_handler(agent):
    """Test that the callback handler works correctly."""
    print("\nTesting callback handler...")
    
    try:
        # Test callback handler methods
        agent.callback_handler.clear_chain_of_thought()
        chain = agent.callback_handler.get_chain_of_thought()
        
        if isinstance(chain, list):
            print("✅ Callback handler working correctly")
            return True
        else:
            print("❌ Callback handler not returning list")
            return False
    except Exception as e:
        print(f"❌ Callback handler test failed: {e}")
        return False

def test_query_processing(agent):
    """Test query processing without actually calling OpenAI."""
    print("\nTesting query processing...")
    
    try:
        # Test preprocessing
        processed = agent._preprocess_query("How many accidents happened last year?")
        print(f"✅ Query preprocessing works: '{processed}'")
        
        # Test domain context loading
        context = agent._load_domain_context()
        if isinstance(context, dict) and "incident_categories" in context:
            print("✅ Domain context loaded successfully")
        else:
            print("❌ Domain context not loaded properly")
            
        return True
    except Exception as e:
        print(f"❌ Query processing test failed: {e}")
        return False

def test_sql_extraction():
    """Test SQL query extraction from chain of thought."""
    print("\nTesting SQL extraction...")
    
    try:
        # Create a mock agent
        api_key = os.getenv("OPENAI_API_KEY", "test-key")
        agent = EPCLSQLAgent("epcl_vehs.db", api_key)
        
        # Mock chain of thought with SQL
        mock_chain = [
            {
                "type": "action",
                "tool": "sql_db_query",
                "tool_input": "SELECT COUNT(*) FROM incident",
                "timestamp": datetime.now().isoformat()
            },
            {
                "type": "tool_end",
                "output": "[(42,)]",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        agent.last_chain_of_thought = mock_chain
        queries = agent._extract_executed_queries()
        
        if queries and "SELECT COUNT(*) FROM incident" in queries:
            print("✅ SQL extraction working correctly")
            return True
        else:
            print(f"❌ SQL extraction failed: {queries}")
            return False
            
    except Exception as e:
        print(f"❌ SQL extraction test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running LangChain SQL Agent Fix Tests")
    print("=" * 50)
    
    # Test 1: Agent initialization
    agent = test_agent_initialization()
    if not agent:
        print("\n❌ Cannot proceed with other tests - agent initialization failed")
        return False
    
    # Test 2: Callback handler
    callback_ok = test_callback_handler(agent)
    
    # Test 3: Query processing
    query_ok = test_query_processing(agent)
    
    # Test 4: SQL extraction
    sql_ok = test_sql_extraction()
    
    # Summary
    print("\n" + "=" * 50)
    print("🧪 Test Summary:")
    print(f"Agent Initialization: {'✅' if agent else '❌'}")
    print(f"Callback Handler: {'✅' if callback_ok else '❌'}")
    print(f"Query Processing: {'✅' if query_ok else '❌'}")
    print(f"SQL Extraction: {'✅' if sql_ok else '❌'}")
    
    all_passed = agent and callback_ok and query_ok and sql_ok
    
    if all_passed:
        print("\n🎉 All tests passed! The fixes are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please review the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()
