#!/usr/bin/env python3
"""
Quick test script to verify the LangChain agent fix
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_agent_initialization():
    """Test if the LangChain agent can be initialized without errors."""
    
    print("Testing LangChain SQL Agent initialization...")
    
    # Check if required files exist
    db_path = project_root / "epcl_vehs.db"
    if not db_path.exists():
        print("❌ Database not found. Please run: python ingest_excel_to_sqlite.py")
        return False
    
    # Check if OpenAI API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("❌ OpenAI API key not configured. Please set OPENAI_API_KEY in .env file")
        return False
    
    try:
        # Import and test agent initialization
        from langchain_sql_agent import EPCLSQLAgent
        
        print("🔄 Initializing SQL Agent...")
        agent = EPCLSQLAgent(
            db_path=str(db_path),
            openai_api_key=openai_key,
            model_name="gpt-4o-mini",
            temperature=0.1
        )
        
        print("✅ SQL Agent initialized successfully!")
        
        # Test basic functionality
        print("🔄 Testing schema retrieval...")
        schema_info = agent.get_schema_info()
        
        if schema_info:
            print(f"✅ Schema retrieved for {len(schema_info)} tables")
            return True
        else:
            print("❌ Schema retrieval failed")
            return False
            
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

def test_safe_executor():
    """Test if the safe SQL executor works."""
    
    print("\nTesting Safe SQL Executor...")
    
    try:
        from safe_sql_executor import SafeSQLExecutor
        
        db_path = project_root / "epcl_vehs.db"
        if not db_path.exists():
            print("❌ Database not found")
            return False
        
        executor = SafeSQLExecutor(str(db_path))
        
        # Test a simple query
        result = executor.execute_query("SELECT COUNT(*) as total FROM incident LIMIT 1")
        
        if result.success:
            print("✅ Safe SQL Executor working correctly")
            print(f"   Query result: {result.data}")
            return True
        else:
            print(f"❌ SQL Executor failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Safe SQL Executor test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("=" * 60)
    print("EPCL VEHS System Component Test")
    print("=" * 60)
    
    # Test 1: Safe SQL Executor
    executor_ok = test_safe_executor()
    
    # Test 2: LangChain Agent
    agent_ok = test_agent_initialization()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if executor_ok and agent_ok:
        print("🎉 All tests passed! The system should work correctly now.")
        print("\nYou can now run:")
        print("  python run_system.py --dev")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
