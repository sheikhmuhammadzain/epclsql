#!/usr/bin/env python3
"""
EPCL VEHS System Runner

This script provides a complete workflow to set up and run the EPCL VEHS SQL Agent system.
It handles data ingestion, system initialization, and service startup.

Usage:
    python run_system.py [--ingest] [--test] [--dev] [--production]

Options:
    --ingest: Run data ingestion from Excel to SQLite
    --test: Run test suite before starting
    --dev: Run in development mode
    --production: Run in production mode with Docker
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EPCLSystemRunner:
    """Main system runner for EPCL VEHS SQL Agent."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.excel_file = self.project_root / "EPCL VEHS Data (Mar23 - Mar24).xlsx"
        self.db_file = self.project_root / "epcl_vehs.db"
        self.env_file = self.project_root / ".env"
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        logger.info("Checking prerequisites...")
        
        # Check if Excel file exists
        if not self.excel_file.exists():
            logger.error(f"Excel file not found: {self.excel_file}")
            logger.info("Please ensure the Excel file is in the project directory")
            return False
        
        # Check if .env file exists
        if not self.env_file.exists():
            logger.warning(f".env file not found. Please copy .env.example to .env and configure it.")
            logger.info("Creating basic .env file...")
            self._create_basic_env_file()
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        logger.info("✅ Prerequisites check passed")
        return True
    
    def _create_basic_env_file(self):
        """Create a basic .env file if it doesn't exist."""
        basic_env_content = """# EPCL VEHS Configuration
OPENAI_API_KEY=your_openai_api_key_here
EPCL_API_KEY=epcl-demo-key-2024
DATABASE_PATH=epcl_vehs.db
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO
MAX_QUERIES_PER_MINUTE=10
MAX_QUERIES_PER_HOUR=100
"""
        with open(self.env_file, 'w') as f:
            f.write(basic_env_content)
        
        logger.info(f"Created basic .env file at {self.env_file}")
        logger.warning("Please update the OPENAI_API_KEY in the .env file before running the system")
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        logger.info("Installing Python dependencies...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, cwd=self.project_root)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def run_ingestion(self) -> bool:
        """Run data ingestion from Excel to SQLite."""
        logger.info("Starting data ingestion...")
        
        try:
            # Remove existing database if it exists
            if self.db_file.exists():
                logger.info("Removing existing database...")
                self.db_file.unlink()
            
            # Run ingestion script
            subprocess.run([
                sys.executable, "ingest_excel_to_sqlite.py"
            ], check=True, cwd=self.project_root)
            
            logger.info("✅ Data ingestion completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Data ingestion failed: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run the test suite."""
        logger.info("Running test suite...")
        
        try:
            result = subprocess.run([
                sys.executable, "test_suite.py"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ All tests passed")
                return True
            else:
                logger.error("❌ Some tests failed")
                logger.error(result.stdout)
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
    
    def check_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        from dotenv import load_dotenv
        load_dotenv(self.env_file)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            logger.error("OpenAI API key not configured in .env file")
            logger.info("Please set OPENAI_API_KEY in the .env file")
            return False
        
        logger.info("✅ OpenAI API key configured")
        return True
    
    def run_development_server(self, use_simple=False):
        """Run the development server."""
        app_module = "main_simple:app" if use_simple else "main:app"
        server_type = "simplified" if use_simple else "full"
        
        logger.info(f"Starting {server_type} development server...")
        logger.info("Server will be available at http://127.0.0.1:8000")
        logger.info("Press Ctrl+C to stop the server")
        
        try:
            subprocess.run([
                sys.executable, "-m", "uvicorn", app_module, 
                "--host", "127.0.0.1", 
                "--port", "8000", 
                "--reload",
                "--log-level", "info"
            ], cwd=self.project_root)
            
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server failed to start: {e}")
            if not use_simple:
                logger.info("Trying simplified version...")
                self.run_development_server(use_simple=True)
    
    def run_production_server(self):
        """Run the production server using Docker."""
        logger.info("Starting production server with Docker...")
        
        try:
            # Check if Docker is available
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            
            # Build and run with docker-compose
            subprocess.run([
                "docker-compose", "up", "--build", "-d"
            ], check=True, cwd=self.project_root)
            
            logger.info("✅ Production server started successfully")
            logger.info("Services:")
            logger.info("  - API: http://localhost:8000")
            logger.info("  - Grafana: http://localhost:3000 (admin/admin123)")
            logger.info("  - Prometheus: http://localhost:9090")
            
            logger.info("To stop the services, run: docker-compose down")
            
        except subprocess.CalledProcessError:
            logger.error("Docker is not available or docker-compose failed")
            logger.info("Please install Docker and Docker Compose for production deployment")
        except Exception as e:
            logger.error(f"Production server startup failed: {e}")
    
    def show_system_status(self):
        """Show current system status."""
        logger.info("System Status:")
        logger.info("=" * 50)
        
        # Check files
        logger.info(f"Excel file: {'✅' if self.excel_file.exists() else '❌'} {self.excel_file}")
        logger.info(f"Database: {'✅' if self.db_file.exists() else '❌'} {self.db_file}")
        logger.info(f"Config: {'✅' if self.env_file.exists() else '❌'} {self.env_file}")
        
        # Check database size if it exists
        if self.db_file.exists():
            size_mb = self.db_file.stat().st_size / (1024 * 1024)
            logger.info(f"Database size: {size_mb:.2f} MB")
        
        # Check if services are running (Docker)
        try:
            result = subprocess.run([
                "docker-compose", "ps", "--services", "--filter", "status=running"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0 and result.stdout.strip():
                running_services = result.stdout.strip().split('\n')
                logger.info(f"Running services: {', '.join(running_services)}")
            else:
                logger.info("No Docker services running")
                
        except Exception:
            logger.info("Docker status: Not available")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EPCL VEHS System Runner")
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--dev", action="store_true", help="Run in development mode")
    parser.add_argument("--simple", action="store_true", help="Run simplified version (SQL-only, no LangChain)")
    parser.add_argument("--production", action="store_true", help="Run in production mode")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--install", action="store_true", help="Install dependencies only")
    
    args = parser.parse_args()
    
    runner = EPCLSystemRunner()
    
    # Show status and exit
    if args.status:
        runner.show_system_status()
        return
    
    # Install dependencies only
    if args.install:
        if runner.check_prerequisites():
            runner.install_dependencies()
        return
    
    # Check prerequisites
    if not runner.check_prerequisites():
        logger.error("Prerequisites check failed. Please fix the issues above.")
        return
    
    # Install dependencies
    if not runner.install_dependencies():
        logger.error("Dependency installation failed.")
        return
    
    # Run data ingestion if requested
    if args.ingest:
        if not runner.run_ingestion():
            logger.error("Data ingestion failed.")
            return
    
    # Check if database exists
    if not runner.db_file.exists():
        logger.warning("Database not found. Running data ingestion...")
        if not runner.run_ingestion():
            logger.error("Data ingestion failed.")
            return
    
    # Run tests if requested
    if args.test:
        if not runner.run_tests():
            logger.warning("Some tests failed, but continuing...")
    
    # Check OpenAI API key (only for full version)
    if not args.simple and not runner.check_openai_key():
        logger.warning("OpenAI API key not configured. Using simplified version.")
        args.simple = True
    
    # Run the appropriate server
    if args.production:
        runner.run_production_server()
    elif args.simple:
        runner.run_development_server(use_simple=True)
    elif args.dev:
        runner.run_development_server(use_simple=False)
    else:
        # Default: show help and status
        parser.print_help()
        print("\n")
        runner.show_system_status()
        print("\nTo start the system:")
        print("  Development: python run_system.py --dev")
        print("  Simplified:  python run_system.py --simple")
        print("  Production:  python run_system.py --production")


if __name__ == "__main__":
    main()
