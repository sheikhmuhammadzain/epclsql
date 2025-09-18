# EPCL VEHS SQL Agent System

A production-ready Natural Language to SQL system for EPCL (Engro Polymer & Chemicals Limited) VEHS (Visible Environmental Health & Safety) data analysis.

## 🌟 Features

- **Natural Language Queries**: Ask questions in plain English about safety incidents, hazards, audits, and inspections
- **Secure SQL Execution**: Comprehensive validation and injection prevention
- **LangChain Integration**: Advanced AI-powered query understanding and generation
- **Production Ready**: FastAPI backend with authentication, rate limiting, and monitoring
- **Interactive Chat Interface**: Web-based UI for easy data exploration
- **Comprehensive Testing**: Full test suite with security, performance, and integration tests
- **Docker Deployment**: Container-based deployment with monitoring stack

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Excel Data    │───▶│  SQLite Database │───▶│  LangChain Agent│
│  (6 Sheets)     │    │  (Normalized)    │    │  (GPT-4o-mini)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │◀───│   FastAPI REST   │◀───│ Safe SQL Executor│
│   (Chat UI)     │    │     API          │    │  (Validation)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Data Schema

The system processes 6 main data types:

- **Incidents** (169 columns): Safety incidents, near misses, injuries
- **Hazard ID** (169 columns): Identified hazards and risk assessments
- **Audits** (50 columns): Safety audits and evaluations
- **Audit Findings** (50 columns): Specific audit observations
- **Inspections** (50 columns): Regular safety inspections
- **Inspection Findings** (50 columns): Inspection results and recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API Key
- Excel file: `EPCL VEHS Data (Mar23 - Mar24).xlsx`

### Installation

1. **Clone or download the project files**

2. **Set up environment**:
   ```bash
   # Copy environment template
   copy .env.example .env
   
   # Edit .env file and add your OpenAI API key
   # OPENAI_API_KEY=your_api_key_here
   ```

3. **Run the system**:
   ```bash
   # Install dependencies, ingest data, and start development server
   python run_system.py --ingest --dev
   ```

4. **Access the interface**:
   - Open http://127.0.0.1:8000 in your browser
   - Start asking questions about your safety data!

### Example Queries

- "How many incidents occurred in the last 6 months?"
- "What are the top 5 incident categories by frequency?"
- "Which locations have the highest incident rates?"
- "Show me incidents with high injury potential"
- "What is the total cost of incidents this year?"
- "What are the most common audit findings?"

## 🔧 Detailed Setup

### 1. Data Ingestion

```bash
# Run data ingestion separately
python ingest_excel_to_sqlite.py
```

This creates:
- `epcl_vehs.db`: Main SQLite database
- Core columns for fast querying
- Extra JSON fields for complete data retention
- Indexes for performance
- Summary tables for common queries

### 2. Development Mode

```bash
# Start development server with auto-reload
python run_system.py --dev
```

Features:
- Auto-reload on code changes
- Detailed logging
- Interactive API docs at `/docs`

### 3. Production Deployment

```bash
# Deploy with Docker (recommended)
python run_system.py --production
```

This starts:
- **API Server**: http://localhost:8000
- **Monitoring**: http://localhost:3000 (Grafana)
- **Metrics**: http://localhost:9090 (Prometheus)
- **Caching**: Redis for performance

### 4. Testing

```bash
# Run comprehensive test suite
python run_system.py --test
```

Tests include:
- SQL injection prevention
- Query validation
- API endpoints
- Performance benchmarks
- Security features

## 🔒 Security Features

### SQL Injection Prevention
- Whitelist-based SQL validation
- Parameter binding enforcement
- Query pattern analysis
- Forbidden keyword detection

### Access Control
- API key authentication
- Rate limiting (10 queries/minute, 100/hour)
- Query complexity limits
- Table access restrictions

### Safe Execution
- Only SELECT statements allowed
- Automatic LIMIT clause addition
- Query timeout protection
- Result size limiting

## 📡 API Endpoints

### Core Endpoints

```http
POST /query
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "query": "How many incidents occurred last month?",
  "include_explanation": true,
  "use_cache": true
}
```

### Other Endpoints

- `GET /`: Interactive chat interface
- `GET /health`: System health check
- `GET /metrics`: Performance metrics
- `GET /schema`: Database schema info
- `GET /suggestions`: Query suggestions

## 🎯 Query Examples by Category

### Incident Analysis
```
"How many incidents occurred in the last 6 months?"
"What are the top 5 incident categories by frequency?"
"Which locations have the highest incident rates?"
"Show me incidents with total cost over $5000"
"What is the trend of incidents over time?"
```

### Risk Assessment
```
"Which departments have the most high-risk incidents?"
"Show me incidents with high injury potential"
"What are the most common causes of incidents?"
"Which equipment failures led to incidents?"
"What are the worst-case consequences we've seen?"
```

### Audit & Compliance
```
"How many audits were completed this year?"
"What are the most common audit findings?"
"Which locations need the most attention based on audits?"
"Show me overdue action items"
"What is the status of corrective actions?"
```

### Performance Metrics
```
"What is our incident closure rate?"
"How long does it take to close incidents on average?"
"Which contractors have the best safety record?"
"Compare this year's performance to last year"
"Show me monthly incident statistics"
```

## 🔧 Configuration

### Environment Variables

```bash
# Core Configuration
OPENAI_API_KEY=your_openai_api_key_here
EPCL_API_KEY=your_secure_api_key
DATABASE_PATH=epcl_vehs.db

# Server Settings
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO

# Rate Limiting
MAX_QUERIES_PER_MINUTE=10
MAX_QUERIES_PER_HOUR=100

# Security
JWT_SECRET_KEY=your_jwt_secret
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

### Database Configuration

The system uses a hybrid approach:
- **Core columns**: Frequently queried fields (incident_number, date_of_occurrence, location, etc.)
- **Extra JSON**: All remaining fields stored as JSON for complete data retention
- **Summary tables**: Pre-computed aggregations for fast analytics

## 📈 Monitoring & Metrics

### Built-in Metrics
- Total queries executed
- Success/failure rates
- Average response times
- Query complexity scores
- Cache hit rates

### Grafana Dashboards
- Query performance trends
- Error rate monitoring
- Resource utilization
- User activity patterns

### Prometheus Metrics
- HTTP request metrics
- Database query times
- LLM API usage
- System resource usage

## 🧪 Testing

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end workflow testing
3. **Security Tests**: SQL injection and access control
4. **Performance Tests**: Query execution speed and scalability
5. **API Tests**: REST endpoint functionality

### Running Tests

```bash
# Run all tests
python test_suite.py

# Run specific test categories
pytest test_suite.py::TestSQLValidator -v
pytest test_suite.py::TestSecurityFeatures -v
pytest test_suite.py::TestAPIEndpoints -v
```

## 🚀 Deployment Options

### Development
```bash
python run_system.py --dev
```
- Single process
- Auto-reload
- Debug logging
- SQLite database

### Production (Docker)
```bash
python run_system.py --production
```
- Multi-container setup
- Load balancing
- Monitoring stack
- Backup automation
- SSL termination

### Production (Manual)
```bash
# Install dependencies
pip install -r requirements.txt

# Set production environment variables
export OPENAI_API_KEY=your_key
export EPCL_API_KEY=production_key

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📁 Project Structure

```
epcl-vehs-sql-agent/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env.example                 # Environment template
├── 📄 Dockerfile                   # Container configuration
├── 📄 docker-compose.yml           # Multi-service deployment
├── 📄 run_system.py                # System runner script
│
├── 🔧 Core Components
│   ├── ingest_excel_to_sqlite.py   # Data ingestion
│   ├── safe_sql_executor.py        # Secure SQL execution
│   ├── langchain_sql_agent.py      # AI agent implementation
│   └── main.py                     # FastAPI application
│
├── 🧪 Testing
│   └── test_suite.py               # Comprehensive tests
│
├── 📊 Data
│   ├── EPCL VEHS Data (Mar23 - Mar24).xlsx  # Source data
│   ├── epcl_vehs.db                # SQLite database (generated)
│   └── excel_analysis_report.txt   # Column analysis
│
└── 📁 Deployment
    ├── nginx/                      # Web server config
    ├── monitoring/                 # Grafana/Prometheus
    └── backups/                    # Database backups
```

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   Error: OpenAI API key not configured
   Solution: Set OPENAI_API_KEY in .env file
   ```

2. **Database Not Found**
   ```
   Error: Database file not found
   Solution: Run python run_system.py --ingest
   ```

3. **Permission Denied**
   ```
   Error: Permission denied accessing database
   Solution: Check file permissions or run as administrator
   ```

4. **Rate Limit Exceeded**
   ```
   Error: Too many requests
   Solution: Wait or increase rate limits in configuration
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python run_system.py --dev
```

### Health Checks

```bash
# Check system status
python run_system.py --status

# Test API endpoint
curl http://localhost:8000/health
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install black flake8 mypy pytest
   ```
4. Run tests: `python test_suite.py`
5. Format code: `black .`
6. Submit pull request

### Code Standards

- Python 3.8+ compatibility
- Type hints for all functions
- Comprehensive docstrings
- Security-first approach
- Performance considerations

## 📝 License

This project is proprietary software developed for EPCL (Engro Polymer & Chemicals Limited).

## 🆘 Support

For technical support or questions:

1. Check the troubleshooting section above
2. Review the logs in `epcl_api.log`
3. Run the test suite to identify issues
4. Contact the development team

## 🔄 Updates & Maintenance

### Regular Tasks

1. **Database Backup**: Automated daily backups
2. **Log Rotation**: Weekly log cleanup
3. **Security Updates**: Monthly dependency updates
4. **Performance Monitoring**: Continuous metrics collection

### Version History

- **v1.0.0**: Initial production release
  - Complete Excel ingestion pipeline
  - LangChain SQL agent implementation
  - FastAPI REST interface
  - Comprehensive security features
  - Docker deployment configuration
  - Full test suite

---

**Built with ❤️ for EPCL Safety Excellence**
