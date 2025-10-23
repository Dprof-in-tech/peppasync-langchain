# PeppaSync LangChain - Business Intelligence Platform

A modern Business Intelligence platform powered by LangGraph and LangChain with Pinecone vector store for AI-driven analytics and automated business insights.

##  System Architecture & Flow

### Core System Flow

```mermaid
graph TD
    A[User Request] --> B{Endpoint Type}

    B -->|Chat| C[Chat Endpoint with Sessions]
    B -->|Direct Analysis| D[Retrieve & Generate]
    B -->|Visualization| E[Retrieve & Visualize]
    B -->|Database| F[Database Management]
    B -->|Shopify| G[Shopify Integration]

    C --> H[Conversation Manager]
    H --> I[Query Classification]
    I --> J[LangGraph Workflow]
    J --> K[Unified Business Agent]

    D --> L[GenBISQL Engine]
    E --> L

    L --> M[Data Retrieval Layer]
    M --> N[Pinecone Vector Store]
    M --> O[Session Manager]

    O --> P{Data Source Type?}
    P -->|PostgreSQL| Q[User Database]
    P -->|Shopify| R[Shopify Orders/Products]
    P -->|None| S[Mock Data Fallback]

    G --> T[Shopify Service]
    T --> U[External OAuth Connector]
    U --> V[Shopify API]

    N --> W[Business Knowledge]
    Q --> X[Real Database Data]
    R --> Y[Real Shopify Data]
    S --> Z[Sample Data]

    W --> AA[LLM Processing]
    X --> AA
    Y --> AA
    Z --> AA

    AA --> AB[Response Generation]
    AB --> AC[Formatted Output]

    K --> AD[Agent Response]
    AD --> AE[Session Storage Redis]
    AE --> AF[Chat Response]
```

### Data Flow Architecture

```mermaid
graph LR
    subgraph "Data Sources"
        A[Pinecone Vector Store<br/>15 Business Docs]
        B[PostgreSQL Database<br/>User Connected]
        C[Shopify Store<br/>OAuth Connected]
        D[Mock Data<br/>Fallback Sample]
    end

    subgraph "Session Management"
        E[Redis Session Manager<br/>Connection State]
        F[DatabaseManager<br/>Query Routing]
    end

    subgraph "Processing Layer"
        G[Query Classification]
        H[Vector Similarity Search]
        I[Data Source Selector]
        J[LLM Processing]
    end

    subgraph "Endpoints"
        K[Chat - Conversational]
        L[Analytics - Metrics]
        M[Shopify - Integration]
    end

    A --> H
    B --> F
    C --> F
    D --> F

    E --> I
    F --> I
    H --> J
    I --> J
    G --> J

    J --> K
    J --> L
    M --> C
```

### Shopify OAuth Connection Flow

```mermaid
sequenceDiagram
    participant U as User/Frontend
    participant B as Backend API
    participant C as External Connector
    participant S as Shopify OAuth
    participant R as Redis Storage

    U->>B: POST /shopify/connect {shop_name, session_id}
    B->>C: POST /connect/shopify {store, redirect_url}
    C->>B: Returns {auth_url}
    B->>R: Store connection info
    B->>U: Returns {auth_url}

    U->>S: User clicks auth_url & authorizes
    S->>C: OAuth callback with access_token
    C->>U: Redirects to frontend callback

    U->>B: GET /shopify/status/{session_id} (polling)
    B->>C: Check if OAuth complete
    C-->>B: Connection data
    B->>R: Update connection with access_token
    B->>S: Fetch orders using access_token
    S->>B: Returns orders data
    B->>R: Store orders in Redis
    B->>U: Returns {connected: true, orders_count}

    Note over U,R: User sees "Connected to Shopify" only after orders synced
```

## 🛡️ Endpoint Architecture

### Active Endpoints (Current System)

| Endpoint | Purpose | Session Tracking | Data Sources |
|----------|---------|------------------|--------------|
| `POST /chat` | **Conversational BI** with context memory | Required | Pinecone + Database/Shopify/Mock |
| `POST /retrieve_and_generate` | **Standalone insights** - independent queries | ❌ Removed | Pinecone + Database/Mock |
| `POST /retrieve_and_visualize` | **Standalone visualizations** - chart generation | ❌ Removed | Pinecone + Database/Mock |
| `POST /database/test` | Test PostgreSQL connection | ❌ None | Direct DB Test |
| `POST /database/connect` | Connect user database | Required | Connection Storage |
| `GET /database/status/{id}` | Check connection status | Required | Connection Info |
| `POST /database/disconnect/{id}` | Disconnect database | Required | Connection Cleanup |
| `POST /shopify/connect` | **Initiate Shopify OAuth** | Required | External Connector API |
| `GET /shopify/status/{id}` | **Check Shopify connection & auto-sync** | Required | Redis + Shopify API |
| `POST /shopify/oauth-callback` | **Receive OAuth access_token** | Required | Token Storage + Sync |
| `POST /shopify/disconnect/{id}` | **Disconnect Shopify store** | Required | Connection Cleanup |
| `GET /analytics/sales` | Sales analytics data | Required | Active Data Source |
| `GET /analytics/orders` | Orders analytics data | Required | Active Data Source |
| `GET /analytics/inventory` | Inventory analytics data | Required | Active Data Source |

### Redundant Endpoints (Should be removed)

| Endpoint | Why Redundant | Better Alternative |
|----------|---------------|-------------------|
| `POST /analytics/{type}` | Hardcoded queries to same agent | Use `/chat` with natural language |
| `POST /agents/inventory/run` | Hardcoded "monitor inventory" | Ask "monitor my inventory" in `/chat` |
| `POST /agents/marketing/run` | Hardcoded "analyze marketing" | Ask "analyze my marketing" in `/chat` |
| `POST /unified-analysis` | Duplicate of `/chat` functionality | Use `/chat` directly |

## 🧠 Intelligence Layer

### Query Classification System

```mermaid
graph TD
    A[User Query] --> B[Query Classifier]
    B --> C{Needs Specific Data?}

    C -->|Yes| D[Data Request Strategy]
    C -->|No| E[General Advice Strategy]

    D --> F[Check Database Connection]
    F -->|Connected| G[Query User Database]
    F -->|Not Connected| H[Use Mock Data + Request Connection]

    E --> I[Business Knowledge Only]

    G --> J[Combine with Pinecone Knowledge]
    H --> J
    I --> J

    J --> K[LLM Generation]
    K --> L[Response with Citations]
```

### Business Knowledge Base (Pinecone)

The system includes 15 pre-loaded business knowledge documents:

- **Products**: Electronics, fashion, home goods categories
- **Sales Patterns**: Seasonal trends (Q4/Q2 peaks), global ecommerce patterns
- **Customer Demographics**: Age groups, gender distribution, behavior
- **Marketing**: Multi-platform campaigns (Facebook, Instagram, Google, TikTok)
- **Inventory**: Just-in-time principles, reorder management
- **Metrics**: KPIs, ROAS targets, performance indicators
- **Market Context**: Global ecommerce characteristics, payment preferences
- **Challenges**: Common business problems and solutions

## 🔧 Technical Implementation

### Core Components

1. **GenBISQL Engine** (`lib/peppagenbi.py`)
   - Pinecone vector store integration
   - Mock data fallback system
   - LLM query processing
   - Response generation and formatting

2. **Conversation Manager** (`lib/conversation_manager.py`)
   - LangGraph workflow orchestration
   - Session memory management
   - Context-aware responses
   - Query classification integration

3. **Database Manager** (`lib/config.py`)
   - PostgreSQL connection management
   - Session-based user database storage
   - Automatic fallback to mock data
   - Query type detection and routing

4. **Query Classifier** (`lib/query_classifier.py`)
   - Determines data needs vs general advice
   - Classifies query types (sales, inventory, customer)
   - Confidence scoring for query understanding

### Data Retrieval Logic

```python
# Simplified flow in _get_database_data()
def _get_database_data(query):
    # 1. Determine query type from keywords
    if "sales" in query: query_type = "sales_data"
    elif "inventory" in query: query_type = "inventory_data"

    # 2. Use DatabaseManager with automatic fallback
    return DatabaseManager.get_data(
        session_id=None,
        query_type=query_type,
        use_mock=True  # Falls back to mock if no DB connected
    )
```

### Vector Store Integration

```python
# Pinecone initialization with dimension handling
def _init_pinecone_vector_store():
    # 1. Check existing index dimensions
    # 2. Delete and recreate if mismatch (2048 → 1536)
    # 3. Populate with business knowledge documents
    # 4. Enable similarity search for context retrieval
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API Key
- Pinecone API Key
- PostgreSQL (optional, uses mock data by default)

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd peppasync-langchain
   ```

2. **Environment Configuration**
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_NAME=peppasync
   DATABASE_URL=postgresql://user:pass@host:port/db
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pinecone langchain-pinecone  # Vector store
   ```

4. **Run Application**
   ```bash
   python app.py
   ```
   Access at: http://localhost:8000

## 📊 Usage Examples

### Chat Endpoint (Conversational with Memory)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the factors causing declining Nike shoe sales?",
    "session_id": "user123"
  }'

# Follow-up in same session
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What should I do about it?",
    "session_id": "user123"
  }'
```

### Retrieve & Generate (Standalone Insights)
```bash
curl -X POST "http://localhost:8000/retrieve_and_generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyze sales trends for electronics"}'
```

### Retrieve & Visualize (Standalone Charts)
```bash
curl -X POST "http://localhost:8000/retrieve_and_visualize" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create charts showing monthly sales performance"}'
```

### Database Connection
```bash
# Test connection
curl -X POST "http://localhost:8000/database/test" \
  -H "Content-Type: application/json" \
  -d '{"database_url": "postgresql://user:pass@host:port/db"}'

# Connect for session
curl -X POST "http://localhost:8000/database/connect" \
  -H "Content-Type: application/json" \
  -d '{
    "database_url": "postgresql://user:pass@host:port/db",
    "session_id": "user123"
  }'
```

## 💾 Data Sources Explained

### 1. Pinecone Vector Store
- **Purpose**: Business knowledge and best practices
- **Content**: 15 documents covering sales, marketing, inventory, market insights
- **Usage**: Provides context for all business questions
- **Automatic**: Always queried for relevant business knowledge

### 2. User PostgreSQL Database (Optional)
- **Purpose**: User's actual business data from custom database
- **Content**: Sales transactions, inventory, customer data
- **Usage**: When user connects their database via `/database/connect`
- **Connection**: Per-session, stored in Redis
- **Authentication**: Direct connection string with credentials

### 3. Shopify Store (Optional)
- **Purpose**: User's Shopify ecommerce data
- **Content**: Orders, products, customers from Shopify store
- **Usage**: When user connects via OAuth flow (`/shopify/connect`)
- **Connection**: Per-session, OAuth 2.0 with access token
- **Auto-sync**: Orders automatically synced after OAuth completion
- **Storage**: Connection data and orders stored in Redis with 24hr TTL

### 4. Mock Data (Fallback)
- **Purpose**: Demo data when no real data source connected
- **Content**: Sample sales/inventory records (iPhone, Samsung, Nike)
- **Usage**: Automatic fallback for demonstrations
- **Limitation**: Limited to 3 sample records

### Data Source Priority (Single Source at a Time)
The system uses **ONE data source per session**:
1. If PostgreSQL connected → Use database data
2. Else if Shopify connected → Use Shopify data
3. Else → Use mock data fallback

**Note**: User cannot have both PostgreSQL and Shopify connected simultaneously. Connecting one disconnects the other.

##  System Benefits

### Simplified Architecture
- **Stateless Core**: Most endpoints don't need session tracking
- **Smart Fallbacks**: Automatic mock data when no database connected
- **Unified Processing**: Single LLM handles all business questions
- **Clean Separation**: Chat for conversations, direct endpoints for standalone use

### Intelligent Query Handling
- **Auto-Classification**: Determines if question needs specific data
- **Context Retrieval**: Always includes relevant business knowledge
- **Flexible Input**: Natural language queries work across all endpoints
- **Consistent Output**: Structured responses with citations

### Global Ecommerce Focus
- **Currency**: All values in US Dollars ($)
- **Local Context**: International market characteristics, payment preferences
- **Business Patterns**: Global seasonal trends, cultural considerations
- **Regional Insights**: Supply chain, logistics, economic factors

## 🛠️ Development

### Technology Stack
- **Framework**: FastAPI for high-performance APIs
- **AI/ML**: LangChain + LangGraph for agent workflows
- **LLM**: OpenAI GPT-4o-mini for cost-effective AI
- **Vector Store**: Pinecone for scalable knowledge retrieval
- **Database**: PostgreSQL with automatic mock fallback
- **Deployment**: Docker-ready, cloud-agnostic

### Key Design Principles
1. **Simplicity**: Remove unnecessary complexity (sessions only where needed)
2. **Reliability**: Always have fallback data sources
3. **Flexibility**: Natural language input, structured output
4. **Performance**: Efficient vector search and LLM processing
5. **Scalability**: Stateless design for horizontal scaling

---

**Built with ❤️ for global businesses using modern AI technologies**