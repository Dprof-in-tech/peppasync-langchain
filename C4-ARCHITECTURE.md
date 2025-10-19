# PeppaSync - System Architecture Documentation

> Complete C4 Model and UML diagrams for the PeppaSync Business Intelligence Platform

---

## Level 1: System Context

**Overview**: How PeppaSync fits into the user's world

```mermaid
graph TB
    User([Business Owner])

    PeppaSync[PeppaSync Platform<br/>AI-Powered BI System]

    Shopify[Shopify API]
    PostgreSQL[(PostgreSQL<br/>Database)]
    Connector[OAuth Connector<br/>connector.fundam.ng]
    OpenAI[OpenAI GPT-4]
    Pinecone[Pinecone<br/>Vector DB]
    Redis[(Redis<br/>Cache)]

    User -->|HTTPS| PeppaSync

    PeppaSync -->|Fetch data| Shopify
    PeppaSync -->|Query| PostgreSQL
    PeppaSync -->|OAuth| Connector
    PeppaSync -->|AI requests| OpenAI
    PeppaSync -->|Knowledge| Pinecone
    PeppaSync -->|Sessions| Redis

    style User fill:#08427b,stroke:#052e56,color:#fff
    style PeppaSync fill:#1168bd,stroke:#0b4884,color:#fff
    style Shopify fill:#999,stroke:#666,color:#fff
    style PostgreSQL fill:#999,stroke:#666,color:#fff
    style Connector fill:#999,stroke:#666,color:#fff
    style OpenAI fill:#999,stroke:#666,color:#fff
    style Pinecone fill:#999,stroke:#666,color:#fff
    style Redis fill:#999,stroke:#666,color:#fff
```

---

## Level 2: Container Diagram

**Overview**: Major components and their responsibilities

```mermaid
graph TB
    User([Business Owner])

    subgraph Platform["PeppaSync Platform"]
        Frontend[Next.js Web App<br/>React + TypeScript]
        Backend[FastAPI Backend<br/>Python]
        AISystem[LangChain/LangGraph<br/>AI Agent System]
        SessionStore[(Redis<br/>Sessions)]
    end

    Shopify[Shopify API]
    UserDB[(PostgreSQL)]
    OpenAI[OpenAI]
    Pinecone[Pinecone]

    User -->|HTTPS| Frontend
    Frontend -->|JSON/REST| Backend
    Backend --> AISystem
    Backend --> SessionStore
    Backend -->|Data| Shopify
    Backend -->|Query| UserDB
    AISystem -->|LLM| OpenAI
    AISystem -->|RAG| Pinecone

    style User fill:#08427b,color:#fff
    style Frontend fill:#1168bd,color:#fff
    style Backend fill:#1168bd,color:#fff
    style AISystem fill:#1168bd,color:#fff
    style SessionStore fill:#1168bd,color:#fff
```

---

## Level 3: Frontend Components

**Overview**: Next.js application structure

```mermaid
graph TB
    subgraph Pages["Pages"]
        Home[Home Page<br/>Dashboard]
        Callback[Shopify Callback<br/>OAuth Handler]
    end

    subgraph Components["React Components"]
        AIChat[AI Assistant]
        Analytics[Analytics Dashboard]
        DBConn[Database Connection]
        ShopifyConn[Shopify Connection]
        Forecast[Forecast Settings]
    end

    subgraph State["State Management"]
        SessionCtx[Session Context<br/>sessionId + dataSource]
        AnalyticsCtx[Analytics Context<br/>Data caching]
    end

    subgraph Services["Services"]
        API[API Service<br/>Singleton client]
    end

    LocalStorage[(Browser<br/>localStorage)]

    Home --> AIChat
    Home --> Analytics
    Home --> DBConn
    Home --> ShopifyConn

    AIChat --> Forecast
    AIChat --> SessionCtx
    AIChat --> API

    Analytics --> AnalyticsCtx
    AnalyticsCtx --> SessionCtx
    AnalyticsCtx --> API

    SessionCtx <--> LocalStorage
    API -->|HTTP| Backend[Backend API]

    style Home fill:#4a90e2,color:#fff
    style AIChat fill:#4a90e2,color:#fff
    style Analytics fill:#4a90e2,color:#fff
    style SessionCtx fill:#50c878,color:#fff
    style AnalyticsCtx fill:#50c878,color:#fff
    style API fill:#f5a623,color:#fff
```

---

## Level 3: Backend Components

**Overview**: FastAPI application structure

```mermaid
graph TB
    subgraph Routes["API Endpoints"]
        Chat[/chat<br/>Conversational BI]
        Analytics[/analytics/*<br/>Sales/Orders/Inventory]
        Forecast[/forecast<br/>Demand forecasting]
        Shopify[/shopify/*<br/>Connect/Status/Disconnect]
        Database[/database/*<br/>Connect/Test/Status]
    end

    subgraph Core["Core Services"]
        ConvMgr[Conversation Manager<br/>LangGraph orchestration]
        BizAgent[Business Agent<br/>Unified query handler]
        QueryClass[Query Classifier<br/>Intent detection]
        GenBISQL[GenBISQL Engine<br/>RAG + LLM]
    end

    subgraph Managers["Resource Managers"]
        DBMgr[Database Manager<br/>Connection pooling]
        ShopifySvc[Shopify Service<br/>Connector API client]
        RedisMgr[Redis Manager<br/>Session persistence]
    end

    Chat --> ConvMgr
    ConvMgr --> BizAgent
    BizAgent --> QueryClass
    BizAgent --> GenBISQL

    Analytics --> DBMgr
    Forecast --> DBMgr
    Shopify --> ShopifySvc
    Database --> DBMgr

    ShopifySvc --> DBMgr
    GenBISQL --> DBMgr

    DBMgr --> RedisMgr
    ShopifySvc --> RedisMgr

    style Chat fill:#4a90e2,color:#fff
    style Analytics fill:#4a90e2,color:#fff
    style ConvMgr fill:#50c878,color:#fff
    style BizAgent fill:#50c878,color:#fff
    style DBMgr fill:#f5a623,color:#fff
    style ShopifySvc fill:#f5a623,color:#fff
```

---

## Shopify OAuth Integration Flow

**Detailed sequence of Shopify connection process**

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend
    participant API as Backend
    participant SS as Shopify Service
    participant Conn as OAuth Connector
    participant Shop as Shopify
    participant Redis
    participant CB as Callback Page

    User->>FE: Click Connect
    FE->>FE: Generate session_id
    Note over FE: Save to localStorage

    FE->>API: POST /shopify/connect
    API->>SS: connect_shopify_store()
    SS->>Conn: POST /connect/shopify
    Conn-->>SS: {auth_url}
    API->>Redis: Store connection
    API-->>FE: {auth_url}

    FE->>FE: Display auth link
    User->>Shop: Authorize app
    Shop->>Conn: OAuth callback
    Conn-->>CB: Redirect with success

    loop Poll for orders
        CB->>API: GET /shopify/status
        API->>SS: get_connection_id()
        SS->>Conn: GET /connections
        Conn-->>SS: [{id, slug}]

        API->>SS: auto_sync_orders()
        SS->>Conn: GET /fetch/orders/{id}
        Conn-->>SS: [orders]
        SS->>Redis: Store orders
        API-->>CB: {connected, orders_count}
    end

    CB->>CB: Success! Redirect home
```

---

## Data Source Architecture

**Single data source per session**

```mermaid
flowchart TD
    User[User Session]

    subgraph Frontend
        Session[Session Context<br/>sessionId + dataSource]
        LS[localStorage]
    end

    subgraph Backend
        Router[API Router]
        DBMgr[Database Manager]
    end

    subgraph "Data Sources (ONE at a time)"
        PG[(PostgreSQL<br/>dataSource='postgres')]
        Shopify[(Shopify<br/>dataSource='shopify')]
        Mock[Mock Data<br/>dataSource=null]
    end

    Redis[(Redis<br/>State Store)]

    User --> Session
    Session <--> LS
    Session -->|Determines source| Router
    Router --> DBMgr

    DBMgr -->|if postgres| PG
    DBMgr -->|if shopify| Shopify
    DBMgr -->|if null| Mock

    DBMgr <--> Redis

    style Session fill:#50c878,color:#fff
    style DBMgr fill:#f5a623,color:#fff
    style PG fill:#e74c3c,color:#fff
    style Shopify fill:#3498db,color:#fff
    style Mock fill:#95a5a6,color:#fff
```

---

## AI Agent Workflow

**LangGraph conversation flow**

```mermaid
graph TB
    UserQuery[User Query]

    ConvMgr[Conversation Manager<br/>Session + History]

    QueryClass{Query Classifier<br/>What type?}

    DataQuery[Data-Specific Query<br/>needs DB/Shopify]
    GeneralAdv[General Business Advice<br/>knowledge only]

    DBCheck{Data Source<br/>Connected?}

    GetData[Fetch from DB/Shopify<br/>via DatabaseManager]
    UseMock[Use Mock Data<br/>+ suggest connection]

    VectorSearch[Pinecone RAG<br/>Business knowledge]

    LLM[OpenAI GPT-4<br/>Generate response]

    Response[AI Response<br/>with citations]

    UserQuery --> ConvMgr
    ConvMgr --> QueryClass

    QueryClass -->|Needs data| DataQuery
    QueryClass -->|General| GeneralAdv

    DataQuery --> DBCheck
    DBCheck -->|Yes| GetData
    DBCheck -->|No| UseMock

    GetData --> VectorSearch
    UseMock --> VectorSearch
    GeneralAdv --> VectorSearch

    VectorSearch --> LLM
    LLM --> Response

    style QueryClass fill:#f39c12,color:#fff
    style DBCheck fill:#f39c12,color:#fff
    style VectorSearch fill:#9b59b6,color:#fff
    style LLM fill:#9b59b6,color:#fff
    style Response fill:#27ae60,color:#fff
```

---

## Deployment Architecture

```mermaid
graph TB
    subgraph Internet
        Users([Users])
    end

    subgraph "Vercel (Frontend)"
        NextJS[Next.js App<br/>SSR + Static]
        Edge[Edge Functions]
    end

    subgraph "Cloud VM (Backend)"
        FastAPI[FastAPI Server<br/>:8000]
        Workers[Background Workers<br/>Forecasting]
    end

    subgraph "Managed Services"
        RedisCloud[(Redis Cloud<br/>Sessions)]
        PineconeCloud[Pinecone Cloud<br/>Vectors]
        OpenAIAPI[OpenAI API<br/>GPT-4]
    end

    subgraph "External"
        ShopifyStores[Shopify Stores]
        Connector[OAuth Connector<br/>connector.fundam.ng]
        UserDB[(Customer DB)]
    end

    Users -->|HTTPS| NextJS
    NextJS -->|API| FastAPI
    FastAPI --> RedisCloud
    FastAPI --> PineconeCloud
    FastAPI --> OpenAIAPI
    FastAPI --> Connector
    Connector --> ShopifyStores
    FastAPI --> UserDB
    Workers --> FastAPI

    style NextJS fill:#1168bd,color:#fff
    style FastAPI fill:#1168bd,color:#fff
    style RedisCloud fill:#dc382d,color:#fff
    style PineconeCloud fill:#000,color:#fff
    style OpenAIAPI fill:#10a37f,color:#fff
```

---

## Technology Stack

### Frontend Stack
```
┌─────────────────────────────────┐
│   Next.js 14 (App Router)       │
│   React 18 + TypeScript          │
│   Tailwind CSS                   │
│   Context API (State)            │
│   Recharts (Visualization)       │
│   Lucide Icons                   │
└─────────────────────────────────┘
```

### Backend Stack
```
┌─────────────────────────────────┐
│   FastAPI (Python 3.9+)          │
│   LangChain + LangGraph          │
│   Prophet + ARIMA (Forecasting)  │
│   OpenAI GPT-4o-mini             │
│   Pinecone (Vector DB)           │
│   Redis (Sessions)               │
│   PostgreSQL (User data)         │
└─────────────────────────────────┘
```

---

## Key Architectural Decisions

### 1. Single Data Source Architecture
**Decision**: Only ONE data source active per session (PostgreSQL OR Shopify OR Mock)

**Rationale**:
- Prevents data confusion and mixing
- Clear source of truth for analytics
- Simpler state management
- Easier to debug and maintain

### 2. Frontend-Tracked Connection State
**Decision**: Store connection state in localStorage + React Context, not backend polling

**Rationale**:
- Reduces backend load
- Faster UI responsiveness
- Works offline/during backend restart
- Survives page refreshes

### 3. External OAuth Connector
**Decision**: Use external connector service instead of direct Shopify OAuth

**Rationale**:
- Outsources OAuth complexity
- No need to manage Shopify app credentials
- Connector handles token refresh
- Reduces security surface area

### 4. Unified AI Agent Pattern
**Decision**: Single LangGraph agent handles all query types

**Rationale**:
- Eliminates endpoint proliferation
- Natural language interface
- Context-aware responses
- Easier to extend with new capabilities

### 5. Redis-First Session Management
**Decision**: All sessions and connections in Redis with 24hr TTL

**Rationale**:
- Fast session lookup
- Automatic cleanup via TTL
- Horizontal scalability
- No database overhead for temporary state

---

## Security Architecture

```mermaid
graph TB
    subgraph "Security Layers"
        HTTPS[HTTPS/TLS<br/>All external comms]
        Session[Session Isolation<br/>Per-user namespacing]
        NoStore[No Credential Storage<br/>Connection strings only]
        OAuth[OAuth 2.0<br/>Delegated auth]
        Validation[Input Validation<br/>SQL injection prevention]
        Keys[API Key Protection<br/>Server-side only]
    end

    Request[Incoming Request]

    Request --> HTTPS
    HTTPS --> Session
    Session --> OAuth
    Session --> NoStore
    Session --> Validation
    Validation --> Keys
    Keys --> Response[Secure Response]

    style HTTPS fill:#e74c3c,color:#fff
    style Session fill:#e67e22,color:#fff
    style OAuth fill:#f39c12,color:#fff
    style Keys fill:#16a085,color:#fff
```

---

## Scalability Patterns

### Horizontal Scaling
- ✅ Stateless FastAPI servers
- ✅ Redis clustering for sessions
- ✅ Pinecone auto-scaling
- ✅ Vercel CDN for frontend

### Performance Optimization
- ✅ Connection pooling (PostgreSQL)
- ✅ Response caching (Analytics)
- ✅ Background workers (Forecasting)
- ✅ Lazy loading (Frontend)

### Resource Management
- ✅ Rate limiting (OpenAI)
- ✅ Query timeout (30s max)
- ✅ Session TTL (24hr auto-cleanup)
- ✅ Vector index optimization

---

## Data Flow Example: "What were my sales last month?"

```mermaid
sequenceDiagram
    actor User
    participant UI as Frontend
    participant API as Backend
    participant Agent as AI Agent
    participant DB as Data Source
    participant Vec as Pinecone
    participant LLM as OpenAI

    User->>UI: "What were my sales last month?"
    UI->>API: POST /chat {prompt, session_id}
    API->>Agent: Process query
    Agent->>Agent: Classify: Data-specific query
    Agent->>DB: Get sales data (last 30 days)
    DB-->>Agent: [sales records]
    Agent->>Vec: Find relevant context
    Vec-->>Agent: [best practices, insights]
    Agent->>LLM: Generate response
    Note over Agent,LLM: Combines data + knowledge
    LLM-->>Agent: Natural language answer
    Agent-->>API: Response + metadata
    API-->>UI: {content, citations, data}
    UI->>User: Display formatted response
```

---

## Error Handling Strategy

```mermaid
graph TB
    Request[API Request]

    Validate{Input Valid?}

    SessionCheck{Session Valid?}

    DataSource{Source Available?}

    Execute[Execute Query]

    Success[Return Success]

    ValidationError[400 Bad Request]
    SessionError[401 Unauthorized]
    SourceError[Use Mock Data<br/>Suggest Connection]
    QueryError[500 Internal Error<br/>Log & Alert]

    Request --> Validate
    Validate -->|No| ValidationError
    Validate -->|Yes| SessionCheck
    SessionCheck -->|No| SessionError
    SessionCheck -->|Yes| DataSource
    DataSource -->|Connected| Execute
    DataSource -->|Not Connected| SourceError
    Execute -->|Success| Success
    Execute -->|Failure| QueryError

    style Success fill:#27ae60,color:#fff
    style ValidationError fill:#e74c3c,color:#fff
    style SessionError fill:#e74c3c,color:#fff
    style SourceError fill:#f39c12,color:#fff
    style QueryError fill:#c0392b,color:#fff
```

---

## File Structure

```
peppasync-main/
├── peppasync-ai/                    # Frontend (Next.js)
│   ├── app/
│   │   ├── components/              # React components
│   │   │   ├── AIAssistant.tsx      # Conversational AI
│   │   │   ├── ShopifyConnection.tsx # Shopify OAuth
│   │   │   ├── DatabaseConnection.tsx
│   │   │   └── ForecastSettingsModal.tsx
│   │   ├── contexts/                # State management
│   │   │   ├── SessionContext.tsx   # Session + dataSource
│   │   │   └── AnalyticsContext.tsx # Data caching
│   │   ├── lib/
│   │   │   └── api.ts               # API client singleton
│   │   ├── shopify/callback/
│   │   │   └── page.tsx             # OAuth callback
│   │   └── page.tsx                 # Home dashboard
│   └── package.json
│
├── peppasync-langchain/             # Backend (FastAPI)
│   ├── app.py                       # Main FastAPI app
│   ├── lib/
│   │   ├── config.py                # Database manager
│   │   ├── shopify_service.py       # Shopify integration
│   │   ├── conversation_manager.py  # LangGraph orchestration
│   │   ├── query_classifier.py      # Query intent detection
│   │   ├── peppagenbi.py            # GenBISQL + RAG
│   │   └── redis_session.py         # Redis manager
│   └── requirements.txt
│
└── C4-ARCHITECTURE.md               # This file
```

---

## Next Steps & Roadmap

### Phase 1: Stability (Current)
- ✅ Single data source architecture
- ✅ Shopify OAuth integration
- ✅ Demand forecasting
- ✅ AI conversational interface

### Phase 2: Enhancement
- 🔄 Multi-warehouse support
- 🔄 Real-time data sync
- 🔄 Custom alert system
- 🔄 Export to Excel/PDF

### Phase 3: Scale
- ⏳ Multi-tenant architecture
- ⏳ Team collaboration
- ⏳ API for third-party integrations
- ⏳ Advanced ML models

---

**Documentation Version**: 1.0
**Last Updated**: 2025-01-19
**System**: PeppaSync Business Intelligence Platform
