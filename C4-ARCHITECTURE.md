# PeppaSync - C4 Architecture Model

## Level 1: System Context Diagram

```mermaid
C4Context
    title System Context Diagram for PeppaSync - Business Intelligence Platform

    Person(user, "Business Owner", "E-commerce business owner needing analytics and forecasting")

    System(peppasync, "PeppaSync Platform", "AI-powered business intelligence platform for e-commerce analytics, forecasting, and insights")

    System_Ext(shopify, "Shopify", "E-commerce platform providing order, product, and customer data")
    System_Ext(postgres, "PostgreSQL Database", "User's custom database with business data")
    System_Ext(connector, "External Connector", "OAuth service for Shopify integration (connector.fundam.ng)")
    System_Ext(openai, "OpenAI API", "GPT-4 for AI-powered analytics and insights")
    System_Ext(pinecone, "Pinecone", "Vector database for business knowledge storage")
    System_Ext(redis, "Redis", "Session and connection state management")

    Rel(user, peppasync, "Uses for analytics, forecasting, and AI insights", "HTTPS")
    Rel(peppasync, shopify, "Fetches orders, products, customers", "REST API")
    Rel(peppasync, postgres, "Queries business data", "PostgreSQL Protocol")
    Rel(peppasync, connector, "Manages Shopify OAuth flow", "REST API")
    Rel(peppasync, openai, "Generates insights and forecasts", "REST API")
    Rel(peppasync, pinecone, "Retrieves business knowledge", "REST API")
    Rel(peppasync, redis, "Stores session state", "Redis Protocol")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## Level 2: Container Diagram

```mermaid
C4Container
    title Container Diagram for PeppaSync Platform

    Person(user, "Business Owner", "Uses web interface for analytics")

    Container_Boundary(peppasync, "PeppaSync Platform") {
        Container(webapp, "Web Application", "Next.js 14, React, TypeScript", "Provides UI for analytics, forecasting, and AI chat")
        Container(api, "Backend API", "FastAPI, Python", "Provides REST API for analytics, forecasting, and data processing")
        Container(langchain, "AI Agent System", "LangChain, LangGraph", "Orchestrates AI workflows and business intelligence")
        ContainerDb(redis, "Session Store", "Redis", "Stores user sessions and connection state")
    }

    System_Ext(shopify, "Shopify API", "E-commerce data source")
    System_Ext(postgres, "User Database", "Custom business data")
    System_Ext(connector, "OAuth Connector", "Shopify OAuth service")
    System_Ext(openai, "OpenAI API", "LLM for insights")
    System_Ext(pinecone, "Pinecone", "Vector knowledge base")

    Rel(user, webapp, "Uses", "HTTPS")
    Rel(webapp, api, "Makes API calls", "JSON/HTTPS")
    Rel(api, langchain, "Delegates AI tasks", "Function calls")
    Rel(api, redis, "Reads/writes sessions", "Redis Protocol")
    Rel(api, shopify, "Fetches data via connector", "REST API")
    Rel(api, postgres, "Queries data", "SQL")
    Rel(api, connector, "OAuth flow", "REST API")
    Rel(langchain, openai, "Generates insights", "REST API")
    Rel(langchain, pinecone, "Retrieves knowledge", "REST API")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

---

## Level 3: Component Diagram - Frontend (Next.js App)

```mermaid
C4Component
    title Component Diagram - Frontend Web Application

    Container_Boundary(webapp, "Next.js Web Application") {
        Component(home, "Home Page", "React Component", "Main dashboard with connection management")
        Component(assistant, "AI Assistant", "React Component", "Conversational AI for business insights")
        Component(analytics, "Analytics Dashboard", "React Components", "Sales, orders, inventory visualizations")
        Component(forecast, "Forecast Settings", "React Component", "Configure economic events and supply chain")
        Component(shopifyconn, "Shopify Connection", "React Component", "OAuth flow and connection management")
        Component(dbconn, "Database Connection", "React Component", "PostgreSQL connection setup")

        Component(sessionctx, "Session Context", "React Context", "Manages user session and data source state")
        Component(analyticsctx, "Analytics Context", "React Context", "Manages analytics data fetching and caching")
        Component(apiservice, "API Service", "TypeScript Class", "Singleton API client for backend calls")
    }

    Container(api, "Backend API", "FastAPI", "REST endpoints")

    Rel(home, sessionctx, "Uses", "Context API")
    Rel(home, dbconn, "Renders", "React")
    Rel(home, shopifyconn, "Renders", "React")
    Rel(home, analytics, "Renders", "React")
    Rel(home, assistant, "Renders", "React")

    Rel(assistant, forecast, "Opens modal", "React")
    Rel(assistant, sessionctx, "Reads session", "Context API")
    Rel(assistant, apiservice, "Sends prompts", "HTTP")

    Rel(shopifyconn, apiservice, "Connect/disconnect", "HTTP")
    Rel(dbconn, apiservice, "Test/connect", "HTTP")

    Rel(analyticsctx, apiservice, "Fetches data", "HTTP")
    Rel(analyticsctx, sessionctx, "Reads session", "Context API")

    Rel(apiservice, api, "All API calls", "JSON/HTTPS")
    Rel(sessionctx, api, "Session validation", "JSON/HTTPS")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## Level 3: Component Diagram - Backend API

```mermaid
C4Component
    title Component Diagram - Backend API (FastAPI)

    Container_Boundary(api, "FastAPI Backend") {
        Component(chatep, "Chat Endpoint", "FastAPI Route", "Conversational BI with LangGraph")
        Component(analyticsep, "Analytics Endpoints", "FastAPI Routes", "Sales, orders, inventory data")
        Component(forecastep, "Forecast Endpoint", "FastAPI Route", "Demand forecasting with Prophet/ARIMA")
        Component(shopifyep, "Shopify Endpoints", "FastAPI Routes", "Connect, status, disconnect, callback")
        Component(dbep, "Database Endpoints", "FastAPI Routes", "Test, connect, disconnect, status")

        Component(convmgr, "Conversation Manager", "Python Class", "LangGraph workflow orchestration")
        Component(bizagent, "Business Agent", "LangGraph Agent", "Unified agent for all business queries")
        Component(queryclassifier, "Query Classifier", "Python Class", "Classifies query intent and data needs")
        Component(genbisql, "GenBISQL Engine", "Python Class", "Vector search and LLM processing")

        Component(dbmgr, "Database Manager", "Python Class", "Manages DB connections and queries")
        Component(shopifysvc, "Shopify Service", "Python Class", "External connector API integration")
        Component(redismgr, "Redis Manager", "Python Class", "Session persistence")
    }

    ContainerDb(redis, "Redis", "Session store")
    System_Ext(postgres, "User DB", "Custom database")
    System_Ext(connector, "OAuth Connector", "Shopify service")
    System_Ext(openai, "OpenAI", "LLM API")
    System_Ext(pinecone, "Pinecone", "Vector DB")

    Rel(chatep, convmgr, "Delegates to", "Function call")
    Rel(convmgr, bizagent, "Orchestrates", "LangGraph")
    Rel(bizagent, queryclassifier, "Classifies query", "Function call")
    Rel(bizagent, genbisql, "Retrieves data", "Function call")
    Rel(bizagent, openai, "Generates response", "REST API")

    Rel(analyticsep, dbmgr, "Fetches analytics", "Function call")
    Rel(forecastep, dbmgr, "Gets historical data", "Function call")
    Rel(forecastep, openai, "Forecast analysis", "REST API")

    Rel(shopifyep, shopifysvc, "Manages Shopify", "Function call")
    Rel(shopifysvc, connector, "OAuth & data fetch", "REST API")
    Rel(shopifysvc, dbmgr, "Stores orders", "Function call")

    Rel(dbep, dbmgr, "Connection mgmt", "Function call")
    Rel(dbmgr, postgres, "Queries data", "SQL")
    Rel(dbmgr, redis, "Stores connections", "Redis Protocol")

    Rel(genbisql, pinecone, "Vector search", "REST API")
    Rel(genbisql, dbmgr, "Gets data", "Function call")

    Rel(redismgr, redis, "Session ops", "Redis Protocol")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## Level 4: Code Diagram - Shopify Integration Flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend (ShopifyConnection)
    participant API as Backend API
    participant SS as ShopifyService
    participant Conn as External Connector
    participant Shop as Shopify OAuth
    participant Redis as Redis Store
    participant Callback as Callback Page

    U->>FE: Click "Connect Shopify"
    FE->>FE: Generate session_id
    FE->>FE: Save session_id to localStorage

    FE->>API: POST /shopify/connect<br/>{shop_name, session_id}
    API->>SS: connect_shopify_store()
    SS->>Conn: POST /connect/shopify<br/>{store, redirect_url}
    Conn-->>SS: {auth_url}
    SS-->>API: {success, auth_url}
    API->>Redis: Store connection info
    API-->>FE: {success, auth_url}

    FE->>FE: Save auth_url to localStorage
    FE->>FE: Display auth URL to user

    U->>Shop: User clicks auth_url & authorizes
    Shop->>Conn: OAuth callback with code
    Conn->>Conn: Exchange code for access_token
    Conn->>Callback: Redirect with success param

    Callback->>Callback: Get session_id from localStorage
    Callback->>Callback: Set data_source='shopify' in localStorage

    loop Poll every 1 second (max 30)
        Callback->>API: GET /shopify/status/{session_id}
        API->>Redis: Get connection info
        API->>SS: get_connection_id_by_shop()
        SS->>Conn: GET /connections
        Conn-->>SS: {connections: [{id, slug, ...}]}
        SS->>SS: Find connection_id by shop name
        SS-->>API: connection_id

        API->>SS: auto_sync_orders(shop_name, bearer_token)
        SS->>Conn: GET /shopify/fetch/orders/{connection_id}
        Conn-->>SS: {orders: [...]}
        SS->>Redis: Store orders
        SS-->>API: orders_count

        API->>Redis: Update last_sync timestamp
        API-->>Callback: {connected: true, orders_count}

        alt orders_count > 0
            Callback->>Callback: Stop polling, show success
            Callback->>Callback: window.location.href = '/'
        end
    end

    Note over Callback,Redis: Frontend reloads, SessionContext<br/>reads from localStorage
```

---

## Data Flow Diagram - Single Data Source Architecture

```mermaid
flowchart TD
    User[User/Business Owner]

    subgraph Frontend[Next.js Frontend]
        SessionCtx[Session Context<br/>sessionId + dataSource]
        AnalyticsCtx[Analytics Context]
        APIService[API Service Singleton]
    end

    subgraph Backend[FastAPI Backend]
        Endpoints[API Endpoints]
        DBMgr[Database Manager]
        ShopifySvc[Shopify Service]
    end

    subgraph DataSources[Data Sources - ONE AT A TIME]
        PG[(PostgreSQL<br/>User Database)]
        Shopify[(Shopify Store<br/>via Connector)]
        Mock[Mock Data<br/>Fallback]
    end

    subgraph Storage[State Management]
        Redis[(Redis<br/>Sessions & Connections)]
        LocalStorage[Browser localStorage<br/>session_id, data_source]
    end

    subgraph AI[AI Layer]
        Pinecone[(Pinecone<br/>Business Knowledge)]
        OpenAI[OpenAI GPT-4<br/>Insights Generation]
    end

    User -->|Interacts| Frontend
    SessionCtx -->|Tracks| LocalStorage
    SessionCtx -->|Determines| DataSources
    AnalyticsCtx -->|Fetches via| APIService
    APIService -->|HTTP Calls| Endpoints

    Endpoints -->|Routes to| DBMgr
    Endpoints -->|Routes to| ShopifySvc

    DBMgr -->|If dataSource=postgres| PG
    DBMgr -->|If dataSource=shopify| Shopify
    DBMgr -->|If dataSource=null| Mock
    DBMgr -->|Stores state| Redis

    ShopifySvc -->|OAuth & Fetch| Shopify
    ShopifySvc -->|Store orders| Redis

    Endpoints -->|Context retrieval| Pinecone
    Endpoints -->|Generate insights| OpenAI

    style DataSources fill:#f9f,stroke:#333,stroke-width:4px
    style SessionCtx fill:#bbf,stroke:#333,stroke-width:2px
```

---

## Deployment Architecture

```mermaid
graph TB
    subgraph Internet
        User[Users/Browsers]
    end

    subgraph Vercel[Vercel - Frontend Hosting]
        NextJS[Next.js App<br/>Static + SSR]
        EdgeFunc[Edge Functions]
    end

    subgraph Backend[Backend Infrastructure]
        API[FastAPI Server<br/>Port 8000]
        Worker[Background Workers<br/>Forecasting/Sync]
    end

    subgraph Databases
        Redis[(Redis Cloud<br/>Sessions)]
        UserDB[(User's PostgreSQL<br/>Customer Provided)]
    end

    subgraph External[External Services]
        Shopify[Shopify Stores]
        Connector[OAuth Connector<br/>connector.fundam.ng]
        OpenAI[OpenAI API]
        Pinecone[Pinecone Vector DB]
    end

    User -->|HTTPS| NextJS
    NextJS -->|API Calls| API
    API -->|Session Mgmt| Redis
    API -->|Query Data| UserDB
    API -->|OAuth Flow| Connector
    Connector -->|Fetch Data| Shopify
    API -->|AI Requests| OpenAI
    API -->|Vector Search| Pinecone
    Worker -->|Async Tasks| API

    style Vercel fill:#e1f5ff
    style Backend fill:#ffe1e1
    style Databases fill:#e1ffe1
    style External fill:#fff4e1
```

---

## Technology Stack

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **UI Library**: React 18
- **Styling**: Tailwind CSS
- **State Management**: React Context API
- **Charts**: Recharts
- **Icons**: Lucide React

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.9+
- **AI/ML**: LangChain, LangGraph
- **Forecasting**: Prophet, ARIMA (statsmodels)
- **LLM**: OpenAI GPT-4o-mini
- **Vector DB**: Pinecone
- **Session Store**: Redis

### Data Sources
- **PostgreSQL**: User's custom database
- **Shopify**: Via external OAuth connector
- **Mock Data**: Fallback for demos

### Infrastructure
- **Frontend Hosting**: Vercel
- **Backend Hosting**: Cloud VM / Docker
- **Session Storage**: Redis Cloud
- **Vector Storage**: Pinecone Cloud

---

## Key Design Patterns

1. **Single Data Source Architecture**: Only ONE data source active per session (PostgreSQL OR Shopify OR Mock)

2. **Session-Based Isolation**: Each user has isolated session with dedicated data source connection

3. **Frontend-Tracked State**: Connection state tracked in localStorage + React Context, not backend polling

4. **OAuth Delegation**: Shopify OAuth handled by external connector service, not direct integration

5. **AI Agent Pattern**: LangGraph orchestrates unified business agent for all query types

6. **Vector-Augmented Generation**: Business knowledge from Pinecone combined with user data

7. **Automatic Fallback**: Mock data used when no data source connected

8. **Redis-First Session**: All sessions and connections persisted in Redis with 24hr TTL

---

## Security Considerations

1. **No Credential Storage**: Database credentials never stored, only connection strings in Redis
2. **Session Isolation**: Each session completely isolated, no cross-contamination
3. **OAuth Best Practices**: External connector handles token exchange, not direct app
4. **API Key Protection**: OpenAI and Pinecone keys server-side only
5. **HTTPS Only**: All external communication encrypted
6. **Input Validation**: All user inputs validated before DB queries
7. **SQL Injection Protection**: Parameterized queries only, no string concatenation

---

## Scalability Considerations

1. **Stateless Backend**: API servers can scale horizontally
2. **Redis Clustering**: Session store can be clustered for high availability
3. **Pinecone Sharding**: Vector DB scales with data volume
4. **Async Processing**: Background workers for heavy forecasting tasks
5. **CDN Distribution**: Frontend static assets served via Vercel CDN
6. **Connection Pooling**: Database connections pooled and reused
7. **Rate Limiting**: OpenAI API calls throttled to prevent quota exhaustion

