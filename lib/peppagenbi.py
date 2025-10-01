import os
import re
import json
import logging
from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import uuid
import json

# Import centralized configuration
from .config import LLMManager, AppConfig, DatabaseManager
from .business_analyzer import BusinessAnalyzer

load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenBISQL:
    
    def __init__(self):
        self.llm = LLMManager.get_chat_llm()
        self.embeddings = LLMManager.get_embeddings()
        self.vector_store = None
        self._needs_population = False

        # Database configuration (using mock data for now)
        self.database_config = AppConfig.DATABASE_CONFIG
        self.use_mock_data = False

        # Initialize Pinecone vector store
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize vector store with Pinecone"""
        try:
            if AppConfig.PINECONE_API_KEY:
                logger.info("Initializing Pinecone Vector Store")
                self._init_pinecone_vector_store()
            else:
                logger.warning("Pinecone Vector Store not configured. Vector search will be disabled.")
                self.vector_store = None

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self.vector_store = None

    def _init_pinecone_vector_store(self):
        """Initialize Pinecone vector store"""
        try:
            from pinecone import Pinecone
            from langchain_pinecone import PineconeVectorStore

            # Initialize Pinecone client
            pc = Pinecone(api_key=AppConfig.PINECONE_API_KEY)

            # Check if index exists, handle dimension mismatch
            index_name = AppConfig.PINECONE_INDEX_NAME
            existing_indexes = pc.list_indexes()

            index_exists = index_name in [index.name for index in existing_indexes]

            if index_exists:
                # Check if dimensions match
                index_info = pc.describe_index(index_name)
                existing_dimension = index_info['dimension']

                if existing_dimension != AppConfig.VECTOR_DIMENSION:
                    logger.warning(f"Index {index_name} has dimension {existing_dimension}, expected {AppConfig.VECTOR_DIMENSION}")
                    # Delete and recreate with correct dimensions
                    logger.info(f"Deleting existing index: {index_name}")
                    pc.delete_index(index_name)
                    index_exists = False

            if not index_exists:
                logger.info(f"Creating Pinecone index: {index_name} with dimension {AppConfig.VECTOR_DIMENSION}")
                pc.create_index(
                    name=index_name,
                    dimension=AppConfig.VECTOR_DIMENSION,
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                # Wait for index to be ready
                import time
                time.sleep(5)

            # Initialize the vector store
            self.vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings,
                pinecone_api_key=AppConfig.PINECONE_API_KEY
            )

            # Check if we need to populate with initial data
            index = pc.Index(index_name)
            stats = index.describe_index_stats()

            if stats['total_vector_count'] == 0:
                logger.info("Pinecone index is empty, will populate with expert knowledge and web content on first use...")
                self._needs_population = True
            else:
                logger.info(f"Pinecone index loaded with {stats['total_vector_count']} vectors")
                self._needs_population = False

            logger.info("Pinecone Vector Store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone vector store: {e}")
            raise

    async def _ensure_knowledge_base_populated(self):
        """Ensure knowledge base is populated before first use"""
        if self._needs_population and self.vector_store:
            logger.info("Populating knowledge base on first use...")
            await self._populate_pinecone_with_expert_and_web_content()
            self._needs_population = False

    async def _populate_pinecone_with_expert_and_web_content(self):
        """Populate Pinecone with expert knowledge + fresh web content (runs once on startup)"""
        try:
            from .expert_knowledge import ExpertKnowledgeBase
            
            logger.info("Fetching expert knowledge and latest web content...")
            
            # Get enhanced knowledge (curated + web content)
            expert_documents = await ExpertKnowledgeBase.get_enhanced_knowledge_with_web_content(
                include_web_content=True,
                max_web_articles_per_source=AppConfig.MAX_WEB_ARTICLES_PER_SOURCE
            )
            
            # Add documents to Pinecone
            self.vector_store.add_documents(expert_documents)
            logger.info(f"Successfully populated Pinecone with {len(expert_documents)} documents (expert + web content)")
            
        except Exception as e:
            logger.error(f"Error populating Pinecone with expert and web content: {e}")
            # Fallback to curated content only
            try:
                from .expert_knowledge import ExpertKnowledgeBase
                fallback_documents = ExpertKnowledgeBase.get_all_expert_knowledge()
                self.vector_store.add_documents(fallback_documents)
                logger.info(f"Fallback: Populated Pinecone with {len(fallback_documents)} curated documents only")
            except Exception as fallback_error:
                logger.error(f"Fallback population also failed: {fallback_error}")
                raise
    def _extract_json(self, text):
        """Extract and parse JSON from text response"""
        try:
            return json.loads(text)  # Direct JSON parsing
        except json.JSONDecodeError:
            logger.info("Direct JSON parsing failed. Attempting regex extraction.")

        try:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error("Failed to extract JSON: %s", str(e))

        logger.warning("Returning raw text as JSON parsing failed.")
        return {"raw_response": text}

    def split_response(self, response: str):
        """
        Splits model response into pairs of insight and code blocks.
        Returns a list of dicts: [{"insight": ..., "plot_code": ...}, ...]
        """
        lines = response.strip().split("\n")
        results = []
        current_insight = []
        current_code = []
        in_code_block = False

        def is_code_line(line):
            stripped = line.strip()
            return (
                stripped.startswith("px.") or
                stripped.startswith("sns.") or
                stripped.startswith("df.") or
                "plt" in stripped or
                "sns" in stripped or
                stripped.startswith("fig =") or
                stripped.startswith("plt.")
            )

        for line in lines:
            if is_code_line(line):
                in_code_block = True
                current_code.append(line)
            else:
                if in_code_block:
                    # End of code block, pair with previous insight
                    results.append({
                        "insight": "\n".join(current_insight).strip(),
                        "plot_code": "\n".join(current_code).strip()
                    })
                    current_insight = []
                    current_code = []
                    in_code_block = False
                current_insight.append(line)

        # Handle trailing code block or insight
        if current_code or current_insight:
            results.append({
                "insight": "\n".join(current_insight).strip(),
                "plot_code": "\n".join(current_code).strip()
            })

        # Remove empty pairs
        return [block for block in results if block["insight"] or block["plot_code"]]

    async def invoke_llm(self, user_message: str, system_message: str = None) -> str:
        """Invoke OpenAI LLM with user and system messages"""
        try:
            messages = []
            
            if system_message:
                messages.append(SystemMessage(content=system_message))
            
            messages.append(HumanMessage(content=user_message))
            
            response = await self.llm.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error("Error invoking LLM: %s", str(e))
            return f"Error: {str(e)}"

    async def retrieve_relevant_data(self, query: str, session_id: str = None, limit: int = 5) -> List[Dict]:
        """Retrieve relevant data from vector store and database"""
        try:
            # Ensure knowledge base is populated
            await self._ensure_knowledge_base_populated()
            
            relevant_data = []

            if self.vector_store:
                # Get relevant documents from Pinecone vector store
                docs = self.vector_store.similarity_search(query, k=limit)
                for doc in docs:
                    relevant_data.append({
                        "type": "knowledge",
                        "content": doc.page_content,
                        "metadata": getattr(doc, 'metadata', {})
                    })

            # Try to get actual database data (with session_id)
            db_data = await self._get_database_data(query, session_id)
            if db_data:
                relevant_data.extend(db_data)

            return relevant_data
            
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return []

    async def _get_database_data(self, query: str, session_id: str = None) -> List[Dict]:
        """Get relevant data from database - falls back to mock if no connection"""
        try:
            # Determine query type from the query
            query_lower = query.lower()
            query_type = "general"

            if any(word in query_lower for word in ["sales", "revenue", "sold"]):
                query_type = "sales_data"
            elif any(word in query_lower for word in ["inventory", "stock", "available"]):
                query_type = "inventory_data"
            elif any(word in query_lower for word in ["customer", "demographic", "user"]):
                query_type = "customer_data"

            # Use DatabaseManager to get data (will fallback to mock if no connection)
            data = DatabaseManager.get_data(session_id=session_id, query_type=query_type, use_mock=False)
            return data

        except Exception as e:
            logger.error(f"Error getting database data: {e}")
            return []

    async def retrieve_and_generate(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Main retrieve and generate function using LangChain"""
        try:
            # Step 1: Retrieve relevant data
            retrieved_data = await self.retrieve_relevant_data(query, session_id)
            
            # Step 2: Format retrieved data for LLM
            context = self._format_retrieved_data(retrieved_data)
            
            
            # Step 3: Create prompt template
            prompt_template = """
You are a Business Intelligence AI Assistant specializing in Conversational Generative BI, AI, Sales and Marketing Analytics.

Context from knowledge base and data:
{context}

User Question: {query}

Instructions:
- Use the provided context to answer the user's question
- If the data shows monetary values, they are in US Dollars ($)
- Provide diagnostic (why did this happen), predictive (what will happen), and prescriptive (what to do) insights
- Be concise but comprehensive
- If you cannot answer from the provided context, say "I don't have enough information to answer that question"

Response:
"""
            
            formatted_prompt = prompt_template.format(
                context=context,
                query=query
            )
            
            # Step 4: Generate response
            system_message = "You are a data analyst helping business stakeholders understand their data through clear, actionable insights."
            response = await self.invoke_llm(formatted_prompt, system_message)
            
            return {
                'output': response,
                'citations': retrieved_data
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_and_generate: {e}")
            return {
                'output': f"I encountered an error processing your request: {str(e)}",
                'citations': []
            }

    async def retrieve_and_visualize(self, prompt: str, session_id: str = None) -> Dict[str, Any]:
        """Generate visualization code and insights"""
        try:
            # Step 1: Get relevant data
            retrieved_data = await self.retrieve_relevant_data(prompt, session_id)
            
            # Step 2: Format data for visualization prompt
            data_context = self._format_retrieved_data(retrieved_data)
            
            # Step 3: Create visualization prompt
            viz_prompt = f"""
You are a Python data visualization expert using pandas and plotly.express.

Business Context: {data_context}

User Request: {prompt}

Generate Python code that creates appropriate charts for this request. 

Requirements:
- Use only plotly.express (px)
- Assume data is already in a pandas DataFrame called 'df'
- Do NOT include import statements or df creation
- Start directly with px. chart code
- Generate 2-3 different visualizations if the request suggests multiple insights
- Begin with a technical summary, then provide the code
- Do NOT use markdown formatting or backticks

Example format:
Technical Summary: This analysis shows sales trends by category with seasonal patterns.

fig1 = px.bar(df, x='category', y='sales', title='Sales by Category')
fig1.show()

fig2 = px.line(df, x='date', y='sales', color='category', title='Sales Trends Over Time')  
fig2.show()
"""
            
            system_message = "You are a data visualization expert. Provide clean, executable Python code for business intelligence dashboards."
            
            response = await self.invoke_llm(viz_prompt, system_message)
            
            # Step 4: Split response into insights and code
            processed_response = self.split_response(response)
            
            return {
                'plot': processed_response,
                'data': retrieved_data
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_and_visualize: {e}")
            return {
                'plot': [{"insight": f"Error generating visualization: {str(e)}", "plot_code": ""}],
                'data': []
            }

    def _format_retrieved_data(self, data: List[Dict]) -> str:
        """Format retrieved data for LLM consumption"""
        if not data:
            return "No relevant data found."
        
        formatted_parts = []
        
        for item in data:
            if item.get('type') == 'knowledge':
                formatted_parts.append(f"Knowledge: {item['content']}")
            elif item.get('type') == 'sales_data':
                product_name = item.get('product_name', 'Unknown Product')
                sales_amount = item.get('sales_amount', 0)
                units_sold = item.get('units_sold', 0)
                category = item.get('category', 'Unknown')
                date_info = item.get('date', item.get('sale_date', 'N/A'))
                
                formatted_parts.append(
                    f"Sales Record: {product_name} - "
                    f"${sales_amount:,} revenue, {units_sold} units sold, "
                    f"Category: {category}, Date: {date_info}"
                )
            elif item.get('type') == 'inventory_data':
                formatted_parts.append(
                    f"Inventory: {item['product_name']} - "
                    f"Current stock: {item['current_stock']}, "
                    f"Reorder level: {item['reorder_level']}, "
                    f"Category: {item['category']}"
                )
        
        return "\n\n".join(formatted_parts)

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and configuration"""
        return {
            "llm_model": "gpt-4o-mini",
            "vector_store_initialized": self.vector_store is not None,
            "database_connected": True,  # Mock for now
            "embeddings_model": "text-embedding-ada-002"
        }
