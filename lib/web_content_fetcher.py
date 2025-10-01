"""
Web Content Fetcher for Marketing and Sales Insights
Scrapes articles and content from top industry voices
"""
import logging
import asyncio
import aiohttp
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from urllib.parse import urljoin, urlparse
import time

logger = logging.getLogger(__name__)

class MarketingContentFetcher:
    """Fetch and process marketing/sales content from top industry sources"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Top marketing and sales content sources
        self.sources = {
            "gary_vaynerchuk": {
                "base_url": "https://www.garyvaynerchuk.com",
                "blog_paths": ["/blog", "/articles"],
                "author": "Gary Vaynerchuk",
                "expertise": ["social_media_marketing", "entrepreneurship", "brand_building"]
            },
            "neil_patel": {
                "base_url": "https://neilpatel.com",
                "blog_paths": ["/blog"],
                "author": "Neil Patel",
                "expertise": ["seo", "content_marketing", "conversion_optimization"]
            },
            "hubspot": {
                "base_url": "https://blog.hubspot.com",
                "blog_paths": ["/marketing", "/sales"],
                "author": "HubSpot",
                "expertise": ["inbound_marketing", "sales_enablement", "crm"]
            },
            "seth_godin": {
                "base_url": "https://seths.blog",
                "blog_paths": [""],
                "author": "Seth Godin",
                "expertise": ["marketing_philosophy", "purple_cow", "permission_marketing"]
            },
            "salesforce": {
                "base_url": "https://www.salesforce.com/resources/articles/sales",
                "blog_paths": [""],
                "author": "Salesforce",
                "expertise": ["sales_methodology", "crm_best_practices", "sales_analytics"]
            }
        }
    
    async def fetch_latest_insights(self, max_articles_per_source: int = 5) -> List[Document]:
        """Fetch latest marketing and sales insights from all sources"""
        all_documents = []
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
        ) as session:
            
            tasks = []
            for source_name, source_config in self.sources.items():
                task = self._fetch_source_content(
                    session, source_name, source_config, max_articles_per_source
                )
                tasks.append(task)
            
            # Execute all fetching tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for source_name, result in zip(self.sources.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching from {source_name}: {result}")
                else:
                    all_documents.extend(result)
                    logger.info(f"Fetched {len(result)} documents from {source_name}")
        
        logger.info(f"Total documents fetched: {len(all_documents)}")
        return all_documents
    
    async def _fetch_source_content(
        self, 
        session: aiohttp.ClientSession, 
        source_name: str, 
        source_config: Dict, 
        max_articles: int
    ) -> List[Document]:
        """Fetch content from a specific source"""
        documents = []
        
        try:
            # Get article URLs
            article_urls = await self._get_article_urls(
                session, source_config, max_articles
            )
            
            # Fetch content from each article
            for url in article_urls[:max_articles]:
                try:
                    article_doc = await self._fetch_article_content(
                        session, url, source_config
                    )
                    if article_doc:
                        # Split long articles into chunks
                        chunks = self.text_splitter.split_documents([article_doc])
                        documents.extend(chunks)
                        
                        # Rate limiting
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.warning(f"Error fetching article {url}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error fetching from {source_name}: {e}")
        
        return documents
    
    async def _get_article_urls(
        self, 
        session: aiohttp.ClientSession, 
        source_config: Dict, 
        max_articles: int
    ) -> List[str]:
        """Extract article URLs from blog listing pages"""
        urls = []
        
        for blog_path in source_config["blog_paths"]:
            try:
                blog_url = urljoin(source_config["base_url"], blog_path)
                
                async with session.get(blog_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Common selectors for article links
                        article_selectors = [
                            'article a[href]',
                            '.post-title a[href]',
                            '.entry-title a[href]',
                            'h2 a[href]',
                            'h3 a[href]',
                            '.blog-post a[href]',
                            '.article-link[href]'
                        ]
                        
                        for selector in article_selectors:
                            links = soup.select(selector)
                            for link in links:
                                href = link.get('href')
                                if href:
                                    full_url = urljoin(source_config["base_url"], href)
                                    if self._is_valid_article_url(full_url, source_config["base_url"]):
                                        urls.append(full_url)
                                        if len(urls) >= max_articles:
                                            break
                            
                            if len(urls) >= max_articles:
                                break
                        
                        if len(urls) >= max_articles:
                            break
                            
            except Exception as e:
                logger.warning(f"Error getting URLs from {blog_url}: {e}")
                continue
        
        return list(set(urls))  # Remove duplicates
    
    def _is_valid_article_url(self, url: str, base_url: str) -> bool:
        """Check if URL is likely a valid article"""
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(base_url)
            
            # Must be from same domain
            if parsed.netloc != base_parsed.netloc:
                return False
            
            # Skip certain paths
            skip_patterns = [
                '/tag/', '/category/', '/author/', '/page/', 
                '/search/', '/archive/', '/contact', '/about',
                '.pdf', '.jpg', '.png', '.gif', '#'
            ]
            
            return not any(pattern in url.lower() for pattern in skip_patterns)
            
        except Exception:
            return False
    
    async def _fetch_article_content(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        source_config: Dict
    ) -> Optional[Document]:
        """Fetch and extract content from a single article"""
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title = self._extract_title(soup)
                
                # Extract main content
                content = self._extract_main_content(soup)
                
                if not content or len(content.strip()) < 200:
                    return None
                
                # Clean and validate content
                cleaned_content = self._clean_content(content)
                
                if not self._is_marketing_sales_content(cleaned_content):
                    return None
                
                # Create document with metadata
                metadata = {
                    "source": source_config["author"],
                    "url": url,
                    "title": title,
                    "expertise": source_config["expertise"],
                    "content_type": "web_article",
                    "fetched_date": time.strftime("%Y-%m-%d"),
                    "estimated_read_time": len(cleaned_content.split()) // 200  # ~200 WPM
                }
                
                return Document(
                    page_content=f"Title: {title}\n\n{cleaned_content}",
                    metadata=metadata
                )
                
        except Exception as e:
            logger.warning(f"Error fetching article content from {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        selectors = ['h1', '.entry-title', '.post-title', 'title']
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        return "Untitled Article"
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main article content"""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try content selectors in order of preference
        content_selectors = [
            '.entry-content',
            '.post-content', 
            '.article-content',
            '.content',
            'article',
            '.main-content',
            '#content'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(separator=' ', strip=True)
        
        # Fallback to body content
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ', strip=True)
        
        return soup.get_text(separator=' ', strip=True)
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common noise
        noise_patterns = [
            r'Share this article.*?$',
            r'Follow us on.*?$',
            r'Subscribe to.*?$',
            r'Comments.*?$',
            r'Related articles.*?$'
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _is_marketing_sales_content(self, content: str) -> bool:
        """Check if content is relevant to marketing/sales"""
        content_lower = content.lower()
        
        marketing_keywords = [
            'marketing', 'sales', 'customer', 'conversion', 'roi', 'revenue',
            'brand', 'advertising', 'campaign', 'lead', 'funnel', 'engagement',
            'social media', 'content marketing', 'email marketing', 'seo',
            'analytics', 'metrics', 'kpi', 'retention', 'acquisition',
            'personalization', 'segmentation', 'targeting', 'optimization'
        ]
        
        # Content should have at least 3 marketing-related keywords
        keyword_count = sum(1 for keyword in marketing_keywords if keyword in content_lower)
        
        return keyword_count >= 3 and len(content) >= 500

    async def fetch_specific_topics(self, topics: List[str], max_results: int = 10) -> List[Document]:
        """Fetch content focused on specific marketing/sales topics"""
        # This could be enhanced with topic-specific search queries
        # For now, filter existing results by topic relevance
        
        all_content = await self.fetch_latest_insights(max_articles_per_source=3)
        
        topic_content = []
        for doc in all_content:
            content_lower = doc.page_content.lower()
            if any(topic.lower() in content_lower for topic in topics):
                doc.metadata["matched_topics"] = [
                    topic for topic in topics 
                    if topic.lower() in content_lower
                ]
                topic_content.append(doc)
                
                if len(topic_content) >= max_results:
                    break
        
        logger.info(f"Found {len(topic_content)} documents matching topics: {topics}")
        return topic_content