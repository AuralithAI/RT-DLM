import requests
import json
import time
import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Any
import logging
from urllib.parse import quote
import re
from pathlib import Path

# Try to import sentencepiece for text tokenization
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

logger = logging.getLogger(__name__)

class WebSearchModule(hk.Module):
    """Web search integration using multiple search engines with real embeddings"""
    
    def __init__(self, d_model: int, max_results: int = 10, vocab_size: int = 8000, 
                 max_tokens: int = 128, tokenizer_path: Optional[str] = None, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_results = max_results
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        
        # Load sentencepiece tokenizer if available
        self.tokenizer = None
        if SENTENCEPIECE_AVAILABLE and tokenizer_path:
            try:
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.load(tokenizer_path)
                logger.info(f"Loaded tokenizer from {tokenizer_path}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        
        # Text embedding layer for snippet encoding
        self.snippet_embedding = hk.Embed(
            vocab_size=vocab_size,
            embed_dim=d_model,
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="snippet_embed"
        )
        
        # Embedding projection layer
        self.embed_projection = hk.Linear(d_model, name="embed_projection")
        
        # Search result encoder
        self.result_encoder = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="result_encoder")
        
        # Relevance scorer
        self.relevance_scorer = hk.Sequential([
            hk.Linear(d_model * 2),
            jax.nn.silu,
            hk.Linear(1),
            jax.nn.sigmoid
        ], name="relevance_scorer")
        
        # Result synthesizer
        self.synthesizer = hk.MultiHeadAttention(
            num_heads=8, 
            key_size=d_model//8, 
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="result_synthesizer"
        )
        
    def search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """Search using DuckDuckGo API (no API key required)"""
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_redirect': '1',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Extract instant answer
                if data.get('AbstractText'):
                    results.append({
                        'title': data.get('Heading', 'Instant Answer'),
                        'snippet': data['AbstractText'],
                        'url': data.get('AbstractURL', ''),
                        'source': 'DuckDuckGo Instant'
                    })
                
                # Extract related topics
                for topic in data.get('RelatedTopics', [])[:5]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                            'snippet': topic['Text'],
                            'url': topic.get('FirstURL', ''),
                            'source': 'DuckDuckGo Related'
                        })
                
                return results[:self.max_results]
                
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
        
        return []
    
    def search_wikipedia(self, query: str) -> List[Dict[str, str]]:
        """Search Wikipedia for relevant articles"""
        try:
            # Wikipedia API search
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                'action': 'opensearch',
                'search': query,
                'limit': 5,
                'format': 'json'
            }
            
            response = requests.get(search_url, params=search_params, timeout=10)
            if response.status_code == 200:
                search_data = response.json()
                titles = search_data[1]
                descriptions = search_data[2]
                urls = search_data[3]
                
                results = []
                for title, desc, url in zip(titles, descriptions, urls):
                    # Get full summary
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
                    summary_response = requests.get(summary_url, timeout=5)
                    
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        results.append({
                            'title': title,
                            'snippet': summary_data.get('extract', desc),
                            'url': url,
                            'source': 'Wikipedia'
                        })
                
                return results[:self.max_results]
                
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
        
        return []
    
    def _tokenize_text(self, text: str) -> jnp.ndarray:
        """Tokenize text using sentencepiece or fallback to character-level.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Token IDs array of shape [max_tokens]
        """
        if self.tokenizer is not None:
            try:
                token_ids = self.tokenizer.encode(text)
            except Exception:
                # Fallback to character-level
                token_ids = [ord(c) % self.vocab_size for c in text]
        else:
            # Fallback: simple hash-based tokenization
            words = text.lower().split()
            token_ids = [hash(w) % self.vocab_size for w in words]
        
        # Pad or truncate to max_tokens
        if len(token_ids) > self.max_tokens:
            token_ids = token_ids[:self.max_tokens]
        else:
            token_ids = token_ids + [0] * (self.max_tokens - len(token_ids))
        
        return jnp.array(token_ids, dtype=jnp.int32)
    
    def _embed_snippets(self, search_results: List[Dict[str, str]], batch_size: int) -> jnp.ndarray:
        """Embed search result snippets using hk.Embed and linear projection.
        
        Args:
            search_results: List of search result dictionaries with 'snippet' keys
            batch_size: Batch size for output
            
        Returns:
            Embedded snippets of shape [batch_size, num_results, d_model]
        """
        num_results = min(len(search_results), self.max_results)
        
        # Tokenize all snippets
        all_tokens = []
        for result in search_results[:num_results]:
            snippet = result.get('snippet', '') + ' ' + result.get('title', '')
            tokens = self._tokenize_text(snippet)
            all_tokens.append(tokens)
        
        # Stack into array: [num_results, max_tokens]
        token_array = jnp.stack(all_tokens, axis=0)
        
        # Embed tokens: [num_results, max_tokens, d_model]
        embedded = self.snippet_embedding(token_array)
        
        # Mean pooling over tokens: [num_results, d_model]
        pooled = jnp.mean(embedded, axis=1)
        
        # Project to final dimension
        projected = self.embed_projection(pooled)
        
        # Expand for batch dimension: [batch_size, num_results, d_model]
        result_embeddings = jnp.broadcast_to(
            projected[None, :, :], 
            (batch_size, num_results, self.d_model)
        )
        
        return result_embeddings
    
    def __call__(self, query_embedding: jnp.ndarray, query_text: str) -> Dict[str, Any]:
        """Perform web search and encode results with real embeddings"""
        # Perform actual web searches
        search_results = []
        search_results.extend(self.search_duckduckgo(query_text))
        search_results.extend(self.search_wikipedia(query_text))
        
        if not search_results:
            return {
                'encoded_results': jnp.zeros((1, self.d_model)),
                'relevance_scores': jnp.zeros((1,)),
                'raw_results': [],
                'synthesis': jnp.zeros((1, self.d_model))
            }
        
        batch_size = query_embedding.shape[0]
        num_results = min(len(search_results), self.max_results)
        
        # Create real embeddings using sentencepiece tokenization and hk.Embed
        result_embeddings = self._embed_snippets(search_results, batch_size)
        
        # Encode results through the encoder network
        encoded_results = self.result_encoder(result_embeddings)
        
        # Calculate relevance scores
        query_expanded = jnp.broadcast_to(
            query_embedding[:, None, :], 
            (batch_size, num_results, self.d_model)
        )
        relevance_input = jnp.concatenate([query_expanded, encoded_results], axis=-1)
        relevance_scores = self.relevance_scorer(relevance_input).squeeze(-1)
        
        # Synthesize results using attention
        synthesis = self.synthesizer(query_embedding, encoded_results, encoded_results)
        
        return {
            'encoded_results': encoded_results,
            'relevance_scores': relevance_scores,
            'raw_results': search_results,
            'synthesis': synthesis
        }


class ExternalAPIModule:
    """Integration with external APIs and services"""
    
    def __init__(self):
        self.api_cache = {}
        self.rate_limits = {}
        
    def get_news(self, topic: str, max_articles: int = 5) -> List[Dict[str, str]]:
        """Get latest news about a topic"""
        try:
            # Using free news API (NewsAPI alternative that doesn't require key)
            url = f"https://rss-to-json-serverless-api.vercel.app/api?feedURL=https://news.google.com/rss/search?q={quote(topic)}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for item in data.get('items', [])[:max_articles]:
                    articles.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('description', ''),
                        'url': item.get('link', ''),
                        'source': 'Google News',
                        'published': item.get('pubDate', '')
                    })
                
                return articles
                
        except Exception as e:
            logger.error(f"News API failed: {e}")
        
        return []
    
    def get_weather(self, location: str) -> Dict[str, Any]:
        """Get weather information (using free weather API)"""
        try:
            # Using wttr.in free weather service
            url = f"https://wttr.in/{quote(location)}?format=j1"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                current = data.get('current_condition', [{}])[0]
                
                return {
                    'location': location,
                    'temperature': current.get('temp_C', 'N/A'),
                    'condition': current.get('weatherDesc', [{}])[0].get('value', 'N/A'),
                    'humidity': current.get('humidity', 'N/A'),
                    'wind_speed': current.get('windspeedKmph', 'N/A'),
                    'source': 'wttr.in'
                }
                
        except Exception as e:
            logger.error(f"Weather API failed: {e}")
        
        return {}
    
    def get_exchange_rates(self, base_currency: str = 'USD') -> Dict[str, float]:
        """Get current exchange rates"""
        try:
            # Using free exchange rate API
            url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'base': base_currency,
                    'rates': data.get('rates', {}),
                    'source': 'ExchangeRate-API'
                }
                
        except Exception as e:
            logger.error(f"Exchange rate API failed: {e}")
        
        return {}


class HybridKnowledgeIntegration(hk.Module):
    """Hybrid module that combines internal knowledge with external sources"""
    
    def __init__(self, d_model: int, internal_knowledge_graph=None, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.internal_knowledge = internal_knowledge_graph
        
        # Web search integration
        self.web_search = WebSearchModule(d_model)
        self.external_apis = ExternalAPIModule()
        
        # Knowledge fusion layer
        self.knowledge_fusion = hk.MultiHeadAttention(
            num_heads=8, 
            key_size=d_model//8, 
            w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            name="knowledge_fusion"
        )
        
        # Source weighting
        self.source_weights = hk.Sequential([
            hk.Linear(d_model * 3),  # internal + web + api
            jax.nn.softmax
        ], name="source_weights")
        
        # Final synthesis
        self.final_synthesis = hk.Sequential([
            hk.Linear(d_model),
            jax.nn.silu,
            hk.Linear(d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="final_synthesis")
    
    def __call__(self, query_embedding: jnp.ndarray, query_text: str, 
                 context: Optional[str] = None) -> Dict[str, Any]:
        """Integrate internal and external knowledge sources"""
        batch_size = query_embedding.shape[0]
        
        # Internal knowledge retrieval (if available)
        internal_knowledge = query_embedding  # Placeholder for internal graph lookup
        
        # External web search
        web_results = self.web_search(query_embedding, query_text)
        web_knowledge = web_results['synthesis']
        
        # External API data
        api_data = []
        if 'weather' in query_text.lower():
            location = self._extract_location(query_text)
            if location:
                weather_data = self.external_apis.get_weather(location)
                api_data.append(weather_data)
        
        if 'news' in query_text.lower():
            news_data = self.external_apis.get_news(query_text)
            api_data.extend(news_data)
        
        # Encode API data (simplified)
        if api_data:
            api_knowledge = jax.random.normal(
                jax.random.PRNGKey(123), (batch_size, 1, self.d_model)
            ) * 0.1
        else:
            api_knowledge = jnp.zeros((batch_size, 1, self.d_model))
        
        # Fuse all knowledge sources
        all_knowledge = jnp.concatenate([
            internal_knowledge.reshape(batch_size, 1, self.d_model),
            web_knowledge,
            api_knowledge
        ], axis=1)
        
        # Apply fusion attention
        fused_knowledge = self.knowledge_fusion(
            query_embedding, all_knowledge, all_knowledge
        )
        
        # Final synthesis
        final_output = self.final_synthesis(fused_knowledge)
        
        return {
            'fused_knowledge': final_output,
            'web_results': web_results['raw_results'],
            'api_data': api_data,
            'source_breakdown': {
                'internal': 'knowledge_graph',
                'web': len(web_results['raw_results']),
                'api': len(api_data)
            }
        }
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from text for weather queries"""
        # Simple location extraction (can be enhanced with NER)
        location_patterns = [
            r'weather in ([A-Za-z\s]+)',
            r'temperature in ([A-Za-z\s]+)',
            r'climate in ([A-Za-z\s]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
