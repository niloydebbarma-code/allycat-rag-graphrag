"""
Knowledge Retrieval Module - Phase C (Steps 6-8)

Performs community search and data extraction using graph database structures.
Handles community retrieval, data extraction, and initial answer generation.
"""

import logging
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from .setup import GraphRAGSetup
from .query_preprocessing import DriftRoutingResult


@dataclass
class CommunityResult:
    """Enhanced community result with comprehensive properties."""
    community_id: str
    similarity_score: float
    summary: str
    key_entities: List[str]
    member_ids: List[str]  # Direct member access
    modularity_score: float  # Community quality
    level: int
    internal_edges: int
    member_count: int
    centrality_stats: Dict[str, float]  # Aggregated centrality measures
    confidence_score: float
    search_index: str  # Optimized search key
    termination_criteria: Dict[str, Any]


@dataclass
class EntityResult:
    """Entity result with attributes from graph database."""
    entity_id: str
    name: str
    content: str
    confidence: float
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    community_id: str
    node_type: str


@dataclass
class RelationshipResult:
    """Relationship result with graph database attributes."""
    start_node: str
    end_node: str
    relationship_type: str
    confidence: float


class CommunitySearchEngine:
    """Knowledge retrieval engine for community search and entity extraction."""
    
    def __init__(self, setup: GraphRAGSetup):
        self.setup = setup
        self.neo4j_conn = setup.neo4j_conn
        self.config = setup.config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize search optimization
        self.community_search_index = {}
        self.centrality_cache = {}
        
    async def execute_primer_phase(self,
                                 query_embedding: List[float],
                                 routing_result: DriftRoutingResult) -> Dict[str, Any]:
        """Execute community search and knowledge retrieval."""
        start_time = datetime.now()
        
        try:
            # Community retrieval
            self.logger.info("Starting community retrieval")
            communities = await self._retrieve_communities_enhanced(
                query_embedding, routing_result
            )
            
            # Data extraction
            self.logger.info("Starting data extraction")
            extracted_data = await self._extract_community_data_enhanced(communities)
            
            # Answer generation
            self.logger.info("Starting answer generation")
            initial_answer = await self._generate_initial_answer_enhanced(
                extracted_data, routing_result
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'communities': communities,
                'extracted_data': extracted_data,
                'initial_answer': initial_answer,
                'execution_time': execution_time,
                'metadata': {
                    'communities_retrieved': len(communities),
                    'entities_extracted': len(extracted_data.get('entities', [])),
                    'relationships_extracted': len(extracted_data.get('relationships', [])),
                    'phase': 'primer',
                    'step_range': '6-8'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Primer phase execution failed: {e}")
            raise
    
    async def _retrieve_communities_enhanced(self,
                                           query_embedding: List[float],
                                           routing_result: DriftRoutingResult) -> List[CommunityResult]:
        """
        Step 6: Enhanced community retrieval using comprehensive properties.
        
        Retrieves relevant communities based on query embedding similarity.
        """
        try:
            # Retrieve HyDE embeddings
            hyde_embeddings = await self._retrieve_hyde_embeddings_enhanced()
            
            if not hyde_embeddings:
                self.logger.warning("No HyDE embeddings found")
                return []
            
            # Compute similarities
            similarities = self._compute_hyde_similarities_enhanced(
                query_embedding, hyde_embeddings
            )
            
            # Rank communities
            ranked_communities = self._rank_communities_enhanced(
                similarities, routing_result
            )
            
            # Apply criteria
            filtered_communities = self._apply_termination_criteria(
                ranked_communities, routing_result
            )
            
            # Fetch community details
            community_results = await self._fetch_community_details_enhanced(
                filtered_communities
            )
            
            self.logger.info(f"Retrieved {len(community_results)} enhanced communities")
            return community_results
            
        except Exception as e:
            self.logger.error(f"Enhanced community retrieval failed: {e}")
            return []
    
    async def _load_community_search_index(self):
        """Load optimized community search index from Neo4j."""
        try:
            query = """
            MATCH (meta:DriftMetadata)
            WHERE meta.community_search_index IS NOT NULL
            RETURN meta.community_search_index as search_index,
                   meta.total_communities as total_communities
            """
            
            results = self.neo4j_conn.execute_query(query)
            
            for record in results:
                # The search index is a nested JSON structure with community IDs as keys
                search_index_data = record['search_index']
                if isinstance(search_index_data, dict):
                    # Each community in the search index 
                    for community_id, community_data in search_index_data.items():
                        self.community_search_index[community_id] = community_data
                else:
                    self.logger.warning(f"Unexpected search index format: {type(search_index_data)}")
            
            self.logger.info(f"Loaded search index for {len(self.community_search_index)} communities")
            
        except Exception as e:
            self.logger.error(f"Failed to load community search index: {e}")
    
    async def _retrieve_hyde_embeddings_enhanced(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve HyDE embeddings and metadata."""
        try:
            # Retrieve community embeddings
            query = """
            MATCH (c:Community)
            WHERE c.hyde_embeddings IS NOT NULL
            OPTIONAL MATCH (meta:CommunitiesMetadata)
            RETURN c.id as community_id,
                   c.hyde_embeddings as hyde_embeddings,
                   c.summary as summary,
                   c.key_entities as key_entities,
                   c.member_ids as member_ids,
                   size(c.hyde_embeddings) as embedding_size,
                   meta.modularity_score as global_modularity_score
            """
            
            results = self.neo4j_conn.execute_query(query)
            hyde_embeddings = {}
            
            for record in results:
                community_id = record['community_id']
                embeddings_data = record.get('hyde_embeddings')
                
                if embeddings_data and community_id:
                    hyde_embeddings[community_id] = {
                        'embeddings': embeddings_data,
                        'summary': record.get('summary', ''),
                        'key_entities': record.get('key_entities', []),
                        'member_ids': record.get('member_ids', []),
                        'embedding_size': record.get('embedding_size', 0),
                        'global_modularity_score': record.get('global_modularity_score', 0.0),
                        'embedding_type': 'hyde'
                    }
            
            self.logger.info(f"Retrieved enhanced HyDE embeddings for {len(hyde_embeddings)} communities")
            return hyde_embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve enhanced HyDE embeddings: {e}")
            # Retry logic for embeddings
            self.logger.info("Attempting retry for HyDE embeddings...")
            try:
                import time
                time.sleep(2)  # Brief delay before retry
                results = self.neo4j_conn.execute_query(query)
                hyde_embeddings = {}
                
                for record in results:
                    community_id = record['community_id']
                    embeddings_data = record.get('hyde_embeddings')
                    
                    if embeddings_data and community_id:
                        hyde_embeddings[community_id] = {
                            'embeddings': embeddings_data,
                            'summary': record.get('summary', ''),
                            'key_entities': record.get('key_entities', []),
                            'member_ids': record.get('member_ids', []),
                            'embedding_size': record.get('embedding_size', 0),
                            'global_modularity': record.get('global_modularity_score', 0.0)
                        }
                
                self.logger.info(f"Retry successful: Retrieved enhanced HyDE embeddings for {len(hyde_embeddings)} communities")
                return hyde_embeddings
                
            except Exception as retry_error:
                self.logger.error(f"Retry also failed: {retry_error}")
                return {}
    
    def _compute_hyde_similarities_enhanced(self,
                                          query_embedding: List[float],
                                          hyde_embeddings: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Enhanced similarity computation with global modularity weighting.
        
        Calculates similarity scores between query embedding and community embeddings.
        """
        similarities = {}
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            self.logger.warning("Query embedding has zero norm")
            return similarities
        
        for community_id, embedding_data in hyde_embeddings.items():
            embeddings_list = embedding_data['embeddings']
            global_modularity = embedding_data.get('global_modularity_score', 0.0)
            
            max_similarity = 0.0
            
            # Compute similarity
            try:
                # Parse embedding string
                if isinstance(embeddings_list, str):
                    embeddings_list = json.loads(embeddings_list)
                
                # Process embeddings
                if isinstance(embeddings_list, list) and len(embeddings_list) > 0:
                    # Use first embedding
                    hyde_vec = np.array(embeddings_list[0] if isinstance(embeddings_list[0], list) else embeddings_list)
                else:
                    hyde_vec = np.array(embeddings_list)
                
                hyde_norm = np.linalg.norm(hyde_vec)
                
                if hyde_norm > 0:
                    # Calculate similarity
                    base_similarity = np.dot(query_vec, hyde_vec) / (query_norm * hyde_norm)
                    
                    # Apply weighting
                    weighted_similarity = base_similarity * (1 + 0.2 * global_modularity)
                    max_similarity = weighted_similarity
                        
            except Exception as e:
                self.logger.warning(f"Error computing similarity for community {community_id}: {e}")
                continue
            
            similarities[community_id] = {
                'similarity': max_similarity,
                'global_modularity_score': global_modularity,
                'embedding_size': embedding_data.get('embedding_size', 0)
            }
        
        self.logger.info(f"Computed enhanced similarities for {len(similarities)} communities")
        return similarities
    
    def _rank_communities_enhanced(self,
                                 similarities: Dict[str, Dict[str, float]],
                                 routing_result: DriftRoutingResult) -> List[Tuple[str, Dict[str, float]]]:
        """
        Enhanced ranking using global modularity and similarity.
        
        Ranks communities based on a weighted combination of similarity score and modularity.
        """
        
        # Rank primarily by similarity, with modularity as secondary factor
        
        def ranking_score(item):
            _, scores = item
            similarity = scores['similarity']
            global_modularity = scores['global_modularity_score']
            
            # Weighted combination (similarity is primary)
            return 0.8 * similarity + 0.2 * global_modularity
        
        # Sort by combined ranking score
        ranked = sorted(similarities.items(), key=ranking_score, reverse=True)
        
        # Apply similarity threshold
        similarity_threshold = routing_result.parameters.get('similarity_threshold', 0.7)
        filtered_ranked = [
            (cid, scores) for cid, scores in ranked 
            if scores['similarity'] >= similarity_threshold
        ]
        
        self.logger.info(f"Enhanced ranking: {len(filtered_ranked)} communities above threshold {similarity_threshold}")
        return filtered_ranked
    
    def _apply_termination_criteria(self,
                                  ranked_communities: List[Tuple[str, Dict[str, float]]],
                                  routing_result: DriftRoutingResult) -> List[Tuple[str, Dict[str, float]]]:
        """
        Apply termination criteria for community selection.
        
        Limits the number of communities selected based on threshold parameters.
        """
        
        # Get termination criteria from routing or defaults
        max_communities = routing_result.parameters.get('max_communities', 3)
        min_global_modularity = routing_result.parameters.get('min_global_modularity', 0.3)
        
        # Apply criteria
        filtered = []
        for community_id, scores in ranked_communities:
            if len(filtered) >= max_communities:
                break
                
            # Check global modularity threshold
            if scores['global_modularity_score'] >= min_global_modularity:
                filtered.append((community_id, scores))
        
        self.logger.info(f"Applied termination criteria: {len(filtered)} communities selected")
        return filtered
    
    async def _fetch_community_details_enhanced(self,
                                              ranked_communities: List[Tuple[str, Dict[str, float]]]) -> List[CommunityResult]:
        """
        Fetch comprehensive community details with all properties.
        
        Retrieves detailed information about selected communities including summaries,
        key entities, and member IDs.
        """
        community_results = []
        
        for community_id, scores in ranked_communities:
            try:
                # Query the Community node directly by ID (since embedding communities have id=community_id)
                detail_query = """
                MATCH (c:Community)
                WHERE c.id = $community_id AND c.hyde_embeddings IS NOT NULL
                OPTIONAL MATCH (meta:CommunitiesMetadata)
                RETURN c.summary as summary,
                       c.key_entities as key_entities,
                       c.member_ids as member_ids,
                       c.internal_edges as internal_edges,
                       c.density as density,
                       c.avg_degree as avg_degree,
                       c.level as level,
                       meta.modularity_score as modularity_score,
                       CASE WHEN c.member_ids IS NOT NULL THEN size(c.member_ids) ELSE 0 END as member_count,
                       c.id as id
                LIMIT 1
                """
                
                results = self.neo4j_conn.execute_query(
                    detail_query, 
                    {'community_id': community_id}
                )
                
                if results:
                    record = results[0]
                    
                    # Create enhanced community result with actual available data from Neo4j
                    community_result = CommunityResult(
                        community_id=community_id,
                        similarity_score=scores['similarity'],
                        summary=record.get('summary', ''),
                        key_entities=record.get('key_entities', []),
                        member_ids=record.get('member_ids', []),
                        modularity_score=record.get('modularity_score', 0.0),
                        level=record.get('level', 1),
                        internal_edges=record.get('internal_edges', 0),
                        member_count=record.get('member_count', 0),
                        confidence_score=scores.get('confidence_score', 0.5),
                        search_index='',
                        termination_criteria={},
                        centrality_stats={
                            'avg_degree': record.get('avg_degree', 0.0),
                            'density': record.get('density', 0.0)
                        }
                    )
                    
                    community_results.append(community_result)
                    
            except Exception as e:
                self.logger.error(f"Failed to fetch details for community {community_id}: {e}")
                continue
        
        self.logger.info(f"Fetched enhanced details for {len(community_results)} communities")
        return community_results
    
    async def _extract_community_data_enhanced(self,
                                             communities: List[CommunityResult]) -> Dict[str, Any]:
        """
        Step 7: Enhanced data extraction with centrality measures.
        
        Extracts:
        - Entities with degree/betweenness/closeness centrality
        - Relationships with confidence scores
        - Community statistics and properties
        """
        try:
            all_entities = []
            all_relationships = []
            community_stats = []
            
            for community in communities:
                # Extract entities with centrality measures
                entities = await self._extract_entities_with_centrality(community)
                all_entities.extend(entities)
                
                # Extract relationships with properties
                relationships = await self._extract_relationships_enhanced(community)
                all_relationships.extend(relationships)
                
                # Collect community statistics
                community_stats.append({
                    'community_id': community.community_id,
                    'member_count': community.member_count,
                    'modularity_score': community.modularity_score,
                    'confidence_score': community.confidence_score,
                    'centrality_stats': community.centrality_stats
                })
            
            extracted_data = {
                'entities': all_entities,
                'relationships': all_relationships,
                'community_stats': community_stats,
                'extraction_metadata': {
                    'communities_processed': len(communities),
                    'entities_extracted': len(all_entities),
                    'relationships_extracted': len(all_relationships),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"Enhanced extraction completed: {len(all_entities)} entities, {len(all_relationships)} relationships")
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Enhanced data extraction failed: {e}")
            return {'entities': [], 'relationships': [], 'community_stats': []}
    
    async def _extract_entities_with_centrality(self,
                                              community: CommunityResult) -> List[EntityResult]:
        """
        Extract entities with comprehensive centrality measures.
        
        Retrieves entities from the community with their associated centrality metrics.
        """
        try:
            # Use member_ids for direct access if available
            member_ids = community.member_ids if community.member_ids else []
            
            if member_ids:
                # Direct member access query based on actual schema
                entity_query = """
                MATCH (n)
                WHERE n.id IN $member_ids
                  AND n.name IS NOT NULL 
                  AND n.content IS NOT NULL
                RETURN n.id as entity_id,
                       n.name as name,
                       n.content as content,
                       n.confidence as confidence,
                       n.degree_centrality as degree_centrality,
                       n.betweenness_centrality as betweenness_centrality,
                       n.closeness_centrality as closeness_centrality,
                       labels(n) as node_types
                ORDER BY n.degree_centrality DESC
                """
                
                results = self.neo4j_conn.execute_query(
                    entity_query,
                    {'member_ids': member_ids}
                )
            else:
                # Fallback: find entities using community_id pattern matching
                entity_query = """
                MATCH (n)
                WHERE n.community_id IS NOT NULL
                  AND n.name IS NOT NULL 
                  AND n.content IS NOT NULL
                RETURN n.id as entity_id,
                       n.name as name,
                       n.content as content,
                       n.confidence as confidence,
                       n.degree_centrality as degree_centrality,
                       n.betweenness_centrality as betweenness_centrality,
                       n.closeness_centrality as closeness_centrality,
                       labels(n) as node_types
                ORDER BY n.degree_centrality DESC
                LIMIT 20
                """
                
                results = self.neo4j_conn.execute_query(entity_query)
            
            entities = []
            for record in results:
                entity = EntityResult(
                    entity_id=record['entity_id'],
                    name=record.get('name', ''),
                    content=record.get('content', ''),
                    confidence=record.get('confidence', 0.0),
                    degree_centrality=record.get('degree_centrality', 0.0),
                    betweenness_centrality=record.get('betweenness_centrality', 0.0),
                    closeness_centrality=record.get('closeness_centrality', 0.0),
                    community_id=community.community_id,
                    node_type=record.get('node_types', ['Unknown'])[0] if record.get('node_types') else 'Unknown'
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Failed to extract entities for community {community.community_id}: {e}")
            return []
    
    async def _extract_relationships_enhanced(self,
                                            community: CommunityResult) -> List[RelationshipResult]:
        """
        Extract relationships with enhanced properties.
        
        Retrieves relationship data between entities within the specified community.
        """
        try:
            relationship_query = """
            MATCH (a)-[r]->(b)
            WHERE a.community_id = $community_id 
              AND b.community_id = $community_id
              AND r.confidence > 0.5
            RETURN startNode(r).id as start_node,
                   endNode(r).id as end_node,
                   type(r) as relationship_type,
                   r.confidence as confidence
            ORDER BY r.confidence DESC
            LIMIT 50
            """
            
            results = self.neo4j_conn.execute_query(
                relationship_query,
                {'community_id': community.community_id}
            )
            
            relationships = []
            for record in results:
                relationship = RelationshipResult(
                    start_node=record['start_node'],
                    end_node=record['end_node'],
                    relationship_type=record['relationship_type'],
                    confidence=record.get('confidence', 0.0)
                )
                relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Failed to extract relationships for community {community.community_id}: {e}")
            return []
    
    async def _generate_initial_answer_enhanced(self, 
                                              extracted_data: Dict[str, Any],
                                              routing_result: DriftRoutingResult) -> Dict[str, Any]:
        """
        Step 8: Context-aware initial answer generation.
        
        Uses:
        - Entity importance from centrality measures
        - Relationship confidence for evidence strength
        - Community statistics for context sizing
        """
        try:
            entities = extracted_data['entities']
            relationships = extracted_data['relationships']
            community_stats = extracted_data['community_stats']
            
            # Rank entities by importance (centrality measures)
            important_entities = sorted(
                entities, 
                key=lambda e: (e.degree_centrality + e.betweenness_centrality) / 2,
                reverse=True
            )[:10]
            
            # Select high-confidence relationships
            strong_relationships = [
                r for r in relationships 
                if r.confidence >= 0.7
            ]
            
            # Prepare context for LLM
            llm_context = self._prepare_llm_context_enhanced(
                important_entities, strong_relationships, community_stats, routing_result
            )
            
            # Generate initial answer using configured LLM
            llm_response = await self._generate_llm_answer(llm_context, routing_result)
            
            initial_answer = {
                'content': llm_response['answer'],
                'llm_context': llm_context,
                'context_used': {
                    'important_entities': len(important_entities),
                    'strong_relationships': len(strong_relationships),
                    'communities_analyzed': len(community_stats)
                },
                'confidence_metrics': {
                    'avg_entity_centrality': np.mean([e.degree_centrality for e in important_entities]) if important_entities else 0,
                    'avg_relationship_confidence': np.mean([r.confidence for r in strong_relationships]) if strong_relationships else 0,
                    'avg_community_modularity': np.mean([c['modularity_score'] for c in community_stats]) if community_stats else 0,
                    'llm_confidence': llm_response['confidence']
                },
                'follow_up_questions': llm_response['follow_up_questions'],
                'reasoning': llm_response['reasoning']
            }
            
            self.logger.info("Enhanced initial answer generated with comprehensive context")
            return initial_answer
            
        except Exception as e:
            self.logger.error(f"Enhanced answer generation failed: {e}")
            return {'content': 'Error generating initial answer', 'error': str(e)}
    
    def _prepare_llm_context_enhanced(self, 
                                    entities: List[EntityResult],
                                    relationships: List[RelationshipResult],
                                    community_stats: List[Dict[str, Any]],
                                    routing_result: DriftRoutingResult) -> str:
        """Prepare enhanced context for LLM with comprehensive information."""
        
        context_parts = [
            f"Query: {routing_result.original_query}",
            f"Search Strategy: {routing_result.search_strategy.value}",
            "",
            "=== IMPORTANT ENTITIES (Use these specific names in your answer) ===",
        ]
        
        for i, entity in enumerate(entities[:10], 1):  # Show more entities
            context_parts.append(
                f"{i}. NAME: '{entity.name}' | Description: {entity.content[:100]}... "
                f"| Centrality: {entity.degree_centrality:.3f} | Confidence: {entity.confidence:.3f}"
            )
        
        context_parts.extend([
            "",
            "=== KEY RELATIONSHIPS (Use these connections in your answer) ===",
        ])
        
        for i, rel in enumerate(relationships[:8], 1):  # Show more relationships
            context_parts.append(
                f"{i}. '{rel.start_node}' --[{rel.relationship_type}]--> '{rel.end_node}' "
                f"| Confidence: {rel.confidence:.3f}"
            )
        
        # Add quick reference list of all entity names
        entity_names = [entity.name for entity in entities[:15]]
        context_parts.extend([
            "",
            "=== ENTITY NAMES FOR REFERENCE ===",
            f"Available entities: {', '.join(entity_names)}",
            "",
            "=== COMMUNITY STATISTICS ===",
        ])
        
        for stat in community_stats:
            context_parts.append(
                f"Community {stat['community_id']}: {stat['member_count']} members, "
                f"modularity: {stat['modularity_score']:.3f}"
            )
        
        context_parts.extend([
            "",
            "REMEMBER: Use the specific entity names listed above in your answer!"
        ])
        
        return "\n".join(context_parts)
    
    async def _generate_llm_answer(self, 
                                 context: str, 
                                 routing_result: DriftRoutingResult) -> Dict[str, Any]:
        """
        Generate actual LLM response using the configured LLM.
        
        Uses the LLM from GraphRAGSetup to generate answers with follow-up questions.
        """
        try:
            # Construct comprehensive prompt for LLM
            prompt = f"""
You are an expert knowledge analyst. Answer the user's query using SPECIFIC NAMES and information from the graph data provided below.

IMPORTANT: Use the actual entity names, organization names, and relationship details from the graph data. Do not give generic answers.

GRAPH DATA CONTEXT:
{context}

INSTRUCTIONS:
1. Answer using SPECIFIC ENTITY NAMES from the "IMPORTANT ENTITIES" section above
2. Reference actual relationships and organizations mentioned in the graph data
3. If the query asks for members/organizations, LIST THE ACTUAL NAMES from the entities
4. Use confidence scores and centrality measures as evidence strength indicators
5. Generate follow-up questions based on the specific entities found

RESPONSE FORMAT:
Answer: [Use specific names and details from the graph data above]
Confidence: [0.0-1.0]
Reasoning: [Why these specific entities answer the query]
Follow-up Questions:
1. [Specific question about entities found]
2. [Question about relationships discovered]
3. [Question about community connections]
4. [Question for deeper exploration]
5. [Question about related entities]
"""

            # Call the configured LLM
            llm_response = await self.setup.llm.acomplete(prompt)
            response_text = llm_response.text
            
            # Parse LLM response
            parsed_response = self._parse_llm_response(response_text)
            
            self.logger.info(f"LLM generated answer with confidence: {parsed_response['confidence']}")
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"LLM answer generation failed: {e}")
            # Fallback response
            return {
                'answer': f"Based on the graph analysis, I found relevant information but encountered an issue generating the full response: {str(e)}",
                'confidence': 0.3,
                'reasoning': "LLM generation encountered an error, providing basic analysis from graph data.",
                'follow_up_questions': [
                    "What specific aspects would you like me to explore further?",
                    "Are there particular entities or relationships of interest?",
                    "Should I focus on a specific community or time period?"
                ]
            }
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured LLM response into components."""
        try:
            lines = response_text.strip().split('\n')
            
            answer = ""
            confidence = 0.5
            reasoning = ""
            follow_up_questions = []
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("Answer:"):
                    current_section = "answer"
                    answer = line.replace("Answer:", "").strip()
                elif line.startswith("Confidence:"):
                    confidence_text = line.replace("Confidence:", "").strip()
                    try:
                        confidence = float(confidence_text)
                    except (ValueError, TypeError):
                        confidence = 0.5
                elif line.startswith("Reasoning:"):
                    current_section = "reasoning"
                    reasoning = line.replace("Reasoning:", "").strip()
                elif line.startswith("Follow-up Questions:"):
                    current_section = "questions"
                elif current_section == "answer" and line:
                    answer += " " + line
                elif current_section == "reasoning" and line:
                    reasoning += " " + line
                elif current_section == "questions" and line.startswith(("1.", "2.", "3.", "4.", "5.")):
                    question = line[2:].strip()  # Remove "1. " etc.
                    follow_up_questions.append(question)
            
            return {
                'answer': answer.strip() if answer else "Unable to generate answer from available context.",
                'confidence': max(0.0, min(1.0, confidence)),  # Clamp between 0-1
                'reasoning': reasoning.strip() if reasoning else "Analysis based on graph structure and entity relationships.",
                'follow_up_questions': follow_up_questions if follow_up_questions else [
                    "What additional information would be helpful?",
                    "Are there specific aspects to explore further?",
                    "Should I analyze different communities or relationships?"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return {
                'answer': response_text[:500] if response_text else "No response generated.",
                'confidence': 0.4,
                'reasoning': "Direct LLM output due to parsing issues.",
                'follow_up_questions': ["What would you like to know more about?"]
            }


# Exports
__all__ = ['CommunitySearchEngine', 'CommunityResult', 'EntityResult', 'RelationshipResult']