"""
Phase 3: Community Summarization using LLM
Loads graph-data-phase-2.json, generates summaries, saves graph-data-final.json
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict

import networkx as nx
import openai
import google.generativeai as genai

# JSON parsing libraries (same as Phase 1)
import orjson
from json_repair import repair_json

from my_config import MY_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphBuilderPhase3:
    """Phase 3: Generate community summaries using LLM"""
    
    def __init__(self, llm_provider: str = "cerebras"):
        """Initialize Phase 3 processor"""
        self.llm_provider = llm_provider.lower()
        self.graph_data = None
        self.nx_graph = None
        self.community_assignments = {}
        self.community_stats = {}
        
        # Initialize LLM API based on provider
        if self.llm_provider == "cerebras":
            if not MY_CONFIG.CEREBRAS_API_KEY:
                raise ValueError("CEREBRAS_API_KEY not set")
            
            self.cerebras_client = openai.OpenAI(
                api_key=MY_CONFIG.CEREBRAS_API_KEY,
                base_url="https://api.cerebras.ai/v1"
            )
            self.model_name = "llama-4-scout-17b-16e-instruct"
            logger.info("🚀 Using Cerebras API")
            
        elif self.llm_provider == "gemini":
            if not MY_CONFIG.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")
            
            genai.configure(api_key=MY_CONFIG.GEMINI_API_KEY)
            self.model_name = "gemini-1.5-flash"
            self.gemini_model = genai.GenerativeModel(self.model_name)
            logger.info("🆓 Using Google Gemini API")
            
        else:
            raise ValueError(f"Invalid provider '{llm_provider}'. Choose: cerebras, gemini")
        
        # Initialize embedding model for DRIFT search metadata
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            self.embedding_model = HuggingFaceEmbedding(
                model_name=MY_CONFIG.EMBEDDING_MODEL
            )
            logger.info(f"🔍 Initialized embedding model: {MY_CONFIG.EMBEDDING_MODEL}")
        except Exception as e:
            logger.warning(f"⚠️ Embedding model initialization failed: {e}")
            self.embedding_model = None
        
        logger.info("✅ Phase 3 initialized: Community Summarization")
        logger.info(f"📊 LLM Provider: {self.llm_provider.upper()}, Model: {self.model_name}")
    
    # STEP 1: Load Phase 2 Output
    def load_graph_data(self, input_path: str = None) -> bool:
        """Load graph-data-phase-2.json from Phase 2"""
        if input_path is None:
            input_path = "workspace/graph_data/graph-data-phase-2.json"
        
        try:
            input_file = Path(input_path)
            if not input_file.exists():
                logger.error(f"❌ Input file not found: {input_path}")
                logger.error("   Please run Phase 2 (2b_process_graph_phase2.py) first")
                return False
            
            with open(input_file, 'r', encoding='utf-8') as f:
                self.graph_data = json.load(f)
            
            node_count = len(self.graph_data.get("nodes", []))
            rel_count = len(self.graph_data.get("relationships", []))
            
            # Verify Phase 2 was completed
            if self.graph_data.get("metadata", {}).get("phase") != "community_detection":
                logger.error("❌ Input file is not from Phase 2 (community_detection)")
                return False
            
            logger.info(f"📂 Loaded graph-data-phase-2.json: {node_count} nodes, {rel_count} relationships")
            
            # Load community stats
            self.community_stats = self.graph_data.get("community_stats", {})
            num_communities = len(self.community_stats)
            logger.info(f"📊 Found {num_communities} communities to summarize")
            
            if num_communities == 0:
                logger.error("❌ No communities found in Phase 2 output")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading graph data: {e}")
            return False
    
    # STEP 2: Build NetworkX Graph
    def _build_networkx_graph(self) -> nx.Graph:
        """Rebuild NetworkX graph from JSON data"""
        logger.info("🔨 Building NetworkX graph from JSON data...")
        
        G = nx.Graph()
        
        # Add nodes with attributes
        for node in self.graph_data["nodes"]:
            node_id = node["id"]
            properties = node.get("properties", {})
            
            G.add_node(
                node_id,
                name=properties.get("name", ""),
                type=node.get("labels", ["Unknown"])[0],
                description=properties.get("content", ""),
                community_id=properties.get("community_id", ""),
                degree_centrality=properties.get("degree_centrality", 0.0)
            )
        
        # Add edges
        for rel in self.graph_data["relationships"]:
            start_node = rel.get("startNode")
            end_node = rel.get("endNode")
            
            if start_node in G.nodes() and end_node in G.nodes():
                G.add_edge(start_node, end_node)
        
        logger.info(f"✅ Built NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    # STEP 3: Extract Community Assignments
    def _extract_community_assignments(self) -> Dict[str, int]:
        """Extract community assignments from node properties"""
        logger.info("📋 Extracting community assignments from nodes...")
        
        assignments = {}
        
        for node in self.graph_data["nodes"]:
            node_id = node["id"]
            comm_id_str = node.get("properties", {}).get("community_id", "")
            
            if comm_id_str and comm_id_str.startswith("comm-"):
                try:
                    comm_id = int(comm_id_str.replace("comm-", ""))
                    assignments[node_id] = comm_id
                except ValueError:
                    logger.warning(f"Invalid community_id format: {comm_id_str}")
        
        logger.info(f"✅ Extracted {len(assignments)} community assignments")
        
        return assignments
    
    # STEP 4: LLM Inference Methods
    def _cerebras_inference(self, system_prompt: str, user_prompt: str) -> str:
        """Call Cerebras API for inference"""
        try:
            # Calculate dynamic parameters based on community size and complexity
            total_nodes = self.nx_graph.number_of_nodes() if hasattr(self, 'nx_graph') else 100
            complexity_factor = min(1.0, total_nodes / 1000)
            
            # Adaptive temperature: higher for complex graphs to encourage creativity
            dynamic_temperature = round(0.1 + (complexity_factor * 0.4), 2)  # Range: 0.1-0.5
            
            # Adaptive tokens: more for larger/complex summaries
            dynamic_tokens = int(300 + (complexity_factor * 400))  # Range: 300-700
            
            response = self.cerebras_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=dynamic_temperature,
                max_tokens=dynamic_tokens
            )
            
            if not response or not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from Cerebras")
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Cerebras inference error: {e}")
            raise e
    
    def _gemini_inference(self, system_prompt: str, user_prompt: str) -> str:
        """Call Gemini API for inference"""
        try:
            # Calculate dynamic generation config based on graph complexity
            total_nodes = self.nx_graph.number_of_nodes() if hasattr(self, 'nx_graph') else 100
            complexity_factor = min(1.0, total_nodes / 1000)
            
            # Adaptive temperature and tokens for Gemini
            dynamic_temperature = round(0.1 + (complexity_factor * 0.4), 2)
            dynamic_tokens = int(300 + (complexity_factor * 400))
            
            generation_config = {
                "temperature": dynamic_temperature,
                "max_output_tokens": dynamic_tokens,
                "candidate_count": 1
            }
            
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.gemini_model.generate_content(
                combined_prompt,
                generation_config=generation_config
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini")
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini inference error: {e}")
            raise e
    
    # STEP 5: Generate Community Summaries
    def _generate_community_summaries(self) -> Dict[int, str]:
        """Generate LLM summaries for each community"""
        logger.info("📝 Generating community summaries with LLM...")
        logger.info(f"   Total communities to summarize: {len(self.community_stats)}")
        
        summaries = {}
        
        # Group nodes by community
        communities = defaultdict(list)
        for node_id, comm_id in self.community_assignments.items():
            communities[comm_id].append(node_id)
        
        start_time = time.time()
        
        for idx, (comm_id_str, stats) in enumerate(self.community_stats.items(), 1):
            comm_id = int(comm_id_str)
            
            logger.info(f"   Processing community {idx}/{len(self.community_stats)}: comm-{comm_id} ({stats['member_count']} members)")
            
            # Get top entities by centrality
            node_ids = communities[comm_id]
            subgraph = self.nx_graph.subgraph(node_ids)
            
            # Get nodes sorted by degree centrality
            centrality = nx.degree_centrality(subgraph)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]
            
            # Prepare entity information for LLM
            entity_info = []
            for node_id, _ in top_nodes:
                node_data = self.nx_graph.nodes[node_id]
                entity_info.append({
                    "name": node_data.get("name", "Unknown"),
                    "type": node_data.get("type", "Unknown"),
                    "description": node_data.get("description", "")[:150]  # Limit length
                })
            
            # Create LLM prompt
            # Senior-developer style system/user prompts with strict output schema
            # Calculate dynamic topic count based on community size
            topic_count = max(2, min(5, stats['member_count'] // 3))  # Scale with community size
            
            system_prompt = (
                "You are a specialized knowledge graph summarization assistant. Your task is to analyze community "
                "structures and generate comprehensive summaries for graph-based retrieval systems.\n\n"
                "CONSTITUTIONAL AI PRINCIPLES:\n"
                "1. Content-Adaptive: Generate summaries based on actual community composition and statistics\n"
                "2. Context-Aware: Consider entity relationships and community density in summarization\n"
                "3. Quality-First: Prioritize accuracy and relevance over brevity\n"
                "4. Structured Output: Ensure consistent JSON format for programmatic consumption\n\n"
                "SUMMARIZATION GUIDELINES:\n"
                "- Analyze entity types, relationships, and community structure\n"
                "- Identify key themes and concepts that define this community\n"
                "- Generate topics that capture semantic meaning, not just entity names\n"
                "- Assess confidence based on data completeness and coherence\n"
                "- Use neutral, factual tone suitable for technical documentation"
            )

            user_prompt = (
                f"Analyze the following community data and generate a structured summary.\n\n"
                f"COMMUNITY STATISTICS:\n"
                f"- Total Members: {stats['member_count']}\n"
                f"- Internal Connections: {stats['internal_edges']}\n"
                f"- Community Density: {stats['density']:.3f}\n"
                f"- Connectivity Strength: {'High' if stats['density'] > 0.1 else 'Medium' if stats['density'] > 0.05 else 'Low'}\n\n"
                f"TOP ENTITIES (name, type, description):\n{json.dumps(entity_info, indent=2)}\n\n"
                f"OUTPUT FORMAT (strict JSON):\n"
                f"{{\n"
                f"    \"summary\": \"2-3 sentence comprehensive summary of community purpose and characteristics\",\n"
                f"    \"primary_topics\": [\"topic_1\", \"topic_2\", \"topic_{topic_count}\"],\n"
                f"    \"confidence\": 0.85\n"
                f"}}\n\n"
                f"VALIDATION REQUIREMENTS:\n"
                f"- summary: Must be 2-3 complete sentences describing community focus and key characteristics\n"
                f"- primary_topics: Array of exactly {topic_count} descriptive phrases (not just entity names)\n"
                f"- confidence: Float between 0.0-1.0 based on data quality and coherence\n\n"
                f"IMPORTANT: Respond with ONLY the JSON object. No markdown formatting, no explanations, no code blocks."
            )
            
            # Call LLM for summary
            try:
                if self.llm_provider == "gemini":
                    summary_response = self._gemini_inference(system_prompt, user_prompt)
                else:  # cerebras
                    summary_response = self._cerebras_inference(system_prompt, user_prompt)
                
                # Parse JSON response
                parsed_summary = self._parse_summary_response(summary_response, comm_id)
                if parsed_summary:
                    summaries[comm_id] = parsed_summary
                else:
                    # Fallback to raw response if parsing fails
                    summaries[comm_id] = summary_response.strip()
                
                # Log progress every 10 communities
                if idx % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / idx
                    remaining = avg_time * (len(self.community_stats) - idx)
                    logger.info(f"   Progress: {idx}/{len(self.community_stats)} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate summary for community {comm_id}: {e}")
                summaries[comm_id] = f"Community with {stats['member_count']} entities focused on {entity_info[0]['type'] if entity_info else 'various'} topics."
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Generated {len(summaries)} community summaries in {elapsed:.1f}s")
        
        return summaries
    
    def _parse_summary_response(self, response: str, comm_id: int) -> str:
        """Parse JSON summary response with fallback to text extraction"""
        try:
            # Clean response
            cleaned_response = response.strip()
            
            # Remove markdown formatting
            if "```json" in cleaned_response:
                parts = cleaned_response.split("```json")
                if len(parts) > 1:
                    json_part = parts[1].split("```")[0].strip()
                    cleaned_response = json_part
            elif "```" in cleaned_response:
                parts = cleaned_response.split("```")
                if len(parts) >= 3:
                    cleaned_response = parts[1].strip()
            
            # Try to parse JSON
            try:
                summary_data = self._smart_json_parse_summary(cleaned_response)
                if summary_data and isinstance(summary_data, dict):
                    summary_text = summary_data.get('summary', '')
                    if summary_text and len(summary_text.strip()) > 10:
                        return summary_text.strip()
            except ValueError as e:
                logger.debug(f"Summary JSON parsing failed for comm-{comm_id}: {e}")
            except Exception as e:
                logger.debug(f"Summary JSON parsing unexpected error for comm-{comm_id}: {e}")
                    
        except Exception as e:
            logger.debug(f"Summary JSON parsing failed for comm-{comm_id}: {e}")
        
        # Fallback: extract first meaningful sentence
        try:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 20 and '.' in line and not line.startswith('{'):
                    return line
        except Exception:
            pass
        
        return None
    
    def _smart_json_parse_summary(self, json_text: str) -> Dict:
        """
        Simple 5-step JSON parsing approach (exactly same as Phase 1)
        """
        cleaned_text = json_text.strip()
        
        # Step 1: orjson
        try:
            result = orjson.loads(cleaned_text.encode('utf-8'))
            logger.debug("✅ Step 1: orjson succeeded")
            return result
        except Exception as e:
            logger.debug(f"❌ Step 1: orjson failed - {e}")
        
        # Step 2: json-repair
        try:
            repaired = repair_json(cleaned_text)
            result = orjson.loads(repaired.encode('utf-8'))
            logger.debug("✅ Step 2: json-repair + orjson succeeded")
            return result
        except Exception as e:
            logger.debug(f"❌ Step 2: json-repair failed - {e}")
        
        # Step 3: standard json
        try:
            result = json.loads(cleaned_text)
            logger.debug("✅ Step 3: standard json succeeded")
            return result
        except Exception as e:
            logger.debug(f"❌ Step 3: standard json failed - {e}")
        
        # Step 4: json-repair + standard json
        try:
            repaired = repair_json(cleaned_text)
            result = json.loads(repaired)
            logger.debug("✅ Step 4: json-repair + standard json succeeded")
            return result
        except Exception as e:
            logger.debug(f"❌ Step 4: json-repair + standard json failed - {e}")
        
        # Step 5: All failed - this will trigger save failed txt files
        raise ValueError("All 4 JSON parsing steps failed")
    
    # STEP 6: Identify Key Entities
    def _identify_key_entities(self) -> Dict[int, List[str]]:
        """Identify key entities in each community based on centrality"""
        logger.info("🔑 Identifying key entities per community...")
        
        key_entities = {}
        
        # Group nodes by community
        communities = defaultdict(list)
        for node_id, comm_id in self.community_assignments.items():
            communities[comm_id].append(node_id)
        
        for comm_id, node_ids in communities.items():
            subgraph = self.nx_graph.subgraph(node_ids)
            
            # Calculate degree centrality
            centrality = nx.degree_centrality(subgraph)
            
            # Get top 5 entities
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            key_entities[comm_id] = [
                self.nx_graph.nodes[node_id].get("name", "Unknown")
                for node_id, _ in top_nodes
            ]
        
        logger.info(f"✅ Identified key entities for {len(key_entities)} communities")
        
        return key_entities
    
    # STEP 7: Create Community Nodes
    def _create_community_nodes(self, community_summaries: Dict[int, str], key_entities: Dict[int, List[str]]) -> List[Dict]:
        """Create community nodes for the graph"""
        logger.info("🏗️ Creating community nodes...")
        
        import uuid
        
        community_nodes = []
        
        for comm_id_str, stats in self.community_stats.items():
            comm_id = int(comm_id_str)
            
            node = {
                "id": f"community-{uuid.uuid4()}",
                "elementId": f"community-{uuid.uuid4()}",
                "labels": ["Community"],
                "properties": {
                    "community_id": f"comm-{comm_id}",
                    "level": 1,
                    "member_count": stats["member_count"],
                    "internal_edges": stats["internal_edges"],
                    "density": round(stats["density"], 4),
                    "avg_degree": round(stats["avg_degree"], 2),
                    "summary": community_summaries.get(comm_id, ""),
                    "key_entities": key_entities.get(comm_id, []),
                    "created_date": datetime.now().isoformat()
                }
            }
            community_nodes.append(node)
        
        logger.info(f"✅ Created {len(community_nodes)} community nodes")
        
        return community_nodes
    
    # STEP 8: Create IN_COMMUNITY Relationships
    def _create_in_community_relationships(self, community_nodes: List[Dict]) -> List[Dict]:
        """Create IN_COMMUNITY relationships linking entities to communities"""
        logger.info("Creating IN_COMMUNITY relationships...")
        
        import uuid
        
        # Create mapping from community_id to community node id
        comm_id_to_node_id = {}
        for node in community_nodes:
            comm_id = node["properties"]["community_id"]
            comm_id_to_node_id[comm_id] = node["id"]
        
        relationships = []
        
        for entity_id, comm_id in self.community_assignments.items():
            comm_node_id = comm_id_to_node_id.get(f"comm-{comm_id}")
            
            if comm_node_id:
                # Calculate confidence based on community membership strength
                entity_node = next((n for n in self.graph_data['nodes'] if n['id'] == entity_id), None)
                if entity_node:
                    degree_centrality = entity_node.get('properties', {}).get('degree_centrality', 0.5)
                    # Higher centrality = higher confidence in community assignment
                    dynamic_confidence = round(0.6 + (degree_centrality * 0.4), 3)  # Range: 0.6-1.0
                else:
                    dynamic_confidence = 0.8  # Default for missing nodes
                
                rel = {
                    "id": f"rel-{uuid.uuid4()}",
                    "startNode": entity_id,
                    "endNode": comm_node_id,
                    "type": "IN_COMMUNITY",
                    "properties": {
                        "confidence": dynamic_confidence,
                        "assigned_date": datetime.now().isoformat()
                    }
                }
                relationships.append(rel)
        
        logger.info(f"✅ Created {len(relationships)} IN_COMMUNITY relationships")
        
        return relationships
    
    # STEP 9: DRIFT Search Metadata Generation
    def _generate_drift_metadata(self, community_summaries: Dict[int, str], key_entities: Dict[int, List[str]]) -> Dict:
        """Generate DRIFT search metadata using existing embedding infrastructure"""
        logger.info("🔍 Generating DRIFT search metadata...")
        
        if not self.embedding_model:
            logger.warning("⚠️ Embedding model not available, skipping DRIFT metadata")
            return {}
        
        # Calculate dynamic values from actual graph data
        total_communities = len(community_summaries)
        total_nodes = self.nx_graph.number_of_nodes()
        total_edges = self.nx_graph.number_of_edges()
        avg_community_size = sum(self.community_stats.get(str(i), {}).get("member_count", 0) 
                               for i in community_summaries.keys()) / total_communities if total_communities > 0 else 0
        graph_density = total_edges / (total_nodes * (total_nodes - 1) / 2) if total_nodes > 1 else 0
        
        # Calculate dynamic thresholds based on graph complexity
        complexity_factor = min(1.0, (total_nodes + total_edges) / 10000)  # Scale 0-1 based on graph size
        base_confidence = 0.6 + (complexity_factor * 0.3)  # Range: 0.6-0.9
        base_response_time = 1.0 + (complexity_factor * 3.0)  # Range: 1-4 seconds
        base_memory = int(20 + (avg_community_size * complexity_factor * 5))  # Scale with size
        
        # Adaptive configuration based on graph characteristics
        max_communities_for_primer = min(total_communities, max(2, total_communities // 4))
        lightweight_communities = max(1, max_communities_for_primer // 2)
        standard_communities = max(2, int(max_communities_for_primer // 1.5))
        comprehensive_communities = max_communities_for_primer
        
        # Calculate dynamic iteration counts based on community distribution
        max_iter = max(2, min(5, int(total_communities / 10) + 2))
        hyde_count = max(2, min(5, int(avg_community_size / 5) + 2))
        
        drift_metadata = {
            "version": "1.0",
            "generated_timestamp": datetime.now().isoformat(),
            "configuration": {
                "max_iterations": max_iter,
                "confidence_threshold": round(base_confidence + 0.1, 2),
                "top_k_communities": max_communities_for_primer,
                "hyde_expansion_count": hyde_count,
                "termination_criteria": "confidence_or_max_iterations"
            },
            "query_routing_config": {
                "lightweight_drift": {
                    "triggers": ["single_entity", "simple_fact", "definition_query"],
                    "config": {
                        "primer_communities": int(lightweight_communities), 
                        "follow_up_iterations": max(1, max_iter - 2), 
                        "confidence_threshold": round(base_confidence, 2)
                    }
                },
                "standard_drift": {
                    "triggers": ["multi_entity", "relationship_query", "how_does"],
                    "config": {
                        "primer_communities": int(standard_communities), 
                        "follow_up_iterations": max(1, max_iter - 1), 
                        "confidence_threshold": round(base_confidence + 0.1, 2)
                    }
                },
                "comprehensive_drift": {
                    "triggers": ["analyze", "compare", "implications", "strategy"],
                    "config": {
                        "primer_communities": int(comprehensive_communities), 
                        "follow_up_iterations": max_iter, 
                        "confidence_threshold": round(base_confidence + 0.2, 2)
                    }
                }
            },
            "performance_monitoring": {
                "response_time_targets": {
                    "p50": round(base_response_time * 1.0, 1),
                    "p95": round(base_response_time * 2.5, 1), 
                    "p99": round(base_response_time * 5.0, 1)
                },
                "resource_tracking": {
                    "memory_per_query": base_memory, 
                    "cache_hit_rate_target": round(0.5 + (complexity_factor * 0.3), 2)
                },
                "bottleneck_identification": ["community_ranking", "follow_up_generation", "embedding_computation"]
            },
            "community_search_index": {},
            "search_optimization": {
                "total_communities": total_communities,
                "avg_community_size": round(avg_community_size, 1),
                "graph_density": round(graph_density, 6),
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "max_primer_communities": max_communities_for_primer
            }
        }
        
        # Process each community
        for comm_id, summary in community_summaries.items():
            comm_key = f"comm-{comm_id}"
            
            try:
                # Generate embeddings using existing HuggingFace model
                summary_embedding = self.embedding_model.get_text_embedding(summary)
                hyde_embeddings = self._generate_hyde_embeddings(summary)
                follow_up_questions = self._generate_follow_up_questions(summary, comm_id, key_entities.get(comm_id, []))
                
                # Add to search index
                drift_metadata["community_search_index"][comm_key] = {
                    "summary": summary,
                    "key_entities": key_entities.get(comm_id, []),
                    "embeddings": {
                        "summary_embedding": summary_embedding,
                        "hyde_embeddings": hyde_embeddings
                    },
                    "follow_up_templates": follow_up_questions,
                    "statistics": self.community_stats.get(str(comm_id), {})
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate metadata for {comm_key}: {e}")
                continue
        
        logger.info(f"✅ Generated DRIFT metadata for {len(drift_metadata['community_search_index'])} communities")
        return drift_metadata
    
    def _generate_hyde_embeddings(self, community_summary: str) -> List[List[float]]:
        """Generate HyDE embeddings for enhanced recall"""
        
        # Create 3 hypothetical document variations
        hyde_templates = [
            f"Research analysis and findings: {community_summary}",
            f"Technical report and documentation: {community_summary}",
            f"Business implications and strategic analysis: {community_summary}"
        ]
        
        hyde_embeddings = []
        for template in hyde_templates:
            try:
                embedding = self.embedding_model.get_text_embedding(template)
                hyde_embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"⚠️ HyDE embedding generation failed: {e}")
                continue
        
        return hyde_embeddings
    
    def _generate_follow_up_questions(self, community_summary: str, comm_id: int, key_entities: List[str]) -> List[Dict]:
        """Generate follow-up questions using existing LLM infrastructure"""
        
        # Professional system prompt matching Phase 1 style
        system_prompt = (
            "You are a specialized DRIFT search question generation assistant. Your task is to analyze community "
            "summaries and generate targeted follow-up questions for iterative knowledge graph exploration.\n\n"
            "CONSTITUTIONAL AI PRINCIPLES:\n"
            "1. Context-Adaptive: Generate questions based on actual community content and entities\n"
            "2. Search-Aware: Choose appropriate search types to guide query routing optimization\n"
            "3. Relevance-First: Prioritize questions that expand understanding of community themes\n"
            "4. Structured Output: Ensure consistent JSON format for programmatic consumption\n\n"
            "QUESTION GENERATION GUIDELINES:\n"
            "- Analyze community summary and key entities to identify knowledge gaps\n"
            "- Generate questions that would reveal additional relevant information\n"
            "- Use local search for entity-specific queries, relationship for connections, global for themes\n"
            "- Assign relevance scores based on potential value for understanding the community\n"
            "- Target entities should guide search focus and retrieval optimization"
        )

        user_prompt = (
            f"Analyze the following community data and generate targeted follow-up questions.\n\n"
            f"COMMUNITY SUMMARY:\n{community_summary}\n\n"
            f"KEY ENTITIES: {', '.join(key_entities[:5]) if key_entities else 'No specific entities identified'}\n\n"
            f"TASK: Generate exactly 3 strategic follow-up questions for DRIFT search.\n\n"
            f"OUTPUT FORMAT (strict JSON):\n"
            f"[\n"
            f"    {{\n"
            f"        \"question\": \"Specific, actionable question about the community\",\n"
            f"        \"relevance_score\": 0.85,\n"
            f"        \"search_type\": \"local\",\n"
            f"        \"target_entities\": [\"entity1\", \"entity2\"]\n"
            f"    }}\n"
            f"]\n\n"
            f"VALIDATION REQUIREMENTS:\n"
            f"- question: Must be a clear, specific question that expands community understanding\n"
            f"- relevance_score: Float 0.0-1.0 based on potential value for knowledge expansion\n"
            f"- search_type: Must be one of 'local', 'relationship', or 'global'\n"
            f"- target_entities: Array of relevant entity names from the key entities list\n\n"
            f"IMPORTANT: Respond with ONLY the JSON array. No markdown formatting, no explanations, no code blocks."
        )
        
        try:
            # Use existing LLM infrastructure
            if self.llm_provider == "cerebras":
                response = self._cerebras_inference(system_prompt, user_prompt)
            else:
                response = self._gemini_inference(system_prompt, user_prompt)
            
            # Parse LLM response to structured questions
            questions = self._parse_questions_response(response, key_entities)
            return questions
            
        except Exception as e:
            logger.error(f"❌ Question generation failed for comm-{comm_id}: {e}")
            return []
    
    def _parse_questions_response(self, response: str, key_entities: List[str]) -> List[Dict]:
        """Parse LLM response into structured questions using robust multi-strategy approach"""
        try:
            # Calculate dynamic default relevance based on community statistics
            total_nodes = self.nx_graph.number_of_nodes() if hasattr(self, 'nx_graph') else 100
            node_density = min(1.0, total_nodes / 500)  # Scale 0-1
            default_relevance = round(0.5 + (node_density * 0.4), 2)  # Range: 0.5-0.9
            max_questions = max(2, min(5, len(key_entities) + 1))  # Adaptive question count
            
            # Strategy 1: JSON array extraction with regex
            try:
                import re
                match = re.search(r"(\[\s*\{[\s\S]*?\}\s*\])", response)
                if match:
                    json_str = match.group(1)
                    try:
                        questions = self._smart_json_parse_questions(json_str)
                        if questions:
                            return self._validate_and_normalize_questions(questions, key_entities, default_relevance, max_questions)
                    except ValueError:
                        pass  # Continue to next strategy if JSON parsing fails
            except Exception:
                pass
            
            # Strategy 2: Multiple JSON objects extraction
            try:
                import re
                pattern = r'\{[^{}]*"question"[^{}]*\}'
                matches = re.findall(pattern, response)
                if matches:
                    json_array = "[" + ",".join(matches) + "]"
                    try:
                        questions = self._smart_json_parse_questions(json_array)
                        if questions:
                            return self._validate_and_normalize_questions(questions, key_entities, default_relevance, max_questions)
                    except ValueError:
                        pass  # Continue to next strategy if JSON parsing fails
            except Exception:
                pass
            
            # Strategy 3: Markdown list extraction
            try:
                questions = self._parse_markdown_questions(response, key_entities, default_relevance)
                if questions:
                    return self._validate_and_normalize_questions(questions, key_entities, default_relevance, max_questions)
            except Exception:
                pass
            
            # Strategy 4: Generate default questions based on entities
            return self._generate_default_questions(key_entities, default_relevance, max_questions)
            
        except Exception as e:
            logger.warning(f"⚠️ All question parsing strategies failed: {e}")
            return self._generate_default_questions(key_entities, 0.7, 3)
    
    def _smart_json_parse_questions(self, json_text: str) -> List[Dict]:
        """
        Simple 5-step JSON parsing approach (exactly same as Phase 1)
        """
        cleaned_text = json_text.strip()
        
        # Step 1: orjson
        try:
            result = orjson.loads(cleaned_text.encode('utf-8'))
            logger.debug("✅ Step 1: orjson succeeded")
            return result
        except Exception as e:
            logger.debug(f"❌ Step 1: orjson failed - {e}")
        
        # Step 2: json-repair
        try:
            repaired = repair_json(cleaned_text)
            result = orjson.loads(repaired.encode('utf-8'))
            logger.debug("✅ Step 2: json-repair + orjson succeeded")
            return result
        except Exception as e:
            logger.debug(f"❌ Step 2: json-repair failed - {e}")
        
        # Step 3: standard json
        try:
            result = json.loads(cleaned_text)
            logger.debug("✅ Step 3: standard json succeeded")
            return result
        except Exception as e:
            logger.debug(f"❌ Step 3: standard json failed - {e}")
        
        # Step 4: json-repair + standard json
        try:
            repaired = repair_json(cleaned_text)
            result = json.loads(repaired)
            logger.debug("✅ Step 4: json-repair + standard json succeeded")
            return result
        except Exception as e:
            logger.debug(f"❌ Step 4: json-repair + standard json failed - {e}")
        
        # Step 5: All failed - this will trigger save failed txt files
        raise ValueError("All 4 JSON parsing steps failed")
    
    def _parse_markdown_questions(self, response: str, key_entities: List[str], default_relevance: float) -> List[Dict]:
        """Parse questions from markdown or plain text format"""
        questions = []
        
        # Look for numbered lists or bullet points
        import re
        patterns = [
            r'\d+\.\s*(.+?)(?=\n\d+\.|\n-|\n\*|$)',  # Numbered list
            r'-\s*(.+?)(?=\n-|\n\*|\n\d+\.|$)',      # Dash list
            r'\*\s*(.+?)(?=\n\*|\n-|\n\d+\.|$)'      # Asterisk list
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
            if matches and len(matches) >= 2:
                for i, match in enumerate(matches[:5]):  # Max 5 questions
                    question_text = match.strip().replace('\n', ' ')
                    if len(question_text) > 10:  # Reasonable question length
                        search_type = 'global' if any(word in question_text.lower() 
                                                    for word in ['analyze', 'compare', 'overall', 'trends']) else 'local'
                        questions.append({
                            'question': question_text,
                            'relevance_score': max(0.6, default_relevance - (i * 0.1)),
                            'search_type': search_type,
                            'target_entities': key_entities[:2] if key_entities else []
                        })
                break
        
        return questions
    
    def _generate_default_questions(self, key_entities: List[str], default_relevance: float, max_questions: int) -> List[Dict]:
        """Generate default questions when parsing fails"""
        if not key_entities:
            return []
        
        # Template questions based on entity analysis
        question_templates = [
            ("What is {entity} and what role does it play?", "local"),
            ("How does {entity} relate to other entities in this community?", "relationship"), 
            ("What are the key characteristics and properties of {entity}?", "local"),
            ("What trends or patterns involve {entity}?", "global"),
            ("How might {entity} impact the broader context?", "global")
        ]
        
        questions = []
        entities_to_use = key_entities[:max_questions]
        
        for i, entity in enumerate(entities_to_use):
            if i < len(question_templates):
                template, search_type = question_templates[i]
                question = template.format(entity=entity)
                questions.append({
                    'question': question,
                    'relevance_score': max(0.6, default_relevance - (i * 0.05)),
                    'search_type': search_type,
                    'target_entities': [entity]
                })
        
        return questions
    
    def _validate_and_normalize_questions(self, questions: List[Dict], key_entities: List[str], 
                                        default_relevance: float, max_questions: int) -> List[Dict]:
        """Validate and normalize question format"""
        normalized = []
        
        for q in questions:
            if not isinstance(q, dict):
                continue
                
            # Extract question text
            question = q.get('question') or q.get('q') or q.get('text')
            if not question or len(str(question).strip()) < 5:
                continue
            
            # Extract and validate relevance score
            relevance = q.get('relevance_score', default_relevance)
            try:
                relevance = float(relevance)
                if relevance <= 0 or relevance > 1:
                    relevance = default_relevance
            except (ValueError, TypeError):
                relevance = default_relevance
            
            # Extract and validate search type
            search_type = q.get('search_type', 'local')
            if search_type not in ('local', 'relationship', 'global'):
                search_type = 'local'
            
            # Extract target entities
            target_entities = q.get('target_entities', [])
            if not isinstance(target_entities, list):
                target_entities = []
            
            # Ensure we have some target entities
            if not target_entities and key_entities:
                target_entities = key_entities[:2]
            
            normalized.append({
                'question': str(question).strip(),
                'relevance_score': round(relevance, 2),
                'search_type': search_type,
                'target_entities': target_entities
            })
            
            if len(normalized) >= max_questions:
                break
        
        return normalized
    
    # STEP 10: Main Processing Entry Point
    def generate_summaries(self, input_path: str = None, output_path: str = None) -> bool:
        """Main entry point for Phase 3"""
        if output_path is None:
            output_path = "workspace/graph_data/graph-data-final.json"
        
        logger.info("🚀 Starting Phase 3: Community Summarization")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Load Phase 2 output
        if not self.load_graph_data(input_path):
            return False
        
        # Step 2: Build NetworkX graph
        self.nx_graph = self._build_networkx_graph()
        
        # Step 3: Extract community assignments
        self.community_assignments = self._extract_community_assignments()
        
        # Step 4: Generate LLM summaries
        community_summaries = self._generate_community_summaries()
        
        # Step 5: Identify key entities
        key_entities = self._identify_key_entities()
        
        # Step 6: Create community nodes
        community_nodes = self._create_community_nodes(community_summaries, key_entities)
        
        # Step 7: Create IN_COMMUNITY relationships
        community_relationships = self._create_in_community_relationships(community_nodes)
        
        # Step 8: Merge everything
        self.graph_data["nodes"].extend(community_nodes)
        self.graph_data["relationships"].extend(community_relationships)
        
        # Step 9: Add communities section
        self.graph_data["communities"] = {
            "algorithm": "Leiden",
            "total_communities": len(community_summaries),
            "modularity_score": self.graph_data["metadata"]["community_detection"]["modularity_score"],
            "summaries": {
                f"comm-{k}": v for k, v in community_summaries.items()
            }
        }
        
        # Step 10: Generate DRIFT search metadata
        drift_metadata = self._generate_drift_metadata(community_summaries, key_entities)
        if drift_metadata:
            self.graph_data["drift_search_metadata"] = drift_metadata
            logger.info("✅ Added DRIFT search metadata to graph data")
        
        # Step 11: Clean up temporary data
        if "community_stats" in self.graph_data:
            del self.graph_data["community_stats"]
        
        # Step 12: Update metadata
        self.graph_data["metadata"]["phase"] = "final"
        self.graph_data["metadata"]["entity_count"] = len([n for n in self.graph_data["nodes"] if "Community" not in n["labels"]])
        self.graph_data["metadata"]["community_count"] = len(community_nodes)
        self.graph_data["metadata"]["total_node_count"] = len(self.graph_data["nodes"])
        self.graph_data["metadata"]["total_relationship_count"] = len(self.graph_data["relationships"])
        
        # Step 13: Save final output
        if self._save_final_output(output_path):
            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"✅ Phase 3 completed successfully in {elapsed:.1f}s")
            logger.info("📊 Final stats:")
            logger.info(f"   - Total nodes: {len(self.graph_data['nodes'])}")
            logger.info(f"   - Entity nodes: {self.graph_data['metadata']['entity_count']}")
            logger.info(f"   - Community nodes: {len(community_nodes)}")
            logger.info(f"   - Total relationships: {len(self.graph_data['relationships'])}")
            logger.info(f"   - Communities with summaries: {len(community_summaries)}")
            logger.info(f"   - Output saved to: {output_path}")
            return True
        else:
            return False
    
    # STEP 14: Save Final Output
    def _save_final_output(self, output_path: str) -> bool:
        """Save graph-data-final.json with DRIFT search metadata"""
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save final output
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.graph_data, f, indent=2, ensure_ascii=False)
            
            # Calculate file size
            output_size = os.path.getsize(output_path)
            output_size_mb = output_size / (1024 * 1024)
            
            logger.info(f"💾 Saved final output: {output_path} ({output_size_mb:.2f} MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving final output: {e}")
            return False


# STEP 15: Main Entry Point
def main():
    """Main function to run Phase 3: Community Summarization with DRIFT Search Metadata"""
    logger.info("🚀 GraphRAG Phase 3: Community Summarization + DRIFT Search Metadata")
    logger.info("   Input: graph-data-phase-2.json (from Phase 2)")
    logger.info("   Output: graph-data-final.json (with DRIFT search metadata)")
    logger.info("")
    
    # Choose LLM provider from environment or default to cerebras
    llm_provider = os.getenv("GRAPH_LLM_PROVIDER", "cerebras").lower()
    logger.info(f"   Using LLM provider: {llm_provider.upper()}")
    
    try:
        # Initialize Phase 3 processor
        processor = GraphBuilderPhase3(llm_provider=llm_provider)
        
        # Generate summaries
        success = processor.generate_summaries()
        
        if success:
            logger.info("")
            logger.info("✅ Phase 3 completed successfully!")
            logger.info("� DRIFT search metadata generated and included")
            logger.info("�📋 Next step: Upload to Neo4j using 3b_save_to_graph_db.py")
            logger.info("   The graph-data-final.json is now ready for Neo4j import with DRIFT capabilities")
            return 0
        else:
            logger.error("")
            logger.error("❌ Phase 3 failed")
            logger.error("   Please check the logs above for details")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Phase 3 pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
