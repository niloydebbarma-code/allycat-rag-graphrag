import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional
from my_config import MY_CONFIG
from neo4j import GraphDatabase, Driver
from tqdm import tqdm
from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_DATA_DIR = MY_CONFIG.GRAPH_DATA_DIR
GRAPH_DATA_FILE = os.path.join(GRAPH_DATA_DIR, "graph-data-final.json")

class Neo4jConnection:
    def __init__(self):
        self.uri = MY_CONFIG.NEO4J_URI
        self.username = MY_CONFIG.NEO4J_USER
        self.password = MY_CONFIG.NEO4J_PASSWORD
        self.database = getattr(MY_CONFIG, "NEO4J_DATABASE", None)
        if not self.uri:
            raise ValueError("NEO4J_URI config is required")
        if not self.username:
            raise ValueError("NEO4J_USERNAME config is required")
        if not self.password:
            raise ValueError("NEO4J_PASSWORD config is required")
        if not self.database:
            raise ValueError("NEO4J_DATABASE config is required")
        self.driver: Optional[Driver] = None
    
    async def connect(self):
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
                
                await asyncio.get_event_loop().run_in_executor(
                    None, self.driver.verify_connectivity
                )
                logger.info("Connected to Neo4j")
                
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self.driver = None
    
    async def disconnect(self):
        if self.driver:
            await asyncio.get_event_loop().run_in_executor(
                None, self.driver.close
            )
            self.driver = None
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        if not self.driver:
            raise ConnectionError("Not connected to Neo4j database")
        
        def run_query():
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]
                summary = result.consume()
                return records, summary
        
        return await asyncio.get_event_loop().run_in_executor(None, run_query)

neo4j_connection = Neo4jConnection()

app = FastMCP("Neo4j Graph Data Upload Server")

@app.tool()
async def execute_cypher(query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database",
                    "details": "Check connection settings and network connectivity"
                }
        
        records, summary = await neo4j_connection.execute_query(query, parameters)
        
        return {
            "status": "success",
            "query": query,
            "parameters": parameters or {},
            "records": records,
            "record_count": len(records),
            "execution_time_ms": summary.result_available_after,
            "summary": {
                "query_type": summary.query_type,
                "counters": dict(summary.counters) if summary.counters else {}
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "error": str(e)
        }


@app.tool()
async def get_database_schema() -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database"
                }
        
        labels_records, _ = await neo4j_connection.execute_query("CALL db.labels()")
        labels = [record["label"] for record in labels_records]
        
        rel_records, _ = await neo4j_connection.execute_query("CALL db.relationshipTypes()")
        relationships = [record["relationshipType"] for record in rel_records]
        
        prop_records, _ = await neo4j_connection.execute_query("CALL db.propertyKeys()")
        properties = [record["propertyKey"] for record in prop_records]

        try:
            constraint_records, _ = await neo4j_connection.execute_query("SHOW CONSTRAINTS")
            constraints = [dict(record) for record in constraint_records]
        except Exception:
            constraints = []
        
        try:
            index_records, _ = await neo4j_connection.execute_query("SHOW INDEXES")
            indexes = [dict(record) for record in index_records]
        except Exception:
            indexes = []
        
        return {
            "status": "success",
            "schema": {
                "node_labels": labels,
                "relationship_types": relationships,
                "property_keys": properties,
                "constraints": constraints,
                "indexes": indexes
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.tool()
async def get_node_count(label: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database"
                }
        
        if label:
            query = f"MATCH (n:`{label}`) RETURN count(n) as count"
        else:
            query = "MATCH (n) RETURN count(n) as count"
        
        records, _ = await neo4j_connection.execute_query(query)
        count = records[0]["count"] if records else 0
        
        return {
            "status": "success",
            "label": label,
            "count": count
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.tool()
async def get_relationship_count(relationship_type: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database"
                }
        
        if relationship_type:
            query = f"MATCH ()-[r:`{relationship_type}`]-() RETURN count(r) as count"
        else:
            query = "MATCH ()-[r]-() RETURN count(r) as count"
        
        records, _ = await neo4j_connection.execute_query(query)
        count = records[0]["count"] if records else 0
        
        return {
            "status": "success",
            "relationship_type": relationship_type,
            "count": count
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.tool()
async def health_check() -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
        
        if not neo4j_connection.driver:
            return {
                "status": "unhealthy",
                "reason": "Unable to connect to Neo4j database",
                "configuration": {
                    "uri": neo4j_connection.uri,
                    "database": neo4j_connection.database,
                    "username": neo4j_connection.username
                }
            }
        
        # A simple query to test connectivity
        records, _ = await neo4j_connection.execute_query("RETURN 1 as test")
        
        if records and records[0]["test"] == 1:
            return {
                "status": "healthy",
                "database": neo4j_connection.database,
                "uri": neo4j_connection.uri,
                "ssl_enabled": neo4j_connection.uri.startswith(('neo4j+s://', 'bolt+s://')),
                "message": "Neo4j connection is working properly"
            }
        else:
            return {
                "status": "unhealthy",
                "reason": "Query execution failed or returned unexpected results"
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "reason": str(e)
        }


async def clear_database_impl() -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database"
                }
        
        node_count_query = "MATCH (n) RETURN count(n) as count"
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
        
        node_records, _ = await neo4j_connection.execute_query(node_count_query)
        rel_records, _ = await neo4j_connection.execute_query(rel_count_query)
        
        nodes_before = node_records[0]["count"] if node_records else 0
        rels_before = rel_records[0]["count"] if rel_records else 0
        
        await neo4j_connection.execute_query("MATCH ()-[r]->() DELETE r")
        await neo4j_connection.execute_query("MATCH (n) DELETE n")
        
        print(f"✅ Cleared: {nodes_before} nodes, {rels_before} relationships")
        
        return {
            "status": "success",
            "message": "Database cleared successfully",
            "statistics": {
                "nodes_removed": nodes_before,
                "relationships_removed": rels_before
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.tool()
async def clear_database() -> Dict[str, Any]:
    return await clear_database_impl()


async def upload_graph_data_impl() -> Dict[str, Any]:
    try:
        if not neo4j_connection.driver:
            await neo4j_connection.connect()
            if not neo4j_connection.driver:
                return {
                    "status": "error",
                    "error": "Unable to connect to Neo4j database"
                }
        
        clear_result = await clear_database_impl()
        if clear_result["status"] != "success":
            return clear_result
        
        # Check if graph data file exists
        if not os.path.exists(GRAPH_DATA_FILE):
            return {
                "status": "error",
                "error": f"Graph data file not found: {GRAPH_DATA_FILE}"
            }
        
        with open(GRAPH_DATA_FILE, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        if not isinstance(graph_data, dict) or 'nodes' not in graph_data:
            return {
                "status": "error",
                "error": "Invalid graph data format. Expected JSON with 'nodes' array"
            }
        
        nodes = graph_data.get('nodes', [])
        relationships = graph_data.get('relationships', [])
        communities_data = graph_data.get('communities', {})
        drift_metadata = graph_data.get('drift_search_metadata', {})
        global_metadata = graph_data.get('metadata', {})
        search_optimization = drift_metadata.get('search_optimization', {}) if drift_metadata else {}
        
        communities_count = len(drift_metadata.get('community_search_index', {})) if drift_metadata else 0
        drift_count = 1 if drift_metadata else 0
        metadata_count = 1 if global_metadata else 0
        optimization_count = 1 if search_optimization else 0
        communities_metadata_count = 1 if communities_data else 0
        drift_config_count = 1 if (drift_metadata and 'configuration' in drift_metadata) else 0
        community_search_index_count = 1 if (drift_metadata and 'community_search_index' in drift_metadata) else 0
        search_optimization_object_count = 1 if (drift_metadata and 'search_optimization' in drift_metadata) else 0
        embeddings_object_count = 1 if (drift_metadata and 'community_search_index' in drift_metadata) else 0
        embeddings_count = communities_count if (drift_metadata and 'community_search_index' in drift_metadata) else 0
        
        total_items = (len(nodes) + len(relationships) + communities_count + drift_count + 
                      metadata_count + optimization_count + communities_metadata_count + 
                      drift_config_count + community_search_index_count + 
                      search_optimization_object_count + embeddings_object_count + embeddings_count)
        
        print(f"Processing: {len(nodes)} nodes, {len(relationships)} relationships, {communities_count} communities, {total_items - len(nodes) - len(relationships) - communities_count} metadata")
        
        upload_stats = {
            "nodes_processed": 0,
            "nodes_created": 0,
            "relationships_processed": 0,
            "relationships_created": 0,
            "communities_processed": 0,
            "communities_created": 0,
            "drift_metadata_created": 0,
            "global_metadata_created": 0,
            "search_optimization_created": 0,
            "communities_metadata_created": 0,
            "drift_config_created": 0,
            "community_search_index_created": 0,
            "search_optimization_object_created": 0,
            "embeddings_object_created": 0,
            "embeddings_created": 0,
            "errors": []
        }
        
        with tqdm(total=len(nodes), desc="Nodes", unit="node", ncols=80, leave=False) as pbar:
            for node in nodes:
                try:
                    upload_stats["nodes_processed"] += 1
                    
                    node_id = node['id']
                    labels = node['labels']
                    properties = node.get('properties', {})
                    
                    # Create node with labels
                    labels_str = ':'.join([f"`{label}`" for label in labels])
                    query = f"MERGE (n:{labels_str} {{id: $id}}) SET n += $props RETURN n"
                    
                    await neo4j_connection.execute_query(query, {
                        "id": node_id,
                        "props": properties
                    })
                    
                    upload_stats["nodes_created"] += 1
                    pbar.update(1)
                    
                except Exception as e:
                    upload_stats["errors"].append(f"Node upload error: {str(e)}")
                    pbar.update(1)
        
        with tqdm(total=len(relationships), desc="Relationships", unit="rel", ncols=80, leave=False) as pbar:
            for rel in relationships:
                upload_stats["relationships_processed"] += 1

                start_node = rel['startNode']
                end_node = rel['endNode'] 
                rel_type = rel['type']
                
                try:
                    query = f"""
                    MATCH (a {{id: $start_node}})
                    MATCH (b {{id: $end_node}})
                    CREATE (a)-[r:`{rel_type}`]->(b)
                    SET r += $props
                    RETURN r
                    """
                    
                    await neo4j_connection.execute_query(query, {
                        "start_node": start_node,
                        "end_node": end_node,
                        "props": properties
                    })
                    
                    upload_stats["relationships_created"] += 1
                    pbar.update(1)
                    
                except Exception as e:
                    error_msg = f"Relationship upload error for rel {rel}: {str(e)}"
                    logger.error(error_msg)
                    upload_stats["errors"].append(error_msg)
                    pbar.update(1)
        
        if drift_metadata and 'community_search_index' in drift_metadata:
            community_index = drift_metadata['community_search_index']
            
            with tqdm(total=len(community_index), desc="Communities", unit="comm", ncols=80, leave=False) as pbar:
                for comm_id, comm_data in community_index.items():
                    try:
                        upload_stats["communities_processed"] += 1
                        
                        embeddings = comm_data.get('embeddings', {})
                        summary_embedding = embeddings.get('summary_embedding', [])
                        hyde_embeddings = embeddings.get('hyde_embeddings', [])
                        
                        follow_up_templates_json = json.dumps(comm_data.get('follow_up_templates', {}))
                        hyde_embeddings_json = json.dumps(hyde_embeddings)
                        
                        # Get statistics
                        stats = comm_data.get('statistics', {})
                        
                        # Community properties with documented attributes from JSON
                        community_props = {
                            "id": comm_id,
                            "summary": comm_data.get('summary', ''),
                            "key_entities": comm_data.get('key_entities', []),
                            "member_count": stats.get('member_count', 0),
                            "member_ids": stats.get('member_ids', []),
                            "internal_edges": stats.get('internal_edges', 0),
                            "density": stats.get('density', 0.0),
                            "avg_degree": stats.get('avg_degree', 0.0),
                            "follow_up_templates": follow_up_templates_json,
                            "hyde_embeddings": hyde_embeddings_json
                        }
                        
                        # Add summary embedding as List<Float> if available
                        if summary_embedding and isinstance(summary_embedding, list):
                            community_props["summary_embedding"] = summary_embedding
                            community_props["embedding_dimensions"] = len(summary_embedding)
                        
                        # Create Community node
                        query = """
                        MERGE (c:Community {id: $id})
                        SET c += $props
                        RETURN c
                        """
                        
                        await neo4j_connection.execute_query(query, {
                            "id": comm_id,
                            "props": community_props
                        })
                        
                        upload_stats["communities_created"] += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        error_msg = f"Community upload error for {comm_id}: {str(e)}"
                        logger.error(error_msg)
                        upload_stats["errors"].append(error_msg)
                        pbar.update(1)
        
        if drift_metadata:
            try:
                query_routing_config_json = json.dumps(drift_metadata.get('query_routing_config', {}))
                performance_monitoring_json = json.dumps(drift_metadata.get('performance_monitoring', {}))
                configuration_json = json.dumps(drift_metadata.get('configuration', {}))
                community_search_index_json = json.dumps(drift_metadata.get('community_search_index', {}))
                search_optimization_json = json.dumps(drift_metadata.get('search_optimization', {}))

                # Build a compact embeddings object (per-community) to store on the DRIFT node
                embeddings_per_community = {}
                for _comm_id, _comm_data in drift_metadata.get('community_search_index', {}).items():
                    emb = _comm_data.get('embeddings')
                    if emb:
                        # Only keep summary and hyde to limit size
                        embeddings_per_community[_comm_id] = {
                            'summary_embedding': emb.get('summary_embedding'),
                            'hyde_embeddings': emb.get('hyde_embeddings')
                        }

                embeddings_json = json.dumps(embeddings_per_community)

                drift_props = {
                    "version": drift_metadata.get('version', '1.0'),
                    "generated_timestamp": drift_metadata.get('generated_timestamp', ''),
                    "query_routing_config": query_routing_config_json,
                    "performance_monitoring": performance_monitoring_json,
                    "configuration": configuration_json,
                    # Nested objects stored as JSON strings for direct inspection
                    "community_search_index": community_search_index_json,
                    "search_optimization": search_optimization_json,
                    "embeddings": embeddings_json,
                    "total_communities": len(drift_metadata.get('community_search_index', {}))
                }
                
                # Create single DRIFT metadata node
                query = """
                MERGE (d:DriftMetadata {version: $version})
                SET d += $props
                RETURN d
                """
                
                await neo4j_connection.execute_query(query, {
                    "version": drift_metadata.get('version', '1.0'),
                    "props": drift_props
                })
                
                upload_stats["drift_metadata_created"] = 1
                
            except Exception as e:
                error_msg = f"DRIFT metadata upload error: {str(e)}"
                logger.error(error_msg)
                upload_stats["errors"].append(error_msg)
        
        if global_metadata:
            try:
                # Convert nested objects to JSON strings for Neo4j compatibility
                recovery_stats_json = json.dumps(global_metadata.get('recovery_stats', {}))
                member_extraction_stats_json = json.dumps(global_metadata.get('member_extraction_stats', {}))
                community_detection_json = json.dumps(global_metadata.get('community_detection', {}))
                
                metadata_props = {
                    "node_count": global_metadata.get('node_count', 0),
                    "relationship_count": global_metadata.get('relationship_count', 0),
                    "generated_at": global_metadata.get('generated_at', ''),
                    "generator": global_metadata.get('generator', ''),
                    "llm_provider": global_metadata.get('llm_provider', ''),
                    "model": global_metadata.get('model', ''),
                    "format_version": global_metadata.get('format_version', ''),
                    "last_updated": global_metadata.get('last_updated', ''),
                    "phase": global_metadata.get('phase', ''),
                    "entity_count": global_metadata.get('entity_count', global_metadata.get('node_count', 0)),
                    "community_count": global_metadata.get('community_count', 0),
                    "total_node_count": global_metadata.get('total_node_count', global_metadata.get('node_count', 0)),
                    "total_relationship_count": global_metadata.get('total_relationship_count', global_metadata.get('relationship_count', 0)),
                    
                    # Complex nested objects as JSON strings
                    "recovery_stats": recovery_stats_json,
                    "member_extraction_stats": member_extraction_stats_json,
                    "community_detection": community_detection_json
                }
                
                # Create Global Metadata node
                query = """
                MERGE (m:GraphMetadata {generator: $generator})
                SET m += $props
                RETURN m
                """
                
                await neo4j_connection.execute_query(query, {
                    "generator": global_metadata.get('generator', 'unknown'),
                    "props": metadata_props
                })
                
                upload_stats["global_metadata_created"] = 1
                
            except Exception as e:
                error_msg = f"Global metadata upload error: {str(e)}"
                logger.error(error_msg)
                upload_stats["errors"].append(error_msg)
        
        if search_optimization:
            try:
                optimization_props = {
                    "total_communities": search_optimization.get('total_communities', 0),
                    "avg_community_size": search_optimization.get('avg_community_size', 0.0),
                    "graph_density": search_optimization.get('graph_density', 0.0),
                    "total_nodes": search_optimization.get('total_nodes', 0),
                    "total_edges": search_optimization.get('total_edges', 0),
                    "max_primer_communities": search_optimization.get('max_primer_communities', 0)
                }
                
                query = """
                MERGE (s:SearchOptimization {id: 'global'})
                SET s += $props
                RETURN s
                """
                
                await neo4j_connection.execute_query(query, {
                    "props": optimization_props
                })
                
                upload_stats["search_optimization_created"] = 1
                
            except Exception as e:
                error_msg = f"Search optimization upload error: {str(e)}"
                logger.error(error_msg)
                upload_stats["errors"].append(error_msg)
        
        if communities_data:
            try:
                communities_props = {
                    "algorithm": communities_data.get('algorithm', ''),
                    "total_communities": communities_data.get('total_communities', 0),
                    "modularity_score": communities_data.get('modularity_score', 0.0),
                    "summaries": json.dumps(communities_data.get('summaries', {})),
                    "statistics": json.dumps(communities_data.get('statistics', {}))
                }
                
                query = """
                MERGE (cm:CommunitiesMetadata {algorithm: $algorithm})
                SET cm += $props
                RETURN cm
                """
                
                await neo4j_connection.execute_query(query, {
                    "algorithm": communities_data.get('algorithm', 'unknown'),
                    "props": communities_props
                })
                
                upload_stats["communities_metadata_created"] = 1
                
            except Exception as e:
                error_msg = f"Communities metadata upload error: {str(e)}"
                logger.error(error_msg)
                upload_stats["errors"].append(error_msg)
        
        if drift_metadata and 'configuration' in drift_metadata:
            try:
                config = drift_metadata['configuration']
                
                drift_config_props = {
                    "max_iterations": config.get('max_iterations', 0),
                    "confidence_threshold": config.get('confidence_threshold', 0.0),
                    "top_k_communities": config.get('top_k_communities', 0),
                    "hyde_expansion_count": config.get('hyde_expansion_count', 0),
                    "termination_criteria": config.get('termination_criteria', ''),
                    "version": drift_metadata.get('version', '1.0'),
                    "generated_timestamp": drift_metadata.get('generated_timestamp', '')
                }
                
                query = """
                MERGE (dc:DriftConfiguration {version: $version})
                SET dc += $props
                RETURN dc
                """
                
                await neo4j_connection.execute_query(query, {
                    "version": drift_metadata.get('version', '1.0'),
                    "props": drift_config_props
                })
                
                upload_stats["drift_config_created"] = 1
                
            except Exception as e:
                error_msg = f"DRIFT Configuration upload error: {str(e)}"
                logger.error(error_msg)
                upload_stats["errors"].append(error_msg)
        
        if drift_metadata and 'community_search_index' in drift_metadata:
            try:
                community_search_index = drift_metadata['community_search_index']
                
                search_index_props = {
                    "version": drift_metadata.get('version', '1.0'),
                    "total_communities": len(community_search_index),
                    "community_data": json.dumps(community_search_index),
                    "generated_timestamp": drift_metadata.get('generated_timestamp', ''),
                    "index_type": "community_search"
                }
                
                query = """
                MERGE (csi:CommunitySearchIndex {version: $version})
                SET csi += $props
                RETURN csi
                """
                
                await neo4j_connection.execute_query(query, {
                    "version": drift_metadata.get('version', '1.0'),
                    "props": search_index_props
                })
                
                upload_stats["community_search_index_created"] = 1
                
            except Exception as e:
                error_msg = f"Community Search Index upload error: {str(e)}"
                logger.error(error_msg)
                upload_stats["errors"].append(error_msg)
        
        if drift_metadata and 'search_optimization' in drift_metadata:
            try:
                search_opt_data = drift_metadata['search_optimization']
                
                search_opt_props = {
                    "total_communities": search_opt_data.get('total_communities', 0),
                    "avg_community_size": search_opt_data.get('avg_community_size', 0.0),
                    "graph_density": search_opt_data.get('graph_density', 0.0),
                    "total_nodes": search_opt_data.get('total_nodes', 0),
                    "total_edges": search_opt_data.get('total_edges', 0),
                    "max_primer_communities": search_opt_data.get('max_primer_communities', 0),
                    "optimization_version": drift_metadata.get('version', '1.0')
                }
                
                query = """
                MERGE (so:SearchOptimizationObject {optimization_version: $version})
                SET so += $props
                RETURN so
                """
                
                await neo4j_connection.execute_query(query, {
                    "version": drift_metadata.get('version', '1.0'),
                    "props": search_opt_props
                })
                
                upload_stats["search_optimization_object_created"] = 1
                
            except Exception as e:
                error_msg = f"Search Optimization object upload error: {str(e)}"
                logger.error(error_msg)
                upload_stats["errors"].append(error_msg)
        
        if drift_metadata and 'community_search_index' in drift_metadata:
            try:
                community_index = drift_metadata['community_search_index']
                
                total_embeddings = 0
                total_dimensions = 0
                embedding_communities = []
                
                for comm_id, comm_data in community_index.items():
                    embeddings_data = comm_data.get('embeddings', {})
                    if embeddings_data:
                        total_embeddings += 1
                        if embeddings_data.get('summary_embedding'):
                            total_dimensions = len(embeddings_data.get('summary_embedding', []))
                        embedding_communities.append(comm_id)
                
                # Create embeddings object properties
                embeddings_obj_props = {
                    "total_embeddings": total_embeddings,
                    "embedding_dimensions": total_dimensions,
                    "embedding_computation": "computed via text-embedding-ada-002",
                    "communities_with_embeddings": embedding_communities,
                    "embedding_type": "community_summaries",
                    "embeddings_version": drift_metadata.get('version', '1.0')
                }
                
                query = """
                MERGE (eo:EmbeddingsObject {embeddings_version: $version})
                SET eo += $props
                RETURN eo
                """
                
                await neo4j_connection.execute_query(query, {
                    "version": drift_metadata.get('version', '1.0'),
                    "props": embeddings_obj_props
                })
                
                upload_stats["embeddings_object_created"] = 1
                
            except Exception as e:
                error_msg = f"Embeddings object upload error: {str(e)}"
                logger.error(error_msg)
                upload_stats["errors"].append(error_msg)
        
        if drift_metadata and 'community_search_index' in drift_metadata:
            community_index = drift_metadata['community_search_index']
            
            for comm_id, comm_data in community_index.items():
                try:
                    embeddings_data = comm_data.get('embeddings', {})
                    if embeddings_data:
                        embeddings_props = {
                            "community_id": comm_id,
                            "summary_embedding": embeddings_data.get('summary_embedding', []),
                            "hyde_embeddings": json.dumps(embeddings_data.get('hyde_embeddings', [])),
                            "embedding_dimensions": len(embeddings_data.get('summary_embedding', [])),
                            "embedding_computation": embeddings_data.get('embedding_computation', 'computed')
                        }
                        
                        query = """
                        MERGE (e:Embeddings {community_id: $community_id})
                        SET e += $props
                        RETURN e
                        """
                        
                        await neo4j_connection.execute_query(query, {
                            "community_id": comm_id,
                            "props": embeddings_props
                        })
                        
                        upload_stats["embeddings_created"] += 1
                        
                except Exception as e:
                    error_msg = f"Embeddings upload error for {comm_id}: {str(e)}"
                    logger.error(error_msg)
                    upload_stats["errors"].append(error_msg)
        
        if communities_count > 0:
            try:
                # Connect entities to their communities based on community_id property
                community_rel_query = """
                MATCH (n) WHERE n.community_id IS NOT NULL
                MATCH (c:Community {id: n.community_id})
                MERGE (n)-[:BELONGS_TO_COMMUNITY]->(c)
                """
                await neo4j_connection.execute_query(community_rel_query, {})
                
            except Exception as e:
                error_msg = f"Community relationship creation error: {str(e)}"
                logger.error(error_msg)
                upload_stats["errors"].append(error_msg)
        
        # Calculate success percentage for all components
        nodes_success_rate = (upload_stats["nodes_created"] / len(nodes) * 100) if nodes else 100
        rels_success_rate = (upload_stats["relationships_created"] / len(relationships) * 100) if relationships else 100
        communities_success_rate = (upload_stats["communities_created"] / communities_count * 100) if communities_count else 100
        drift_success_rate = (upload_stats["drift_metadata_created"] / drift_count * 100) if drift_count else 100
        
        embedding_dimensions = 0
        if drift_metadata and 'community_search_index' in drift_metadata:
            for comm_data in drift_metadata['community_search_index'].values():
                embeddings = comm_data.get('embeddings', {})
                summary_embedding = embeddings.get('summary_embedding', [])
                if summary_embedding:
                    embedding_dimensions = len(summary_embedding)
                    break
        
        total_created = (upload_stats["nodes_created"] + upload_stats["relationships_created"] + 
                        upload_stats["communities_created"] + upload_stats["drift_metadata_created"] +
                        upload_stats["global_metadata_created"] + upload_stats["search_optimization_created"] +
                        upload_stats["communities_metadata_created"] + upload_stats["drift_config_created"] +
                        upload_stats["community_search_index_created"] + upload_stats["search_optimization_object_created"] +
                        upload_stats["embeddings_object_created"] + upload_stats["embeddings_created"])
        overall_success_rate = (total_created / total_items * 100) if total_items else 100
        
        result = {
            "status": "success",
            "message": "Graph data upload completed successfully",
            "statistics": upload_stats,
            "success_rates": {
                "nodes": f"{nodes_success_rate:.1f}%",
                "relationships": f"{rels_success_rate:.1f}%",
                "communities": f"{communities_success_rate:.1f}%",
                "drift_metadata": f"{drift_success_rate:.1f}%",
                "global_metadata": f"{100.0 if upload_stats['global_metadata_created'] > 0 else 0:.1f}%",
                "search_optimization": f"{100.0 if upload_stats['search_optimization_created'] > 0 else 0:.1f}%",
                "communities_metadata": f"{100.0 if upload_stats['communities_metadata_created'] > 0 else 0:.1f}%",
                "drift_config": f"{100.0 if upload_stats['drift_config_created'] > 0 else 0:.1f}%",
                "community_search_index": f"{100.0 if upload_stats['community_search_index_created'] > 0 else 0:.1f}%",
                "search_optimization_object": f"{100.0 if upload_stats['search_optimization_object_created'] > 0 else 0:.1f}%",
                "embeddings_object": f"{100.0 if upload_stats['embeddings_object_created'] > 0 else 0:.1f}%",
                "embeddings": f"{(upload_stats['embeddings_created']/communities_count*100) if communities_count > 0 else 0:.1f}%",
                "overall": f"{overall_success_rate:.1f}%"
            },
            "source_file": GRAPH_DATA_FILE,
            "architecture_summary": {
                "nodes": f"{upload_stats['nodes_created']}/{len(nodes)}",
                "relationships": f"{upload_stats['relationships_created']}/{len(relationships)}",
                "communities": f"{upload_stats['communities_created']}/{communities_count}",
                "drift_metadata": f"{upload_stats['drift_metadata_created']}/{drift_count}",
                "global_metadata": f"{upload_stats['global_metadata_created']}/1",
                "search_optimization": f"{upload_stats['search_optimization_created']}/1",
                "embeddings_stored": communities_count > 0,
                "vector_dimensions": embedding_dimensions,
                "complete_metadata_coverage": upload_stats['global_metadata_created'] > 0 and upload_stats['search_optimization_created'] > 0
            }
        }
        
        # Print concise upload summary
        print(f"\n✅ Upload completed: {total_created}/{total_items} items ({overall_success_rate:.1f}%)")
        
        # Show all node types created
        total_entity_nodes = upload_stats['nodes_created']
        total_metadata_nodes = (upload_stats['drift_metadata_created'] + 
                               upload_stats['global_metadata_created'] + 
                               upload_stats['search_optimization_created'] +
                               upload_stats['communities_metadata_created'] + 
                               upload_stats['drift_config_created'] +
                               upload_stats['community_search_index_created'] + 
                               upload_stats['search_optimization_object_created'] +
                               upload_stats['embeddings_object_created'])
        
        print(f"   Entity Nodes: {total_entity_nodes}, Community Nodes: {upload_stats['communities_created']}, Metadata Nodes: {total_metadata_nodes}, Embedding Nodes: {upload_stats['embeddings_created']}")
        print(f"   Relationships: {upload_stats['relationships_created']}")
        
        if upload_stats['errors']:
            print(f"   ⚠️  {len(upload_stats['errors'])} errors encountered")
        
        return result
        
    except Exception as e:
        logger.error(f"Graph data upload failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.tool()
async def upload_graph_data() -> Dict[str, Any]:
    return await upload_graph_data_impl()


@app.tool()
async def check_graph_data_file() -> Dict[str, Any]:
    try:
        if not os.path.exists(GRAPH_DATA_FILE):
            return {
                "status": "not_found",
                "path": GRAPH_DATA_FILE,
                "message": "Graph data file does not exist"
            }
        
        # Get file stats
        file_stats = os.stat(GRAPH_DATA_FILE)
        file_size = file_stats.st_size
        
        # Try to parse the JSON to validate format
        try:
            with open(GRAPH_DATA_FILE, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            nodes_count = len(graph_data.get('nodes', []))
            relationships_count = len(graph_data.get('relationships', []))
            
            return {
                "status": "found",
                "path": GRAPH_DATA_FILE,
                "file_size_bytes": file_size,
                "nodes_count": nodes_count,
                "relationships_count": relationships_count,
                "valid_json": True
            }
            
        except json.JSONDecodeError as e:
            return {
                "status": "invalid",
                "path": GRAPH_DATA_FILE,
                "file_size_bytes": file_size,
                "valid_json": False,
                "json_error": str(e)
            }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.tool()
async def get_connection_info() -> Dict[str, Any]:
    try:
        # Always return configuration info even if not connected
        deployment_type = "Self-hosted"
        if "databases.neo4j.io" in neo4j_connection.uri:
            deployment_type = "Neo4j Aura"
        elif "sandbox" in neo4j_connection.uri:
            deployment_type = "Neo4j Sandbox"
        elif any(cloud in neo4j_connection.uri for cloud in ["aws", "gcp", "azure"]):
            deployment_type = "Enterprise Cloud"
        
        connection_info = {
            "status": "success",
            "connection": {
                "uri": neo4j_connection.uri,
                "database": neo4j_connection.database,
                "username": neo4j_connection.username,
                "deployment_type": deployment_type,
                "ssl_enabled": neo4j_connection.uri.startswith(('neo4j+s://', 'bolt+s://')),
                "connected": neo4j_connection.driver is not None
            },
            "capabilities": {
                "cypher_queries": True,
                "schema_inspection": True,
                "bulk_operations": True,
                "graph_algorithms": "unknown",
                "multi_database": "unknown"
            }
        }
        
        if neo4j_connection.driver:
            try:
                server_info_records, _ = await neo4j_connection.execute_query(
                    "CALL dbms.components() YIELD name, versions, edition"
                )
                connection_info["server_info"] = server_info_records[0] if server_info_records else {}
            except Exception:
                connection_info["server_info"] = {}
        
        return connection_info
        
    except Exception as e:
        logger.error(f"Connection info retrieval failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    import sys
    try:
        asyncio.run(neo4j_connection.connect())
        print(f"Looking for graph data at: {GRAPH_DATA_FILE}")
        print(f"File exists: {os.path.exists(GRAPH_DATA_FILE)}")
        
        result = asyncio.run(upload_graph_data_impl())
        print(f"Upload result: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'error':
            print(f"❌ Error details: {result.get('error', 'Unknown error')}")
            if 'error_type' in result:
                print(f"Error type: {result['error_type']}")
        
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.warning(f"Connection Warning: {e}")