"""
Phase 2: Community Detection using Leiden Algorithm
Loads graph-data-initial.json, runs community detection, saves graph-data-phase-2.json
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

import networkx as nx
import igraph as ig
import leidenalg
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphBuilderPhase2:
    """Phase 2: Detect communities using graph algorithms (NetworkX + Leiden)"""
    
    def __init__(self):
        """Initialize Phase 2 processor"""
        self.graph_data = None
        self.nx_graph = None
        self.community_result = None
        self.community_stats = None
        self.centrality_metrics = None
        
        # Configuration from environment or defaults
        self.min_community_size = int(os.getenv("GRAPH_MIN_COMMUNITY_SIZE", "5"))
        self.leiden_resolution = float(os.getenv("GRAPH_LEIDEN_RESOLUTION", "1.0"))
        self.leiden_iterations = int(os.getenv("GRAPH_LEIDEN_ITERATIONS", "-1"))  # -1 = until convergence
        self.leiden_seed = int(os.getenv("GRAPH_LEIDEN_SEED", "42"))
        
        logger.info("✅ Phase 2 Initialized: Community Detection")
        logger.info(f"   - Min Community Size: {self.min_community_size}")
        logger.info(f"   - Leiden Resolution: {self.leiden_resolution}")
    
    # STEP 1: Load Graph Data from Phase 1
    def load_graph_data(self, input_path: str = None) -> bool:
        """Load graph data from the specified JSON file."""
        if input_path is None:
            input_path = "workspace/graph_data/graph-data-initial.json"
        
        logger.info(f"Loading graph data from {input_path}...")
        
        try:
            input_file = Path(input_path)
            if not input_file.exists():
                logger.error(f"❌ Input file not found: {input_path}")
                logger.warning("   Please run Phase 1 (2b_process_graph_phase1.py) to generate the graph data.")
                return False
            
            with open(input_file, 'r', encoding='utf-8') as f:
                self.graph_data = json.load(f)
            
            node_count = len(self.graph_data.get("nodes", []))
            rel_count = len(self.graph_data.get("relationships", []))
            
            logger.info(f"   - Found {node_count} nodes and {rel_count} relationships")
            
            if node_count == 0:
                logger.error("❌ Graph data is empty. Cannot proceed.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading graph data: {e}")
            return False
    
    # STEP 2: Build NetworkX Graph
    def _build_networkx_graph(self) -> nx.Graph:
        """Convert graph_data JSON to NetworkX graph for analysis"""
        logger.info("Building NetworkX graph from JSON data...")
        
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
                source=properties.get("source", ""),
                confidence=properties.get("confidence", 0.0)
            )
        
        # Add edges with attributes
        for rel in self.graph_data["relationships"]:
            start_node = rel.get("startNode")
            end_node = rel.get("endNode")
            
            # Only add edge if both nodes exist
            if start_node in G.nodes() and end_node in G.nodes():
                G.add_edge(
                    start_node,
                    end_node,
                    type=rel.get("type", "RELATED_TO"),
                    evidence=rel.get("evidence", ""),
                    confidence=rel.get("confidence", 0.0)
                )
        
        logger.info(f"✅ Built NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Log basic graph statistics
        if G.number_of_nodes() > 0:
            density = nx.density(G)
            logger.info(f"📊 Graph density: {density:.4f}")
            
            if G.number_of_edges() > 0:
                avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
                logger.info(f"📊 Average degree: {avg_degree:.2f}")
        
        return G
    
    # STEP 3: Convert to igraph for Leiden
    def _convert_to_igraph(self, G: nx.Graph) -> ig.Graph:
        """Convert NetworkX graph to igraph for Leiden algorithm"""
        logger.info("🔄 Converting to igraph format for Leiden algorithm...")
        
        # Create mapping from node IDs to indices
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Create edge list with indices
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        
        # Create igraph
        ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=False)
        
        # Add node attributes
        ig_graph.vs["name"] = [G.nodes[node].get("name", "") for node in node_list]
        ig_graph.vs["node_id"] = node_list
        
        logger.info(f"✅ Converted to igraph: {ig_graph.vcount()} vertices, {ig_graph.ecount()} edges")
        
        return ig_graph
    
    # STEP 4: Run Leiden Algorithm
    def _run_leiden_algorithm(self, ig_graph: ig.Graph) -> Dict[str, Any]:
        """Run Leiden algorithm for community detection"""
        logger.info("🔍 Running Leiden community detection algorithm...")
        logger.info(f"Parameters: resolution={self.leiden_resolution}, iterations={self.leiden_iterations}, seed={self.leiden_seed}")
        
        start_time = time.time()
        
        try:
            # Run Leiden algorithm
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                n_iterations=self.leiden_iterations,
                seed=self.leiden_seed
            )
            
            # Extract community assignments
            community_assignments = {}
            for idx, community_id in enumerate(partition.membership):
                node_id = ig_graph.vs[idx]["node_id"]
                community_assignments[node_id] = community_id
            
            # Calculate statistics
            num_communities = len(set(partition.membership))
            modularity = partition.modularity
            
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Leiden algorithm completed in {elapsed:.2f}s")
            logger.info(f"Detected {num_communities} communities")
            logger.info(f"Modularity score: {modularity:.4f}")
            
            return {
                "assignments": community_assignments,
                "num_communities": num_communities,
                "modularity": modularity,
                "algorithm": "Leiden",
                "execution_time": elapsed
            }
            
        except Exception as e:
            logger.error(f"❌ Leiden algorithm failed: {e}")
            raise e
    
    # STEP 5: Calculate Community Statistics
    def _calculate_community_stats(self, G: nx.Graph, community_assignments: Dict[str, int]) -> Dict[int, Dict]:
        """Calculate statistics for each community"""
        logger.info("Calculating community statistics...")
        
        # Group nodes by community
        communities = defaultdict(list)
        for node_id, comm_id in community_assignments.items():
            communities[comm_id].append(node_id)
        
        # Calculate stats for each community
        stats = {}
        for comm_id, node_ids in communities.items():
            # Skip very small communities if configured
            if len(node_ids) < self.min_community_size:
                logger.debug(f"Skipping small community {comm_id} with {len(node_ids)} members")
                continue
            
            subgraph = G.subgraph(node_ids)
            
            stats[comm_id] = {
                "member_count": len(node_ids),
                "internal_edges": subgraph.number_of_edges(),
                "density": nx.density(subgraph) if len(node_ids) > 1 else 0.0,
                "avg_degree": sum(dict(subgraph.degree()).values()) / len(node_ids) if len(node_ids) > 0 else 0.0,
                "member_ids": node_ids[:20]  # Store top 20 for summary generation
            }
        
        logger.info(f"Calculated statistics for {len(stats)} communities (filtered by min_size={self.min_community_size})")
        
        # Log top 5 largest communities
        sorted_communities = sorted(stats.items(), key=lambda x: x[1]["member_count"], reverse=True)
        logger.info("Top 5 largest communities:")
        for comm_id, stat in sorted_communities[:5]:
            logger.info(f"   Community {comm_id}: {stat['member_count']} members, {stat['internal_edges']} edges, density={stat['density']:.3f}")
        
        return stats
    
    # STEP 6: Calculate Centrality Metrics
    def _calculate_centrality_metrics(self, G: nx.Graph) -> Dict[str, Dict]:
        """Calculate centrality metrics for all nodes"""
        logger.info("Calculating node centrality metrics...")
        
        start_time = time.time()
        
        # Degree centrality (fast, always calculate)
        degree_centrality = nx.degree_centrality(G)
        
        # Betweenness centrality (expensive, only for smaller graphs)
        if G.number_of_nodes() < 5000:
            logger.info("   Calculating betweenness centrality...")
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        else:
            logger.info("   Skipping betweenness centrality (graph too large)")
            betweenness_centrality = {node: 0.0 for node in G.nodes()}
        
        # Closeness centrality (expensive, only for smaller graphs)
        if G.number_of_nodes() < 5000:
            logger.info("Calculating closeness centrality...")
            closeness_centrality = nx.closeness_centrality(G)
        else:
            logger.info("   Skipping closeness centrality (graph too large)")
            closeness_centrality = {node: 0.0 for node in G.nodes()}
        
        # Combine metrics
        centrality_metrics = {}
        for node in G.nodes():
            centrality_metrics[node] = {
                "degree": G.degree(node),
                "degree_centrality": degree_centrality.get(node, 0.0),
                "betweenness_centrality": betweenness_centrality.get(node, 0.0),
                "closeness_centrality": closeness_centrality.get(node, 0.0)
            }
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Calculated centrality for {len(centrality_metrics)} nodes in {elapsed:.2f}s")
        
        return centrality_metrics
    
    # STEP 7: Add Community Data to Nodes
    def _add_community_data_to_nodes(self, community_assignments: Dict[str, int], centrality_metrics: Dict[str, Dict]) -> None:
        """Add community_id and centrality metrics to node properties"""
        logger.info("Adding community assignments and centrality to nodes...")
        
        nodes_updated = 0
        
        for node in self.graph_data["nodes"]:
            node_id = node["id"]
            
            # Add community_id
            if node_id in community_assignments:
                node["properties"]["community_id"] = f"comm-{community_assignments[node_id]}"
                nodes_updated += 1
            
            # Add centrality metrics
            if node_id in centrality_metrics:
                metrics = centrality_metrics[node_id]
                node["properties"]["degree"] = metrics["degree"]
                node["properties"]["degree_centrality"] = round(metrics["degree_centrality"], 4)
                node["properties"]["betweenness_centrality"] = round(metrics["betweenness_centrality"], 4)
                node["properties"]["closeness_centrality"] = round(metrics["closeness_centrality"], 4)
        
        logger.info(f"✅ Updated {nodes_updated} nodes with community and centrality data")
    
    # STEP 8: Main Processing Entry Point
    def run_community_detection(self, input_path: str = None, output_path: str = None) -> bool:
        """Main entry point for Phase 2"""
        if output_path is None:
            output_path = "workspace/graph_data/graph-data-phase-2.json"
        
        logger.info("🚀 Starting Phase 2: Community Detection")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Load Phase 1 output
        if not self.load_graph_data(input_path):
            return False
        
        # Step 2: Build NetworkX graph
        self.nx_graph = self._build_networkx_graph()
        
        if self.nx_graph.number_of_nodes() == 0:
            logger.error("❌ Cannot run community detection on empty graph")
            return False
        
        # Step 3: Convert to igraph
        ig_graph = self._convert_to_igraph(self.nx_graph)
        
        # Step 4: Run Leiden algorithm
        self.community_result = self._run_leiden_algorithm(ig_graph)
        
        # Step 5: Calculate community statistics
        self.community_stats = self._calculate_community_stats(
            self.nx_graph, 
            self.community_result["assignments"]
        )
        
        # Step 6: Calculate centrality metrics
        self.centrality_metrics = self._calculate_centrality_metrics(self.nx_graph)
        
        # Step 7: Add community data to nodes
        self._add_community_data_to_nodes(
            self.community_result["assignments"],
            self.centrality_metrics
        )
        
        # Step 8: Update metadata
        self.graph_data["metadata"]["phase"] = "community_detection"
        self.graph_data["metadata"]["community_detection"] = {
            "algorithm": "Leiden",
            "num_communities": self.community_result["num_communities"],
            "modularity_score": round(self.community_result["modularity"], 4),
            "execution_time_seconds": round(self.community_result["execution_time"], 2),
            "min_community_size": self.min_community_size,
            "resolution": self.leiden_resolution
        }
        
        # Step 9: Add community statistics to output
        self.graph_data["community_stats"] = self.community_stats
        
        # Step 10: Save Phase 2 output
        if self._save_phase2_output(output_path):
            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"✅ Phase 2 completed successfully in {elapsed:.2f}s")
            logger.info(f"Final stats:")
            logger.info(f"   - Communities detected: {self.community_result['num_communities']}")
            logger.info(f"   - Modularity score: {self.community_result['modularity']:.4f}")
            logger.info(f"   - Nodes with community assignments: {len(self.community_result['assignments'])}")
            logger.info(f"   - Output saved to: {output_path}")
            return True
        else:
            return False
    
    # STEP 9: Save Phase 2 Output
    def _save_phase2_output(self, output_path: str) -> bool:
        """Save graph-data-phase-2.json"""
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save Phase 2 output
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.graph_data, f, indent=2, ensure_ascii=False)
            
            # Calculate file size
            output_size = os.path.getsize(output_path)
            output_size_mb = output_size / (1024 * 1024)
            
            logger.info(f"Saved Phase 2 output: {output_path} ({output_size_mb:.2f} MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving Phase 2 output: {e}")
            return False


# STEP 10: Main Entry Point
def main():
    """Main function to run Phase 2: Community Detection"""
    logger.info("🚀 GraphRAG Phase 2: Community Detection")
    logger.info("   Input: graph-data-initial.json (from Phase 1)")
    logger.info("   Output: graph-data-phase-2.json")
    logger.info("")
    
    try:
        # Initialize Phase 2 processor
        processor = GraphBuilderPhase2()
        
        # Run community detection
        success = processor.run_community_detection()
        
        if success:
            logger.info("")
            logger.info("✅ Phase 2 completed successfully!")
            logger.info("Next step: Run Phase 3 (2b_process_graph_phase3.py) for community summarization")
            return 0
        else:
            logger.error("")
            logger.error("❌ Phase 2 failed")
            logger.error("   Please check the logs above for details")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Phase 2 pipeline failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
