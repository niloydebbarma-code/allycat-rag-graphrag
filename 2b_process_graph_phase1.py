"""
GraphRAG Phase 1: LLM-based Entity and Relationship Extraction
Builds initial knowledge graph from markdown files using LLMs (Cerebras or Gemini)
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import orjson
from json_repair import repair_json
import google.generativeai as genai
import openai
from my_config import MY_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphBuilder:
    
    def __init__(self, llm_provider="cerebras"):
        self.llm_provider = llm_provider.lower()
        
        # Global entity registry for deduplication across files
        self.global_entity_registry = {}
        
        # Initialize graph data structure
        self.graph_data = {"nodes": [], "relationships": []}
        self.processed_files = 0
        
        # Initialize LLM API based on provider
        if self.llm_provider == "cerebras":
            if not MY_CONFIG.CEREBRAS_API_KEY:
                raise ValueError("CEREBRAS_API_KEY environment variable not set. Get free key at: https://cloud.cerebras.ai/")
            
            # Configure Cerebras client
            self.cerebras_client = openai.OpenAI(
                api_key=MY_CONFIG.CEREBRAS_API_KEY,
                base_url="https://api.cerebras.ai/v1"
            )
            self.model_name = "llama-4-scout-17b-16e-instruct"  
            logger.info("🚀 Using Cerebras API")
            
        elif self.llm_provider == "gemini":
            if not MY_CONFIG.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY environment variable not set. Get free key at: https://aistudio.google.com/")
            
            # Configure Gemini with FREE tier
            genai.configure(api_key=MY_CONFIG.GEMINI_API_KEY)
            self.model_name = "gemini-1.5-flash" 
            self.gemini_model = genai.GenerativeModel(self.model_name)
            logger.info("🆓 Using Google Gemini API,)")
            
        else:
            valid_providers = ["cerebras", "gemini"]
            raise ValueError(f"Invalid provider '{llm_provider}'. Choose from: {valid_providers}")

        # Configure extraction parameters
        self.min_entities = int(os.getenv("GRAPH_MIN_ENTITIES", "5"))
        self.max_entities = int(os.getenv("GRAPH_MAX_ENTITIES", "15"))
        self.min_relationships = int(os.getenv("GRAPH_MIN_RELATIONSHIPS", "3"))
        self.max_relationships = int(os.getenv("GRAPH_MAX_RELATIONSHIPS", "8"))
        self.min_confidence = float(os.getenv("GRAPH_MIN_CONFIDENCE", "0.8"))
        self.max_content_chars = int(os.getenv("GRAPH_MAX_CONTENT_CHARS", "12000"))
        self.sentence_boundary_ratio = float(os.getenv("GRAPH_SENTENCE_BOUNDARY_RATIO", "0.7"))
        
        logger.info(f"✅ Initialized {self.llm_provider.upper()} provider with model: {self.model_name}")
        logger.info(f"Extraction config: {self.min_entities}-{self.max_entities} entities, {self.min_relationships}-{self.max_relationships} relationships, min confidence: {self.min_confidence}")
        logger.info(f"Content processing: {self.max_content_chars} chars per chunk with overlap for FULL analysis")

    # STEP 0: Clean Graph Data Folder
    def clean_graph_folder(self, graph_dir: str = None):
        if graph_dir is None:
            graph_dir = "workspace/graph_data"
        try:
            graph_path = Path(graph_dir)
            if graph_path.exists():
                # Remove all files in the directory
                for file_path in graph_path.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        logger.debug(f"Removed: {file_path.name}")
                logger.info(f"Cleaned graph folder: {graph_dir}")
            else:
                # Create directory if it doesn't exist
                graph_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created graph folder: {graph_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean graph folder: {e}")

    # STEP 1: Content Preprocessing and Chunking
    def _preprocess_content(self, text: str, max_chars: int = None) -> str:
        # Remove excessive whitespace but keep full content
        text = ' '.join(text.split())
        return text.strip()
    
    def _chunk_content(self, text: str, chunk_size: int = None, overlap: int = 200) -> List[str]:
        if chunk_size is None:
            chunk_size = self.max_content_chars
            
        # If content fits in one chunk, return as-is
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to find good break point (sentence boundary)
            chunk_text = text[start:end]
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            
            # Use best break point
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.7:  # Good break point
                actual_end = start + break_point + 1
                chunks.append(text[start:actual_end])
                start = actual_end - overlap  # Overlap for context
            else:
                # No good break point, use hard split
                chunks.append(text[start:end])
                start = end - overlap
                
        return chunks

    # STEP 2: LLM Prompt Generation
    def get_entity_extraction_prompt(self) -> str:
        return f"""You are a specialized knowledge graph extraction assistant. Your task is to analyze content and extract entities and relationships to build comprehensive knowledge graphs.

DYNAMIC EXTRACTION REQUIREMENTS:
- Extract {self.min_entities}-{self.max_entities} most important entities from the content
- Create {self.min_relationships}-{self.max_relationships} meaningful relationships between entities
- Confidence threshold: {self.min_confidence} (only include high-confidence extractions)
- Focus on extracting diverse entity types relevant to the content domain

CONSTITUTIONAL AI PRINCIPLES:
1. Content-Adaptive: Determine entity types based on content analysis, not predefined categories
2. Relationship-Rich: Focus on meaningful semantic relationships between entities
3. Context-Aware: Consider document context and domain when extracting entities
4. Quality-First: Prioritize extraction quality over quantity

ENTITY EXTRACTION GUIDELINES:
- Identify the most important concepts, terms, people, places, organizations, technologies, events
- Extract entities that would be valuable for knowledge graph queries
- Include both explicit entities (directly mentioned) and implicit entities (strongly implied)
- Assign appropriate types based on semantic analysis of the entity's role in the content

RELATIONSHIP EXTRACTION GUIDELINES:
- Create relationships that capture semantic meaning, not just co-occurrence
- Use descriptive relationship types that express the nature of the connection
- Include hierarchical, associative, and causal relationships where appropriate
- Ensure relationships are bidirectionally meaningful and contextually accurate

OUTPUT FORMAT (strict JSON):
{{
    "entities": [
        {{
            "text": "Entity Name",
            "type": "DynamicType",
            "content": "Comprehensive description of the entity",
            "confidence": 0.95
        }}
    ],
    "relationships": [
        {{
            "startNode": "Entity Name 1",
            "endNode": "Entity Name 2",
            "type": "DESCRIPTIVE_RELATIONSHIP_TYPE",
            "description": "Clear description of the relationship",
            "evidence": "Direct evidence from text supporting this relationship",
            "confidence": 0.90
        }}
    ]
}}

IMPORTANT: Respond with ONLY the JSON object. No explanations, no markdown formatting, no code blocks."""

    # STEP 3: LLM Inference Methods  
    def _cerebras_inference(self, system_prompt: str, user_prompt: str) -> str:
        try:
            # Cerebras uses OpenAI-compatible chat format
            response = self.cerebras_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Check for empty response 
            if not response or not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from Cerebras")
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Check for quota/rate limit exceeded errors
            error_str = str(e).lower()
            if "429" in str(e) and "quota" in error_str:
                logger.error(f"🚫 QUOTA EXCEEDED: Cerebras API rate/quota limit reached - {e}")
                raise Exception("QUOTA_EXCEEDED") from e
            else:
                logger.error(f"Error with Cerebras inference: {e}")
                raise e
    
    def _gemini_inference(self, system_prompt: str, user_prompt: str) -> str:
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.gemini_model.generate_content(combined_prompt)    
            if not response or not response.text:
                raise ValueError("Empty response from Gemini")
                
            return response.text.strip()
            
        except Exception as e:
            # Check for quota exceeded error
            if "429" in str(e) and "quota" in str(e).lower():
                logger.error(f"🚫 QUOTA EXCEEDED: Gemini API daily limit reached - {e}")
                raise Exception("QUOTA_EXCEEDED") from e
            else:
                logger.error(f"Error with Gemini inference: {e}")
                raise e
    
    # STEP 4: JSON Parsing Pipeline
    def _smart_json_parse(self, json_text: str) -> Dict[str, Any]:
        
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

    # STEP 5: Response Parsing and Validation
    def _parse_llm_extraction_response(self, llm_response: str, file_name: str) -> Dict[str, Any]:
        
        # Clean up response first
        cleaned_response = llm_response.strip()
        
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
        
        # Use the 5-step JSON parsing pipeline
        try:
            extraction_data = self._smart_json_parse(cleaned_response)
            
            # Validate complete format
            if self._validate_complete_format(extraction_data):
                return extraction_data
            else:
                self._save_failed_response(cleaned_response, file_name, "Format validation failed", "Missing required fields or empty values")
                return None
        except Exception as e:
            logger.error(f"❌ All JSON parsing steps failed for file {file_name}: {str(e)}")
            self._save_failed_response(cleaned_response, file_name, "All parsing steps failed", str(e))
            return None
    
    # STEP 6: Format Validation
    def _validate_complete_format(self, extraction_data: Dict[str, Any]) -> bool:

        if not isinstance(extraction_data, dict):
            return False

        if "entities" not in extraction_data or "relationships" not in extraction_data:
            return False

        entities = extraction_data.get("entities", [])
        relationships = extraction_data.get("relationships", [])
        if not isinstance(entities, list) or len(entities) == 0:
            return False
        for entity in entities:
            if not isinstance(entity, dict):
                return False
            
            required_fields = ["text", "type", "content", "confidence"]
            for field in required_fields:
                if field not in entity:
                    return False
                value = entity[field]
                if value is None or value == "" or (isinstance(value, str) and not value.strip()):
                    return False
                
            if not isinstance(entity["confidence"], (int, float)) or entity["confidence"] <= 0:
                return False
        
        if isinstance(relationships, list):
            for rel in relationships:
                if not isinstance(rel, dict):
                    return False
                
                required_fields = ["startNode", "endNode", "type", "description", "evidence", "confidence"]
                for field in required_fields:
                    if field not in rel:
                        return False
                    value = rel[field]
                    if value is None or value == "" or (isinstance(value, str) and not value.strip()):
                        return False
                
                if not isinstance(rel["confidence"], (int, float)) or rel["confidence"] <= 0:
                    return False
        
        return True
    
    # STEP 7: Error Handling and Failed Response Logging
    def _save_failed_response(self, llm_response: str, file_name: str, _json_error: str, _repair_error: str):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output_dir = Path("workspace/graph_data")
            output_dir.mkdir(parents=True, exist_ok=True)
    
            with open(output_dir / "failed_responses.txt", 'a', encoding='utf-8') as f:
                f.write(f"# Failed response from file: {file_name} at {timestamp}\n")
                f.write(llm_response)
                f.write("\n---\n") 
                f.flush()  
                
        except Exception as save_error:
            logger.error(f"Failed to save failed response from {file_name}: {save_error}")
            
    # STEP 8: Main Entity Extraction
    def extract_entities_with_llm(self, content: str, file_name: str) -> Dict[str, Any]:
        # Preprocess content
        processed_content = self._preprocess_content(content)
        
        # Split into chunks 
        chunks = self._chunk_content(processed_content)
        
        logger.info(f"📄 Processing {file_name}: {len(processed_content)} chars in {len(chunks)} chunk(s)")
        
        # Collect all entities and relationships from chunks
        all_entities = []
        all_relationships = []
        
        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"🔄 Processing chunk {chunk_idx + 1}/{len(chunks)} for {file_name}")
            
            # Simple retry mechanism for empty content - just send to LLM again
            max_retries = 3
            for attempt in range(max_retries):
                # Get the optimized prompt for entity extraction based on provider
                system_prompt = self.get_entity_extraction_prompt()
                
                # Create user prompt with chunk content
                chunk_info = f" (chunk {chunk_idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                user_prompt = f"""
                Analyze the following content from file "{file_name}"{chunk_info}:
                
                ```
                {chunk}
                ```
                
                Extract all relevant entities, concepts, and their relationships from this content.
                """
                
                # Call appropriate LLM API
                try:
                    if self.llm_provider == "gemini":
                        llm_response = self._gemini_inference(system_prompt, user_prompt)
                    elif self.llm_provider == "cerebras":
                        llm_response = self._cerebras_inference(system_prompt, user_prompt)
                    else:
                        raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
                except Exception as e:
                    if "QUOTA_EXCEEDED" in str(e):
                        logger.error(f"🚫 QUOTA EXCEEDED on file {file_name}, chunk {chunk_idx + 1} - stopping processing")
                        # Return partial results if we have any
                        return {
                            "entities": all_entities,
                            "relationships": all_relationships,
                            "file": file_name,
                            "structure": {"section": "partial_quota_exceeded"},
                            "chunks_processed": chunk_idx,
                            "total_content_length": len(processed_content),
                            "quota_exceeded": True
                        }
                    else:
                        raise e
                    
                # Parse the JSON response
                result = self._parse_llm_extraction_response(llm_response, f"{file_name}_chunk_{chunk_idx}")
                if result is not None or attempt == max_retries - 1:
                    if result is None:
                        logger.warning(f"❌ Chunk {chunk_idx + 1} of {file_name} failed all validation attempts, skipping")
                        break

                    # Chunk results to collections
                    chunk_entities = result.get("entities", [])
                    chunk_relationships = result.get("relationships", [])
                    
                    # Add chunk identifier to entities for deduplication
                    for entity in chunk_entities:
                        entity["chunk_id"] = chunk_idx
                        entity["source_chunk"] = f"chunk_{chunk_idx}"
                    
                    # Add chunk identifier to relationships
                    for rel in chunk_relationships:
                        rel["chunk_id"] = chunk_idx
                        rel["source_chunk"] = f"chunk_{chunk_idx}"
                    
                    all_entities.extend(chunk_entities)
                    all_relationships.extend(chunk_relationships)
                    
                    logger.info(f"✅ Chunk {chunk_idx + 1}: {len(chunk_entities)} entities, {len(chunk_relationships)} relationships")
                    break
                else:
                    logger.info(f"Chunk {chunk_idx + 1} attempt {attempt + 1}/{max_retries}: Validation failed, retrying")
        
        # Deduplicate entities across chunks (same entity name = same entity)
        unique_entities = {}
        for entity in all_entities:
            entity_key = entity.get("text", "").lower().strip()
            if entity_key and entity_key not in unique_entities:
                unique_entities[entity_key] = entity
            elif entity_key:
                # Merge information from duplicate entities
                existing = unique_entities[entity_key]
                existing["confidence"] = max(existing.get("confidence", 0), entity.get("confidence", 0))
                # Combine descriptions
                existing_desc = existing.get("content", "")
                new_desc = entity.get("content", "")
                if new_desc and new_desc not in existing_desc:
                    existing["content"] = f"{existing_desc}; {new_desc}".strip("; ")
        
        # Deduplicate relationships (same startNode+endNode+type = same relationship)
        unique_relationships = {}
        for rel in all_relationships:
            rel_key = f"{rel.get('startNode', '').lower()}||{rel.get('endNode', '').lower()}||{rel.get('type', '').lower()}"
            if rel_key and rel_key not in unique_relationships:
                unique_relationships[rel_key] = rel
            elif rel_key:
                # Keep highest confidence relationship
                existing = unique_relationships[rel_key]
                if rel.get("confidence", 0) > existing.get("confidence", 0):
                    unique_relationships[rel_key] = rel
        
        final_entities = list(unique_entities.values())
        final_relationships = list(unique_relationships.values())
        
        logger.info(f"Final results for {file_name}: {len(final_entities)} unique entities, {len(final_relationships)} unique relationships")
        
        return {
            "entities": final_entities,
            "relationships": final_relationships,
            "file": file_name,
            "structure": {"section": "full_analysis"},
            "chunks_processed": len(chunks),
            "total_content_length": len(processed_content)
        }
    

    
    # STEP 9: Single File Processing
    def process_md_file(self, md_file_path: str) -> Dict[str, Any]:
        logger.info(f"Processing: {md_file_path}")
        
        try:
            # Read file content
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_name = os.path.basename(md_file_path)
            
            # Extract entities and relationships using LLM-only approach
            llm_data = self.extract_entities_with_llm(content, file_name)
            
            # Use LLM data - create nodes and relationships from validated data
            entities_added = 0
            relationships_added = 0
            
            # Check if quota was exceeded during extraction
            quota_exceeded = llm_data.get("quota_exceeded", False)
            if quota_exceeded:
                return {
                    "file": file_name,
                    "status": "quota_exceeded",
                    "entities_extracted": len(llm_data.get("entities", [])),
                    "unique_entities_added": 0,
                    "relationships_generated": 0,
                    "processed_at": datetime.now().isoformat(),
                    "error": "API quota exceeded during processing"
                }
            
            # Process entities from LLM
            for entity in llm_data.get("entities", []):
                entity_text = entity["text"]
                semantic_key = entity_text.lower().strip()
                
                # Add to global registry if new
                if semantic_key not in self.global_entity_registry:
                    # Use LLM data directly
                    entity["id"] = str(uuid.uuid4())
                    entity["source_file"] = file_name
                    
                    self.global_entity_registry[semantic_key] = entity
                    self.graph_data["nodes"].append(entity)
                    entities_added += 1
            
            # Process relationships from LLM 
            for rel in llm_data.get("relationships", []):
                # Apply confidence threshold filtering
                rel_confidence = rel.get("confidence", 0.0)
                if rel_confidence < self.min_confidence:
                    continue  # Skip low-confidence relationships
                
                start_text = rel["startNode"].lower().strip()
                end_text = rel["endNode"].lower().strip()
                
                # Only create if both entities exist
                if start_text in self.global_entity_registry and end_text in self.global_entity_registry:
                    # Use original relationship type without sanitization
                    original_type = rel["type"]
                    
                    # Create clean relationship with only Neo4j fields
                    clean_rel = {
                        "id": str(uuid.uuid4()),
                        "startNode": self.global_entity_registry[start_text]["id"],
                        "endNode": self.global_entity_registry[end_text]["id"],
                        "type": original_type,  # Use original type preserving semantic meaning
                        "description": rel.get("description", ""),
                        "evidence": rel.get("evidence", ""),
                        "confidence": rel_confidence,
                        "chunk_id": rel.get("chunk_id", 0),
                        "source_chunk": rel.get("source_chunk", ""),
                        "source_file": file_name
                    }
                    
                    self.graph_data["relationships"].append(clean_rel)
                    relationships_added += 1
            
            result = {
                "file": file_name,
                "status": "success",
                "entities_extracted": len(llm_data.get("entities", [])),
                "unique_entities_added": entities_added,
                "relationships_generated": relationships_added,
                "processed_at": datetime.now().isoformat()
            }
            
            self.processed_files += 1
            logger.info(f"✅ Processed {file_name}: {entities_added} new entities, {relationships_added} relationships")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing {md_file_path}: {e}")
            return {
                "file": os.path.basename(md_file_path),
                "status": "error",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }
    
    # STEP 10: Batch File Processing
    def process_all_md_files(self, input_dir: str = None, output_path: str = None) -> Dict[str, Any]:
        if input_dir is None:
            input_dir = "workspace/processed"
        if output_path is None:
            output_path = os.path.join("workspace/graph_data", "graph-data-initial.json")
        
        # Clean the graph folder before starting fresh processing
        graph_dir = os.path.dirname(output_path)
        self.clean_graph_folder(graph_dir)
        
        input_path = Path(input_dir)
        md_files = list(input_path.glob("**/*.md"))  # Include subdirectories
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not md_files:
            logger.warning(f"No markdown files found in {input_dir}")
            return {"status": "no_files", "message": "No markdown files found"}
        
        logger.info(f"Found {len(md_files)} markdown files to process")
        
        # Reset data structures for a clean batch processing
        self.graph_data = {"nodes": [], "relationships": []}
        self.global_entity_registry = {}  # Reset global registry
        self.processed_files = 0
        
        logger.info(f"🚀 Starting document processing with Neo4j format output ({self.llm_provider.upper()})...")
        
        # Process files with progress tracking
        results = []
        processed_successfully = []
        failed_files = []
        quota_exceeded_files = []
        start_time = time.time()
        
        for i, md_file in enumerate(md_files, 1):
            file_start_time = time.time()
            logger.info(f"Processing file {i}/{len(md_files)}: {md_file.name}")
            
            # Track registry size before processing
            initial_registry_size = len(self.global_entity_registry)
            initial_relationship_count = len(self.graph_data["relationships"])
            
            # Process the file
            result = self.process_md_file(str(md_file))
            results.append(result)
            
            # Track file status for detailed logging
            file_status = result.get("status", "unknown")
            if file_status == "success":
                processed_successfully.append(md_file.name)
            elif file_status == "quota_exceeded":
                quota_exceeded_files.append(md_file.name)
                logger.warning(f"🚫 QUOTA EXCEEDED - Stopping batch processing at file {i}/{len(md_files)}")
                break  # Stop processing when quota exceeded
            else:
                failed_files.append((md_file.name, result.get("error", "Unknown error")))
            
            # Calculate processing metrics
            file_time = time.time() - file_start_time
            new_entities = len(self.global_entity_registry) - initial_registry_size
            new_relationships = len(self.graph_data["relationships"]) - initial_relationship_count
            
            # Show detailed progress information
            logger.info(f" File processed in {file_time:.2f}s: {new_entities} new entities, {new_relationships} relationships")
            
            # Show batch progress at regular intervals
            if i % 5 == 0 or i == len(md_files):
                successful_so_far = sum(1 for r in results if r.get("status") == "success")
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (len(md_files) - i)
                
                logger.info(f"Progress: {i}/{len(md_files)} files ({successful_so_far} successful)")
                logger.info(f" Current stats: {len(self.global_entity_registry)} unique entities, {len(self.graph_data['relationships'])} relationships")
                logger.info(f"Time elapsed: {elapsed:.1f}s (avg {avg_time:.1f}s per file, ~{remaining:.1f}s remaining)")
        
        # Generate comprehensive summary with detailed tracking
        elapsed = time.time() - start_time
        successful = len(processed_successfully)
        quota_exceeded = len(quota_exceeded_files)
        failed = len(failed_files)
        unique_entities = len(self.global_entity_registry)
        
        # Save detailed processing lists
        self._save_processing_logs(processed_successfully, quota_exceeded_files, failed_files, output_path)
        
        # Count entity types
        entity_types = {}
        for entity_info in self.global_entity_registry.values():
            entity_type = entity_info["type"]
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Count relationship types
        relationship_types = {}
        for rel in self.graph_data["relationships"]:
            rel_type = rel["type"]
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        summary = {
            "status": "completed",
            "total_files": len(md_files),
            "successful": successful,
            "quota_exceeded": quota_exceeded,
            "failed": failed,
            "unique_entities": unique_entities,
            "total_relationships": len(self.graph_data["relationships"]),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "processing_time_seconds": elapsed,
            "average_time_per_file": elapsed / len(md_files) if md_files else 0,
            "model": self.model_name,
            "llm_provider": self.llm_provider,
            "processed_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Processing complete in {elapsed:.1f}s: {successful}/{len(md_files)} files successful")
        if quota_exceeded > 0:
            logger.warning(f"🚫 {quota_exceeded} files hit quota limit")
        if failed > 0:
            logger.error(f"❌ {failed} files failed with errors")
        logger.info(f"Final stats: {unique_entities} unique entities, {len(self.graph_data['relationships'])} relationships")
        
        # Log entity and relationship type breakdown
        logger.info("Entity types:")
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  - {entity_type}: {count}")
        
        logger.info("Relationship types:")
        for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  - {rel_type}: {count}")
            
            return summary
    
    # STEP 10.5: Processing Logs Tracking
    def _save_processing_logs(self, successful_files: List[str], quota_exceeded_files: List[str], failed_files: List[tuple], output_path: str):
        try:
            output_dir = Path(output_path).parent
            
            # Save successfully processed files
            with open(output_dir / "processed_successfully.txt", 'w', encoding='utf-8') as f:
                f.write(f"# Successfully Processed Files ({len(successful_files)} total)\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                for file_name in successful_files:
                    f.write(f"{file_name}\n")
            
            # Save quota exceeded files
            if quota_exceeded_files:
                with open(output_dir / "quota_exceeded_files.txt", 'w', encoding='utf-8') as f:
                    f.write(f"# Files That Hit Quota Limit ({len(quota_exceeded_files)} total)\n")
                    f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                    for file_name in quota_exceeded_files:
                        f.write(f"{file_name}\n")
            
            # Save failed files with errors
            if failed_files:
                with open(output_dir / "failed_files.txt", 'w', encoding='utf-8') as f:
                    f.write(f"# Files That Failed Processing ({len(failed_files)} total)\n")
                    f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                    for file_name, error in failed_files:
                        f.write(f"{file_name}: {error}\n")
                        
            logger.info(f"📋 Processing logs saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save processing logs: {e}")
    
    # STEP 11: Graph Data Output
    def save_graph_data(self, output_path: str = None) -> bool:
        if output_path is None:
            output_path = os.path.join("workspace/graph_data", "graph-data-initial.json")
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Compile final data from global entity registry
            final_nodes = []
            
            for semantic_key, entity_info in self.global_entity_registry.items():
                entity_id = entity_info["id"]
                
                # Create Neo4j node
                node = {
                    "id": entity_id,
                    "elementId": entity_id,
                    "labels": [entity_info["type"]], 
                    "properties": {
                        "name": entity_info["text"],
                        "content": entity_info.get("content", ""),
                        "source": entity_info.get("source_file", ""),
                        "confidence": entity_info["confidence"],
                        "created_date": datetime.now().strftime("%Y-%m-%d"),
                        "extraction_method": self.llm_provider
                    }
                }
                final_nodes.append(node)
            
            # Use relationships
            final_relationships = self.graph_data["relationships"]
            
            # Prepare final graph data
            final_graph = {
                "nodes": final_nodes,
                "relationships": final_relationships,
                "metadata": {
                    "node_count": len(final_nodes),
                    "relationship_count": len(final_relationships),
                    "generated_at": datetime.now().isoformat(),
                    "generator": "Allycat GraphBuilder",
                    "llm_provider": self.llm_provider,
                    "model": self.model_name,
                    "format_version": "neo4j-2025"
                }
            }
            
            # Save final graph data
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_graph, f, indent=2, ensure_ascii=False)
            
            # Calculate final output size
            output_size = os.path.getsize(output_path)
            output_size_mb = output_size / (1024 * 1024)
            
            logger.info(f"✅ Neo4j graph data saved to {output_path} ({output_size_mb:.2f} MB)")
            logger.info(f"Final stats: {len(final_nodes)} nodes, {len(final_relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving graph data: {e}")
            return False
    
# STEP 12: Main Entry Point
def main():
    """Main function to run the content analysis pipeline."""
    logger.info(" Starting Content Analysis Pipeline (Cloud-based APIs)")
    
    # Choose LLM provider from environment or default to cerebras
    llm_provider = os.getenv("GRAPH_LLM_PROVIDER", "cerebras").lower()
    logger.info(f" Using LLM provider: {llm_provider.upper()}")
    
    # Validate provider choice
    valid_providers = ["cerebras", "gemini"]
    if llm_provider not in valid_providers:
        logger.warning(f"⚠️ Invalid provider '{llm_provider}'. Using 'cerebras' (default)")
        llm_provider = "cerebras"
    
    try:
        analyzer = GraphBuilder(llm_provider=llm_provider)
        
        # Normal processing
        summary = analyzer.process_all_md_files()
        
        if summary["status"] == "no_files":
            logger.warning("⚠️ No files to process")
            return 1
        
        if analyzer.save_graph_data():
            logger.info("✅ Content Analysis completed successfully!")
            logger.info(f" Results: {summary['successful']}/{summary['total_files']} files processed")
            logger.info(f"Graph: {summary['unique_entities']} nodes, {summary['total_relationships']} relationships")
            logger.info(f"Model used: {analyzer.model_name} via {llm_provider.upper()}")
            return 0
        else:
            logger.error("❌ Failed to save graph data")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())