# RAG vs GraphRAG Comparison

## Query Files

### Vector RAG Query
```bash
python 4_query.py
```

### GraphRAG Query
```bash
python 4b_query_graph.py
```

## Application Comparison

### Flask Applications

#### Vector RAG Flask
```bash
python app_flask.py
```
**Port:** 8080

#### GraphRAG Flask
```bash
python app_flask_graph.py
```
**Port:** 8081

### Chainlit Applications

#### Vector RAG Chainlit
```bash
chainlit run app_chainlit.py
```
**Port:** 8000

#### GraphRAG Chainlit
```bash
chainlit run app_chainlit_graph.py --port 8001
```
**Port:** 8001

## Port Summary

| Technology | Vector RAG | GraphRAG |
|------------|------------|----------|
| **Query Script** | `4_query.py` | `4b_query_graph.py` |
| **Flask** | Port 8080 | Port 8081 |
| **Chainlit** | Port 8000 | Port 8001 |

## Quick Access URLs

| Application | URL |
|-------------|-----|
| Vector RAG Flask | http://localhost:8080 |
| GraphRAG Flask | http://localhost:8081 |
| Vector RAG Chainlit | http://localhost:8000 |
| GraphRAG Chainlit | http://localhost:8001 |