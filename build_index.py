from mcp_server import MCPServer
from mcp_ollama import MCPWrapperOllama

# Initialize
mcp_server = MCPServer()
mcp = MCPWrapperOllama(chunk_size=300)

# Load files from MCP
files = mcp_server.list_files()
documents = [mcp_server.get_file(f["id"])["content"] for f in files]

# Build FAISS index
mcp.ingest_documents(documents, refresh=True)
print("FAISS index built with", len(documents), "documents.")

# Optional: Test a query and print chunk scores
query = "Your test question here"
relevant_chunks, scores = mcp.retrieve_with_scores(query, top_k=5)

for chunk, score in zip(relevant_chunks, scores):
    print(f"Score: {score:.4f} | Chunk preview: {chunk[:100]}...")

response = mcp.query(
    "What is the punishment for black money in 2025?",
    strong_threshold=2.0,
    weak_threshold=3.0
)
print(response)

