from mcp_server import MCPServer

mcp_server = MCPServer()
result = mcp_server.ingest_file("mcp_storage/indian laws in 2025.txt")
print(result)

# Optional: list all files to confirm
files = mcp_server.list_files()
print("Files in MCP:", files)
