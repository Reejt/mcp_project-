from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mcp_server import MCPServer
from mcp_ollama import MCPWrapperOllama

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

mcp_server = MCPServer()
mcp = MCPWrapperOllama(model_name="llama3.2:3b-8k", chunk_size=300)
chat_history = []

# Keep track of ingested file IDs to detect changes
ingested_files = set()

# ---- Build FAISS index at startup ----
@app.on_event("startup")
async def build_index():
    global ingested_files
    files = mcp_server.list_files()
    documents = [mcp_server.get_file(f["id"])["content"] for f in files]
    ingested_files = {f["id"] for f in files}
    if documents:
        print("[INFO] Building FAISS index at startup...")
        mcp.ingest_documents(documents, refresh=True)

@app.get("/", response_class=HTMLResponse)
async def chat_home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "chat_history": chat_history, "upload_status": None})

# File upload route
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    upload_status = None
    if file:
        file_location = f"mcp_storage/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        result = mcp_server.ingest_file(file_location)
        upload_status = f"File '{file.filename}' uploaded and ingested. Status: {result.get('status', result.get('error'))}"
    return templates.TemplateResponse("chat.html", {"request": request, "chat_history": chat_history, "upload_status": upload_status})

@app.post("/send", response_class=HTMLResponse)
async def send_message(request: Request, user_input: str = Form(...)):
    global ingested_files

    # ---- Check if new documents were added ----
    files = mcp_server.list_files()
    current_file_ids = {f["id"] for f in files}
    new_files = current_file_ids - ingested_files

    if new_files:
        print("[INFO] New documents detected, updating FAISS index...")
        new_docs = [mcp_server.get_file(f["id"])["content"] for f in files if f["id"] in new_files]
        mcp.ingest_documents(new_docs, refresh=False)
        ingested_files = current_file_ids

    # ---- Hybrid Query ----
    response = mcp.query(user_input, strong_threshold=0.7, weak_threshold=0.5)

    # ---- Update chat history ----
    chat_history.append({"user": user_input, "bot": response})

    return templates.TemplateResponse("chat.html", {"request": request, "chat_history": chat_history})
