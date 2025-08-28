#!/usr/bin/env python3
"""
MCP Server for File Ingestion
"""

import json
import logging
from pathlib import Path
import mimetypes
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
# PDF support
import PyPDF2
# Vector search support
from vector_utils import VectorIndex

class MCPServer:
    def __init__(self, storage_dir: str = "./mcp_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.files_db = self.storage_dir / "files_db.json"
        self.ingested_files = self.load_files_db()
        # Vector index for semantic search
        self.vector_index = VectorIndex()
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64, file_id: str = None, file_name: str = None, file_type: str = None) -> list:
        # Sliding window chunking, with metadata for each chunk
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+chunk_size]
            chunk_text = ' '.join(chunk)
            chunk_info = {
                'text': chunk_text,
                'file_id': file_id,
                'file_name': file_name,
                'file_type': file_type
            }
            chunks.append(chunk_info)
            i += chunk_size - overlap
        return chunks
        
    def load_files_db(self) -> Dict[str, Any]:
        if self.files_db.exists():
            try:
                with open(self.files_db, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading files db: {e}")
        return {}
    
    def save_files_db(self):
        try:
            with open(self.files_db, 'w', encoding='utf-8') as f:
                json.dump(self.ingested_files, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Error saving files db: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logging.error(f"Error hashing file {file_path}: {e}")
            return ""
    
    def read_text_file(self, file_path: Path) -> str:
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
                break
        return f"[Error: Could not decode file {file_path}]"
    
    def ingest_file(self, file_path: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}

            file_hash = self.get_file_hash(path)
            file_size = path.stat().st_size
            file_type = mimetypes.guess_type(str(path))[0] or 'unknown'

            if file_hash in self.ingested_files:
                return {"status": "already_exists", "file_id": file_hash, "message": f"File already ingested: {path.name}"}

            content = ""
            # PDF support
            if path.suffix.lower() == '.pdf':
                try:
                    with open(path, 'rb') as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        content = "\n".join(page.extract_text() or '' for page in reader.pages)
                except Exception as e:
                    content = f"[Error reading PDF: {e}]"
            elif file_type and file_type.startswith('text/'):
                content = self.read_text_file(path)
            elif path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.yml', '.yaml']:
                content = self.read_text_file(path)
            else:
                content = f"[Binary file: {path.name}, size: {file_size} bytes]"

            # Chunk and add to vector index if text-like
            if content and not content.startswith('[Binary file') and not content.startswith('[Error'):
                chunks = self.chunk_text(content, file_id=file_hash, file_name=path.name, file_type=file_type)
                self.vector_index.add_documents([c['text'] for c in chunks])
                # Store chunk metadata for search results
                if not hasattr(self, 'chunk_metadata'):
                    self.chunk_metadata = []
                self.chunk_metadata.extend(chunks)

            file_info = {
                "id": file_hash,
                "name": path.name,
                "path": str(path.absolute()),
                "size": file_size,
                "type": file_type,
                "content": content,
                "ingested_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }

            self.ingested_files[file_hash] = file_info
            self.save_files_db()

            return {"status": "success", "file_id": file_hash, "name": path.name, "size": file_size, "type": file_type}

        except Exception as e:
            return {"error": f"Error ingesting file: {str(e)}"}
    
    def get_file(self, file_id: str) -> Dict[str, Any]:
        return self.ingested_files.get(file_id, {"error": f"File not found: {file_id}"})
    
    def list_files(self) -> List[Dict[str, Any]]:
        return [{"id": f["id"], "name": f["name"], "size": f["size"], "type": f["type"], "ingested_at": f["ingested_at"]} for f in self.ingested_files.values()]
    
    def search_files(self, query: str) -> List[Dict[str, Any]]:
        # Semantic search using vector index
        semantic_results = self.vector_index.search(query, top_k=5)
        results = []
        # Map chunk text back to metadata
        if not hasattr(self, 'chunk_metadata'):
            return []
        for chunk_text, score in semantic_results:
            meta = next((c for c in self.chunk_metadata if c['text'] == chunk_text), None)
            if meta:
                results.append({
                    "chunk": chunk_text,
                    "score": score,
                    "file_id": meta['file_id'],
                    "file_name": meta['file_name'],
                    "file_type": meta['file_type']
                })
            else:
                results.append({"chunk": chunk_text, "score": score})
        return results
    
    def get_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        query_lower = query.lower()
        index = content.lower().find(query_lower)
        if index == -1:
            return content[:max_length] + "..." if len(content) > max_length else content
        start = max(0, index - max_length//2)
        end = min(len(content), index + len(query) + max_length//2)
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        return snippet
