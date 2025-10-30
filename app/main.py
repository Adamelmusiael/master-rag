import os
from contextlib import asynccontextmanager
from typing import List, Optional
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from .retrive.load_documents import chunk_document, load_document

DATABASE_DIR = "../database"
vectorstore = None
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore

    os.makedirs(DATABASE_DIR, exist_ok=True)

    try:
        vectorstore = Chroma(
            persist_directory=DATABASE_DIR,
            embedding_function=embeddings
        )
        print("Chroma DB exists")
    except Exception as e:
        print(f"Failed initializing Chroma DB: {e}")
    yield


app = FastAPI(lifespan=lifespan)


async def check_database_connection():
    if vectorstore is None:
        raise HTTPException(status_code=500,
                            detail="Chroma database does not exist.")


@app.post("/hello/{user_name}/", tags=["users"])
def hello_user(user_name: str, depends=Depends(check_database_connection)):
    return f"Hello {user_name}!"


@app.post("/index/add_document/", tags=["Vector Database"])
async def add_document(file_path: str,
                       depends=Depends(check_database_connection)):
    """Validate path, load, chunk, attach document_id to each chunk and add to vectorstore.

    Security: only allow loading files located under the project's `raw_docs/` directory to
    avoid path traversal. Returns a structured JSON with the document_id and chunk ids.
    """
    project_root = Path(__file__).resolve().parents[1]
    allowed_dir = (project_root / "raw_docs").resolve()

    req_path = Path(file_path)
    if not req_path.is_absolute():
        candidate = (project_root / file_path).resolve()
    else:
        candidate = req_path.resolve()

    try:
        candidate.relative_to(allowed_dir)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid file path; allowed directory: raw_docs/")

    if not candidate.exists():
        raise HTTPException(status_code=404, detail="File not found")

    doc = load_document(str(candidate))
    chunked_doc = chunk_document(doc)

    # Attach a document_id to each chunk
    document_id = str(uuid4())
    for chunk in chunked_doc:
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["document_id"] = document_id
        chunk.metadata["source"] = str(candidate)

    ids = vectorstore.add_documents(chunked_doc)
    vectorstore.persist()

    return {"document_id": document_id, "chunk_count": len(ids), "chunk_ids": ids}


@app.delete("/index/delete_document/", tags=["Vector Database"])
async def delete_documents(ids: List[str],
                          depends=Depends(check_database_connection)):

    try:
        vectorstore.delete(ids=ids)
        vectorstore.persist()
        return {"message": f"Document {id} succesfully deleted."}
    except Exception as e:
        return {"message": f"Document {id} failed to delete.\nE: {e}"}


@app.delete("/index/delete_by_doc_id/", tags=["Vector Database"])
async def delete_by_doc_id(document_id: str,
                           deepends=Depends(check_database_connection)):
    try:
        vectorstore.delete(where={"document_id": document_id})
        vectorstore.persist()
        return {"deleted_document_id": document_id}
    except TypeError:
        docs = vectorstore.get(where={"document_id": document_id})
        ids = [d["id"] for d in docs["documents"]] if "documents" in docs else []
        if ids:
            vectorstore.delete(ids=ids)
            vectorstore.persist()
        return {"deleted_document_id": document_id, "deleted_chunk_count": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/index/get_documents_meta/", tags=["Vector Database"])
async def get_documents_meta(depends=Depends(check_database_connection)):
    try:
        docs = vectorstore.get()
        return {"documents": docs}
    except Exception as e:
        return {"message": f"Erro: {e}"}
