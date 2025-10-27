import os
from contextlib import asynccontextmanager
# from typing import Annotated

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


app = FastAPI()


async def check_database_connection():
    if vectorstore is None:
        raise HTTPException(status_code=500,
                            detail="Chroma database does not exist.")


@app.post("/hello/{user_name}/", tags=["users"])
def hello_user(user_name: str, depends=Depends(check_database_connection)):
    return f"Hello {user_name}!"


@app.post("/indx/add_document/", tags=["Vector Database"])
async def add_document(file_path: str,
                       depends=Depends(check_database_connection)):
    doc = load_document(file_path)
    chunked_doc = chunk_document(doc)

    ids = vectorstore.add_documents(chunked_doc)

    vectorstore.persist()
    return {"message": f"Document Succesfully loaded, id: {ids}"}


@app.post("index/delete_document/", tags=["users"])
async def delete_document(id: int,
                          depends=Depends(check_database_connection)):

    try:
        vectorstore.delete(ids=[id])
        return {"message": f"Document {id} succesfully deleted."}
    except Exception as e:
        return {"message": f"Document {id} failed to delete.\nE: {e}"}
