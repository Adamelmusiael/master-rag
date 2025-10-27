from typing import List

from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_document(file_path: str) -> List[Document]:
    """
    Using UnstructeredLoader laod the document. This lib splits into logical elements.
    This elements are not ready for embedding they need to be connected.
    Thus we connect them (separating each element by \n\n) into single object
    of Document.
    """
    loader = UnstructuredLoader(file_path=file_path,
                                post_processing=[clean_extra_whitespace])
    docs = loader.load()

    full_text = "\n\n".join([doc.page_content for doc in docs])
    full_doc = [Document(page_content=full_text,
                         metadata={"source": file_path})]
    return full_doc


def chunk_document(docs: List[Document]) -> List[Document]:
    """properly chunk documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(docs)
