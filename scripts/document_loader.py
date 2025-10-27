from typing import List

from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_document(file_path: str) -> List[Document]:
    """
    Using UnstructeredLoader laod the document.
    This lib splits into logical elements.
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(docs)


def embedding(chunks: List[Document], q: str):
    pass


def main():
    path = "./raw_docs/151105741v1.pdf"

    docs = load_document(path)

    final_chunks = chunk_document(docs)
    print("Final chunks: ", len(final_chunks))
    print(f"Content: {final_chunks[0].page_content}")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

    vectorstore = Chroma.from_documents(
        documents=final_chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    retriver = vectorstore.as_retriever()

    retrived_documents = retriver.invoke(
        "Who did propose random forest algorithm?"
    )

    print(retrived_documents[0].page_content)


if __name__ == "__main__":
    main()

# file_path = "./raw_docs/151105741v1.pdf"

# loader = UnstructuredLoader(file_path=file_path,
#                             post_processing=[clean_extra_whitespace])
# docs = loader.load()

# # doc1 = docs[100]

# # # print(len(docs))
# # print(f"Page content: {doc1.page_content}")

# full_text = ""
# for doc in docs:
#     full_text += doc.page_content + "\n\n"
#     # print(str(doc.page_content) + "\n\n")


# full_doc_list = [Document(page_content=full_text,
#                           metadata={"source": file_path})]


# # Text splitter - instantiate the class we imported above
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# final_chunks = text_splitter.split_documents(full_doc_list)

# print(f"Final number of chunks: {len(final_chunks)}\n")

# print(f"Lenght of the first chunk: {len(final_chunks[0].page_content)}\n")

# print(f"Content: {final_chunks[0].page_content}")



