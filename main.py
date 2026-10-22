
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import hashlib
from langchain_chroma import Chroma



def load_documents(folder_path: str = "data"):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder.resolve()}"
        )

    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {folder.resolve()}"
        )

    documents = []

    for pdf_file in pdf_files:
        try:
            print(f"Loading: {pdf_file.name}")

            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            # Keep track of each document's source
            for page in pages:
                page.metadata["source"] = pdf_file.name

            documents.extend(pages)

            print(f"  Pages loaded: {len(pages)}")

        except Exception as error:
            print(f"  Failed to load {pdf_file.name}: {error}")

    if not documents:
        raise RuntimeError(
            "No PDF pages could be loaded successfully."
        )

    return documents



def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    print(f"\nTotal chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Page: {chunk.metadata.get('page')}")
        print(f"Characters: {len(chunk.page_content)}")
        print(f"Start index: {chunk.metadata.get('start_index')}")
        print(chunk.page_content[:200])

    return chunks


def create_vector_store(chunks, embedding_model):
    if not chunks:
        raise ValueError("No chunks available to store.")

    persist_directory = "./chroma_db"

    # Create a stable ID for each chunk to avoid duplicate records
    ids = []
    for chunk in chunks:
        unique_text = (
            f"{chunk.metadata.get('source', '')}|"
            f"{chunk.metadata.get('page', '')}|"
            f"{chunk.metadata.get('start_index', '')}|"
            f"{chunk.page_content}"
        )
        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()
        ids.append(chunk_id)

    # Open or create the persistent Chroma database
    vector_store = Chroma(
        collection_name="rag_docs",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )

    # Store chunks, metadata, and their embeddings
    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    print("\nVector database created successfully!")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Database location: {persist_directory}")
    print(f"Collection: rag_docs")
    print(f"Total records in database: {vector_store._collection.count()}")

    return vector_store





def generate_embeddings(chunks):
    if not chunks:
        raise ValueError("No chunks available for embedding.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [chunk.page_content for chunk in chunks]

    vectors = embedding_model.embed_documents(texts)

    print("\nEmbedding generation completed!")
    print(f"Total chunks embedded: {len(vectors)}")
    print(f"Dimensions per vector: {len(vectors[0])}")
    print(f"First vector preview: {vectors[0][:10]}")

    return embedding_model, vectors






def search_documents(vector_store, question, k=3):
    results = vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    print(f"\nQuestion: {question}")
    print(f"Retrieved chunks: {len(results)}")

    for i, (document, score) in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {document.metadata.get('source', 'Unknown')}")
        print(f"Page: {document.metadata.get('page', 'Unknown')}")
        print(f"Distance score: {score:.4f}")
        print(f"Content:\n{document.page_content}")

    return results



def answer_question(vector_store, llm, question, k=4):
    # 1. Retrieve relevant chunks
    results = vector_store.similarity_search(
        question,
        k=k,
    )

    if not results:
        print("No relevant documents were found.")
        return

    # 2. Build context with source information
    context_parts = []

    for document in results:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")

        context_parts.append(
            f"Source: {source}, Page: {page}\n"
            f"{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # 3. Create a prompt that grounds the answer in the PDFs
    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using only the context below.

Rules:
- Do not invent facts that are not in the context.
- If the context does not contain the answer, say:
  "I couldn't find enough information in the uploaded documents."
- Give a clear, well-organized answer.
- Mention the source PDF when relevant.

Context:
{context}

Question:
{question}

Answer:
"""

    # 4. Generate the answer with Llama 3
    response = llm.invoke(prompt)

    print("\n" + "=" * 50)
    print("GENERATED ANSWER")
    print("=" * 50)
    print(response.content)

    print("\nSOURCES")
    for document in results:
        print(
            f"- {document.metadata.get('source', 'Unknown')}, "
            f"page {document.metadata.get('page', 'Unknown')}"
        )

    return response.content


if __name__ == "__main__":
    docs = load_documents()
    print(f"\nTotal pages loaded: {len(docs)}")

    chunks = split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = create_vector_store(
        chunks,
        embedding_model,
    )

    # Initialize the local LLM
    llm = ChatOllama(
        model="llama3",
        temperature=0,
    )

    # Ask questions interactively
    while True:
        question = input(
            "\nEnter your question (or type 'exit' to quit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        answer_question(
            vector_store,
            llm,
            question,
            k=4,
        )


