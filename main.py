import fitz  # PyMuPDF
import openpyxl
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate


# Step 1: PDF Parser
def parse_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()


# Step 2: Excel Parser (.xlsx)
def parse_excel(path):
    wb = openpyxl.load_workbook(path)
    text = ""
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            text += " | ".join([str(cell) for cell in row if cell is not None]) + "\n"
    return text.strip()


# Step 3: Build FAISS Vector Index
def build_vector_index(text_chunks, embed_model):
    embeddings = embed_model.encode(text_chunks).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


# Step 4: Search FAISS
def search_index(query, index, docs, embed_model, k=1):
    q_embedding = embed_model.encode([query]).astype('float32')
    distances, indices = index.search(q_embedding, k)
    return [docs[i] for i in indices[0]]


# Step 5: Generate Answer with Ollama
def generate_answer_ollama(context, question, model="llama3.2"):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the context below to answer the question.

Context:
{context}

Question: {question}
Answer:"""
    )

    prompt = prompt_template.format(context=context, question=question)
    llm = Ollama(model=model)
    return llm.invoke(prompt)


# Step 6: Full RAG Pipeline
def rag_pipeline(file_paths, question):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = []

    for path in file_paths:
        if path.endswith(".pdf"):
            docs.append(parse_pdf(path))
        elif path.endswith(".xlsx"):
            docs.append(parse_excel(path))
        else:
            print(f"Unsupported file type: {path}")

    index, _ = build_vector_index(docs, embed_model)
    top_contexts = search_index(question, index, docs, embed_model, k=1)
    return generate_answer_ollama(top_contexts[0], question)


# 🧪 Example usage
if __name__ == "__main__":
    files = [
        "Test_doc.pdf"
    ]
    question = "what is the leave policy"
    answer = rag_pipeline(files, question)
    print("\n🧠 Answer:\n", answer)
