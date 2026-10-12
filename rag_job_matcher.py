"""Job-Resume Matching RAG prototype.
Setup: pip install langchain langchain-community langchain-huggingface langchain-chroma sentence-transformers chromadb
Run:   python rag_job_matcher.py
"""
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Knowledge base: job descriptions (replace/extend with 30-50 real ones for a stronger project)
JOBS = {
    "qa_automation": "QA Automation Engineer: Selenium, Playwright, Pytest, API testing, CI/CD pipelines, regression suites.",
    "ml_engineer": "Machine Learning Engineer: Python, PyTorch, model training, feature engineering, MLOps, deployment.",
    "genai_intern": "Gen AI Intern: LLMs, prompt engineering, LangChain, RAG pipelines, vector stores, Hugging Face, OpenAI.",
    "backend_dev": "Backend Developer: Node.js or Java, REST APIs, SQL databases, microservices, authentication.",
    "frontend_dev": "Frontend Developer: React, TypeScript, CSS, responsive UI, accessibility, state management.",
    "data_analyst": "Data Analyst: SQL, Excel, Tableau, dashboards, statistics, business reporting.",
    "devops_engineer": "DevOps Engineer: Docker, Kubernetes, Jenkins, GitHub Actions, Terraform, AWS, monitoring.",
    "hr_executive": "HR Executive: recruitment, onboarding, employee records, payroll coordination, HRMS tools.",
}

docs = [Document(page_content=text, metadata={"id": jid}) for jid, text in JOBS.items()]

# 2. Embeddings + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma.from_documents(docs, embeddings, collection_name="jobs")
retriever = store.as_retriever(search_kwargs={"k": 3})

# 3. Evaluation set: (candidate query, expected job id)
TESTS = [
    ("I write Selenium and Pytest scripts and run regression suites", "qa_automation"),
    ("I train neural networks in PyTorch and deploy models", "ml_engineer"),
    ("I build chatbots with LangChain and embeddings", "genai_intern"),
    ("I design REST APIs with Java and SQL", "backend_dev"),
    ("I build React UIs in TypeScript", "frontend_dev"),
    ("I make Tableau dashboards from SQL data", "data_analyst"),
    ("I manage Docker containers and Jenkins pipelines on AWS", "devops_engineer"),
    ("I handle hiring and employee onboarding", "hr_executive"),
]

def hit_at_k(k=3):
    hits = 0
    for query, expected in TESTS:
        results = store.similarity_search(query, k=k)
        if expected in [r.metadata["id"] for r in results]:
            hits += 1
    return hits, len(TESTS)

if __name__ == "__main__":
    query = "I automate API tests and use LLM tools like Claude Code and Copilot"
    print("Query:", query)
    for r in retriever.invoke(query):
        print(" ->", r.metadata["id"], "|", r.page_content[:70])
    h, n = hit_at_k(3)
    print(f"\nRetrieval hit@3: {h}/{n} ({100*h/n:.0f}%)")
