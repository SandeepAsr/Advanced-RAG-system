# Create requirements.txt file
requirements_content = """# Core Dependencies
openai>=1.0.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-openai>=0.0.5
pymilvus>=2.3.0
sentence-transformers>=2.2.0
transformers>=4.35.0
torch>=2.0.0
faiss-cpu>=1.7.4
rank-bm25>=0.2.2

# Document Processing
PyPDF2>=3.0.1
pypdf>=3.17.0
python-docx>=0.8.11
unstructured>=0.10.0
pdfplumber>=0.9.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Vector Databases
chromadb>=0.4.0
weaviate-client>=3.25.0
qdrant-client>=1.6.0

# Web Framework
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.28.0
gradio>=3.50.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
tqdm>=4.66.0
loguru>=0.7.0
redis>=5.0.0
tiktoken>=0.5.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Development
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
"""

with open(base_dir / "requirements.txt", "w") as f:
    f.write(requirements_content)
    
print("requirements.txt created successfully!")