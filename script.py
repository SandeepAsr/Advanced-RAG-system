# Let's start by creating the complete RAG system structure
import os
from pathlib import Path

# Create the directory structure for the RAG system
base_dir = Path("advanced_rag_system")
directories = [
    "src",
    "src/ingestion",
    "src/retrieval", 
    "src/generation",
    "src/utils",
    "src/database",
    "config",
    "data",
    "data/pdfs",
    "data/chunks",
    "tests",
    "docs"
]

for directory in directories:
    (base_dir / directory).mkdir(parents=True, exist_ok=True)
    
print("Directory structure created successfully!")
print("Created directories:")
for directory in directories:
    print(f"  - {directory}")