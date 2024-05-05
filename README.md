### Langchain-RAG
🔗 Link model : https://huggingface.co/vilm/vinallama-7b-chat-GGUF
# Note : 
-Faiss : a Python library that provides developers with a fast and efficient means of finding similar embeddings, as opposed to conventional hash-based approaches.
-Langchain: a framework written in Python and JavaScript, it provides tools to manipulate and build applications based on LLMs.
-Hugging Face: Huggingface is a company that provides tools and AI models to help developers build AI applications.
-RAG( Retrieval Augmented Generation ) :
# How to setup all necessary libs : 
In terminal window : pip install -r setup.txt

# How to create vector database :
- If you have content in a File or sth else , you have to extract content file (in this Repo I used PyPDF , PDF file ) to the text
- Text spliter
- Embedding ( computer can just understand the number not word ) - GPT4AllEmbedding : After text spliter
- With words,content was processed above , using Faiss ( Faceboook AI Similarity Search ) to creaate vector database then save

# How to apply available model into your project :
- Preparing db
- Loading LLM ( Large Language Model ) from Hugging Face
- Create prompt
- Using a large language model (LLM), prompt , a database (db) and RetrievalQA to create_qa_chain 
