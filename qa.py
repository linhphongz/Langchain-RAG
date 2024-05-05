from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

model_file = r"./vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

#Create prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt

# Tao simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

# Read  VectorDB
def read_vectors_db():
    # Converted embedding vectordatabase
    embedding_model = GPT4AllEmbeddings(model_file=r"./all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model)
    return db


#Testing
db = read_vectors_db()
llm = load_llm(model_file)

#Make Prompt
template = """<|im_start|>system\nTrả lời câu hỏi theo thông tin được cho sẵn\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)

llm_chain  =create_qa_chain(prompt, llm, db)

#Run
question = "Your question"
response = llm_chain.invoke({"query": question})
print(response)
