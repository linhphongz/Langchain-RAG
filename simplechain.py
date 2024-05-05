from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
#Load LLM 
model_file = r"./vinallama-7b-chat_q5_0.gguf"
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template,input_variables=["question"])
    return prompt

def create_simple_chain(llm,prompt):
    llm_chain = LLMChain(prompt = prompt , llm=llm)
    return llm

template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""
llm = load_llm(model_file=model_file)
prompt = create_prompt(template=template)
simple_chain = create_simple_chain(llm,prompt)
question = "Hôm nay là ngày bao nhiêu ?"
respone = simple_chain.invoke({'question':question})
print(respone)