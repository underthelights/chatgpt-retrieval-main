import os
import sys
import constants

# for rag
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# for template runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import Runnable

def format_docs(docs):
  return "\n\n".join([d.page_content for d in docs])

class FormatTexts(Runnable):
  def __init__(self, text):
    self.text = text
  
  def run(self):
    return "\n".join(self.text.splitlines())

os.environ["OPENAI_API_KEY"] = constants.APIKEY

loader = DirectoryLoader('./', glob="data/data1.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# [KYU] CO-STAR Framework 
context = "일상, 현실"
objective = "이들의 대화를 통해 그들의 성격과 생활을 보여준다."
style = "대화는 캐주얼하고 친근한 톤으로 진행된다."
tone = "대화는 밝고 긍정적이다."
audience = "일상 대화에 익숙한 사람들."
dialogue = 0

# Get input
# query = None
# if len(sys.argv) > 1:
#   query = sys.argv[1]

# print(f"Input is : [{query}]")

# Get My Persona P
with open('./data_kor/data1.txt', 'r') as file:
  # 파일 내용을 읽어 변수에 저장
  dialogue = file.read()
  # Persona = format_texts([Persona])
# print(f"My Persona is : {Persona}")

print(f"Dialogue is : {dialogue}")

# Generate Answer
template = """

### CONTEXT ### 
{context}
### OBJECTIVE ### 
{objective}
### STYLE ### 
{style}
### TONE ### 
{tone}
### AUDIENCE ### 
{audience}
### DIALOGUE ### 
{dialogue}
### Summary ###
해당 dialogue를 요약해줘. 이건 나 자신의 Persona 에 반영할거야
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

chain = (
    {
      "dialogue": lambda x: dialogue, 
      "objective": lambda x: objective, 
      "style": lambda x: style, 
      "tone": lambda x: tone, 
      "audience": lambda x: audience, 
      "context": retriever | format_docs, 
      "question": RunnablePassthrough()
      }
    | prompt
    | model
    | StrOutputParser()
)

print(chain.invoke("data1_summary.txt"))
