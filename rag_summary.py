import os
import constants
from tqdm import tqdm

# for rag
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# for template runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

class FormatTexts(Runnable):
    def __init__(self, text):
        self.text = text

    def run(self):
        return "\n".join(self.text.splitlines())

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Load documents from the data_kor directory
loader = DirectoryLoader('./data_kor', glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# CO-STAR Framework
context = "일상, 현실"
objective = "이들의 대화를 통해 그들의 성격과 생활을 보여준다."
style = "대화는 캐주얼하고 친근한 톤으로 진행된다."
tone = "대화는 밝고 긍정적이다."
audience = "일상 대화에 익숙한 사람들."

# Load dialogue from the data_kor directory
dialogues = []
for filename in os.listdir('./data_kor'):
    if filename.endswith('.txt') and filename != 'my_persona.txt':
        with open(os.path.join('./data_kor', filename), 'r', encoding='utf-8') as file:
            dialogues.append(file.read())

# Load existing persona
persona_path = './data_kor/my_persona.txt'
with open(persona_path, 'r', encoding='utf-8') as file:
    persona = file.read()

# Generate summaries
prompt_template = """
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
prompt = ChatPromptTemplate.from_template(prompt_template)
model = ChatOpenAI()

chain = (
    {
        "dialogue": lambda x: x["dialogue"],
        "objective": lambda x: objective,
        "style": lambda x: style,
        "tone": lambda x: tone,
        "audience": lambda x: audience,
        "context": lambda x: context,
    }
    | prompt
    | model
    | StrOutputParser()
)

# Summarize each dialogue and append to persona
for dialogue in tqdm(dialogues, desc="Summarizing Dialogues"):
    summary = chain.invoke({"dialogue": dialogue})
    persona += f"\n\n{summary}"

# Save updated persona to a new file
augmented_persona_path = './data_kor/my_persona_augmented.txt'
with open(augmented_persona_path, 'w', encoding='utf-8') as file:
    file.write(persona)

print("Summaries added to my_persona_augmented.txt")
