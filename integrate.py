import os
import openai
import constants
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain.runnables import RunnableSequence


# CO-STAR Framework
context = "일상, 현실"
objective = "이들의 대화를 통해 그들의 성격과 생활을 보여준다."
style = "대화는 캐주얼하고 친근한 톤으로 진행된다."
tone = "대화는 밝고 긍정적이다."
audience = "일상 대화에 익숙한 사람들."

# OpenAI API 키 설정
openai.api_key = constants.APIKEY

class ChatAssistant:
    def __init__(self):
        self.personas = {}
        self.persona_path = './data_kor/my_persona.txt'
        self.load_personas()
        self.model = OpenAI(openai_api_key=constants.APIKEY)  # API 키 직접 전달
        self.prompt_template = ChatPromptTemplate.from_template(
            """
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
        )
        self.chain = RunnableSequence(
            self.prompt_template,
            self.model
        )

    def load_personas(self):
        if os.path.exists(self.persona_path):
            with open(self.persona_path, 'r', encoding='utf-8') as file:
                persona_data = file.read().split("\n\n")
                for persona in persona_data:
                    if persona.strip():
                        id, details = persona.split("\n", 1)
                        self.personas[id.strip()] = details.strip()
        else:
            self.personas = {}

    def save_personas(self):
        with open(self.persona_path, 'w', encoding='utf-8') as file:
            for id, details in self.personas.items():
                file.write(f"{id}\n{details}\n\n")

    def translate_to_english(self, text: str) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates Korean text to English."},
                {"role": "user", "content": f"Translate the following Korean text to English: {text}"}
            ],
            max_tokens=100,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()

    def remove_tone_with_gpt(self, translated_text: str) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that helps translate text to English without any tone or mannerisms."},
                {"role": "user", "content": f"Translate the following text to English without adding any tone or mannerisms: {translated_text}"}
            ],
            max_tokens=100,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()

    def preprocess_dialogue(self, dialogue: dict) -> dict:
        P_partner = dialogue["P_partner"]
        Q = dialogue["Q"]
        A = dialogue["A"]
        A_eng_raw = self.translate_to_english(A)
        A_eng = self.remove_tone_with_gpt(A_eng_raw)
        
        return {
            "P_partner": P_partner,
            "Q": Q,
            "A": A,
            "A_eng": A_eng
        }

    def generate_response(self, P_partner: str, P_me: str, Q: str, A_eng: str, A: str) -> str:
        prompt = f"""
        상대는 {P_partner}인 사람이고,
        나는 {P_me}인 사람이야.
        그리고 상대가 나에게 {Q}라는 질문을 했을 때, 나는 평소 {A_eng}라는 말을 {A}라고 답하곤 했어.
        이때 나와 상대를 고려하고, 평소 말투를 고려했을 때 나는 뭐라고 대답할 지를 생성해줘. (다른 설명은 덧붙이지 말고, 대답만 만들어줘)
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates natural responses based on given personas and context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()

    def summarize_dialogues(self, dialogues: list):
        persona = ""
        for dialogue in tqdm(dialogues, desc="Summarizing Dialogues"):
            summary = self.chain.invoke({
                "dialogue": dialogue,
                "context": "일상, 현실",
                "objective": "이들의 대화를 통해 그들의 성격과 생활을 보여준다.",
                "style": "대화는 캐주얼하고 친근한 톤으로 진행된다.",
                "tone": "대화는 밝고 긍정적이다.",
                "audience": "일상 대화에 익숙한 사람들."
            })["text"]
            persona += f"\n\n{summary}"
        return persona

    def train(self, id: str, q: str, a: str) -> None:
        if id in self.personas:
            self.personas[id] += f"\nQ: {q}\nA: {a}"
        else:
            self.personas[id] = f"Q: {q}\nA: {a}"
        self.save_personas()

    def run(self, my_id: str, target_id: str, q: str) -> str:
        if my_id not in self.personas or target_id not in self.personas:
            return "Persona not found for given IDs."

        P_me = self.personas[my_id]
        P_partner = self.personas[target_id]
        A_eng = self.translate_to_english("안자고 있어. 늦은 시간에 내게 연락하지마")  # 예시 응답 번역
        A = "안자고 있어. 늦은 시간에 내게 연락하지마"  # 예시 응답

        return self.generate_response(P_partner, P_me, q, A_eng, A)


# 예시 사용
chat_assistant = ChatAssistant()

# 학습 데이터 추가
chat_assistant.train("I am 27 years old.", "선생님 자요?", "안자고 있어. 늦은 시간에 내게 연락하지마")
chat_assistant.train("My gender is male.", "선생님 뭐하고 주무세요?", "너는 학생이고 나는 선생님이야. 그런 식으로 말하지마")

chat_assistant.train("채식주의자 22살 윤보미는 귀여운 이십대 여성의 말투를 사용한다.", "너 지금 뭐해? 오늘 저녁에 같이 고기먹을래?", "ㅎㅎㅎ 지금 집에서 쉬고 있엉 ㅎㅎㅎ 너무 좋당 !! ")

# 대화 요약 및 퍼소나 반영
# Load dialogue from the data_kor directory
dialogues = []
for filename in os.listdir('./data_kor'):
    if filename.endswith('.txt') and filename != 'my_persona.txt':
        with open(os.path.join('./data_kor', filename), 'r', encoding='utf-8') as file:
            dialogues.append(file.read())

summarized_persona = chat_assistant.summarize_dialogues(dialogues)

# Save updated persona to a new file
augmented_persona_path = './data_kor/my_persona_augmented.txt'
with open(augmented_persona_path, 'w', encoding='utf-8') as file:
    file.write(summarized_persona)

print("Summaries added to my_persona_augmented.txt")

# 응답 생성
response = chat_assistant.run("I am 27 years old.", "My gender is male.", "선생님 자요?")
print(response)
response = chat_assistant.run("I am 27 years old.", "채식주의자 22살 보미", "이따 밤에 뭐해? 소고기 먹을래?")
print(response)
