import os
import constants
from tqdm import tqdm
import openai

openai.api_key = constants.APIKEY

def translate_to_english(text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates Korean text to English."},
            {"role": "user", "content": f"Translate the following Korean text to English, and make the shortest translated sentence as short as possible. do not consider any tone, manner. 존댓말이 해당 번역된 문구에 들어가지 않도록 해주어야해. 최대한 짧게 번역해줘야함: {text}"}
        ],
        max_tokens=100,
        temperature=0.5,
    )
    return response.choices[0].message['content'].strip()

def remove_tone_with_gpt(translated_text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that helps translate text to English without any tone or mannerisms."},
            {"role": "user", "content": f"Do not consider any tone or mannerisms. : {translated_text}"}
        ],
        max_tokens=100,
        temperature=0.5,
    )
    return response.choices[0].message['content'].strip()

def preprocess_dialogue(dialogue: dict) -> dict:
    P_partner = dialogue["P_partner"]
    Q = dialogue["Q"]
    A = dialogue["A"]
    A_eng_raw = translate_to_english(A)
    A_eng = remove_tone_with_gpt(A_eng_raw)
    
    return {
        "P_partner": P_partner,
        "Q": Q,
        "A": A,
        "A_eng": A_eng
    }

def generate_llm_response(P_partner: str, Q: str, A_eng: str, A: str) -> str:
    return f"상대 페르소나 {P_partner}를 지닌 상대방의 질문 '{Q}'에 대한 나의 대답인 영어로 번역된 응답 '{A_eng}'은 실제 응답 '{A}' 와 같다"

# 예시 데이터
dialogue = [
    {
        "P_partner": "문크예거",
        "Q": "너 오늘 뭐합니까?",
        "A": "난 오늘 집에서 쉰다"
    },
    {
        "P_partner": "엄마",
        "Q": "너 오늘 뭐하니?",
        "A": "저는 오늘 집에서 쉬려고 합니다."
    }
]

# 데이터 전처리 및 LLM 응답 생성
S_manner = [preprocess_dialogue(d) for d in dialogue]
for entry in tqdm(S_manner, desc="[Generating LLM Responses]"):
    entry["LLM_response"] = generate_llm_response(entry["P_partner"], entry["Q"], entry["A_eng"], entry["A"])

# 결과 출력
for entry in S_manner:
    print(entry)
