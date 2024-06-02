import os
import constants
from tqdm import tqdm
import openai

# OpenAI API 키 설정
openai.api_key = constants.APIKEY

# 텍스트 번역 함수
def translate_to_english(text: str) -> str:
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

# 톤 제거 함수
def remove_tone_with_gpt(translated_text: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that helps translate text to English without any tone or mannerisms."},
            {"role": "user", "content": f"Translate the following text to English without adding any tone or mannerisms. 존댓말이 해당 번역된 문구에 들어가지 않도록 해주어야해. 최대한 짧게 번역해줘야함: {translated_text}"}
        ],
        max_tokens=100,
        temperature=0.5,
    )
    return response.choices[0].message['content'].strip()

# 대화 전처리 함수
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

# 응답 생성 함수
def generate_response(P_partner: str, P_me: str, Q: str, A_eng: str, A: str) -> str:
    prompt = f"""
    상대는 {P_partner}인 사람이고,
    나는 {P_me}인 사람이야.
    그리고 상대가 나에게 {Q}라는 질문을 했을 때, 나는 평소 {A_eng}라는 말을 {A}라고 답하곤 했어.
    이때 나와 상대를 고려하고, 평소 말투를 고려했을 때 상대가 선생님 뭐하고 주무세요?라고 나는 뭐라고 대답할 지를 생성해줘. (다른 설 명은 덧붙이지 말고, 대답만 만들어줘)
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

# 예시 데이터
dialogue = [
    # {
    #     "P_partner": "애인",
    #     "P_me": "낭만적인 사람",
    #     "Q": "저녁 뭐 먹을까?",
    #     "A_eng": "Yes, I would love to.",
    #     "A": "너랑 함께라면 언제든."
    # },
    # {
    #     "P_partner": "24살 한국인 남자",
    #     "P_me": "채식주의자 22살 윤보미, 귀여운 이십대 여성 말투 사용",
    #     "Q": "오늘 저녁에 같이 고기먹을래?",
    #     "A_eng": "I’m resting in my home",
    #     "A": "ㅎㅎㅎ 지금 집에서 쉬고 있엉 ㅎㅎㅎ 너무 좋당 !! "
    # },
    {
        "P_partner": "18살 여학생", 
        "P_me": "상대를 가르치고 있는 29살 선생님",
        "Q": "선생님 자요?",
        "A_eng": "No",
        "A": "안자고 있어. 늦은 시간에 내게 연락하지마"
    },
    {
        "P_partner": "18살 여학생", 
        "P_me": "상대를 가르치고 있는 29살 선생님",
        "Q": "선생님 뭐하고 주무세요?",
        "A_eng": "You're a student and I'm a teacher. Don't talk like that",
        "A": "너는 학생이고 나는 선생님이야. 그런 식으로 말하지마"
    },
]

# 데이터 전처리 및 LLM 응답 생성
S_manner = [preprocess_dialogue(d) for d in dialogue]
for entry in tqdm(S_manner, desc="[Generating LLM Responses]"):
    P_partner = entry["P_partner"]
    P_me = dialogue[S_manner.index(entry)]["P_me"]
    Q = entry["Q"]
    A_eng = entry["A_eng"]
    A = entry["A"]
    entry["LLM_response"] = generate_response(P_partner, P_me, Q, A_eng, A)

# 결과 출력
for entry in S_manner:
    print(entry["LLM_response"])
