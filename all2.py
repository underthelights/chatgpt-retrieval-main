import os
import constants
import openai

# OpenAI API 키 설정
openai.api_key = constants.APIKEY
id2Persona = {}

# 기존 페르소나 로드 함수
def load_persona(persona_path: str) -> str:
    if os.path.exists(persona_path):
        with open(persona_path, 'r', encoding='utf-8') as file:
            return file.read()
    return ""
# def load_persona(id: str) -> str:
#     return id2Persona.get(id, "")

# def save_persona(id: str, persona: str):
#     id2Persona[id] = persona
#     return id2Persona[id]
# 페르소나 저장 함수
def save_persona(persona: str, persona_path: str):
    with open(persona_path, 'w', encoding='utf-8') as file:
        file.write(persona)

# 요약 생성 함수
def generate_summary(context: str, objective: str, style: str, tone: str, audience: str, dialogue: str) -> str:
    prompt = f"""
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
    해당 dialogue를 요약해줘. 이건 나 자신의 Persona 에 반영할거야.
    dialogue로부터 대화가 어떤 톤으로 진행되었으며, 질문자가 무엇을 물었고, 응답자가 어떻게 대답했는지, 그리고 전체적인 대화의 흐름를 요약해줘.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes dialogues."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.5,
    )
    return response.choices[0].message['content'].strip()

# 학습 함수
def train(user_id: str, target_id: str, q: str, a: str):
    persona_path = f"{user_id}"
    # 기존 페르소나 로드
    persona = load_persona(persona_path)

    # CO-STAR 프레임워크 설정
    context = "일상, 현실"
    objective = "이들의 대화를 통해 그들의 성격과 생활을 보여준다."
    style = "대화는 캐주얼하고 친근한 톤으로 진행된다."
    tone = "대화는 밝고 긍정적이다."
    audience = "일상 대화에 익숙한 사람들."

    # 대화 텍스트
    dialogue = f"\"{target_id}\"가 {user_id}에게 {q}라는 질문을 질문을 했고, 나는 {a}라고 대답했다."

    # 요약 생성
    summary = generate_summary(context, objective, style, tone, audience, dialogue)

    # 페르소나 업데이트
    persona += f"\n\n{summary}"

    # 업데이트된 페르소나 저장
    save_persona(persona, persona_path)
    
    # 업데이트된 페르소나 출력
    return 0


# 텍스트 번역 함수: 한국어 텍스트를 영어로 번역
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

# 톤 제거 함수: 번역된 텍스트에서 톤이나 말투를 제거
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

# 응답 생성 함수
def generate_response(P_partner: str, P_me: str, Q: str, A_eng: str, A: str) -> str:
    prompt = f"""
        나는 {P_me}인 사람이야.
        상대는 {P_partner}인 사람이고,
        그리고 상대가 나에게 {Q}라는 질문을 했을 때, 나는 평소 {A_eng}라는 말을 {A}라고 답하곤 했어.
        이때 나와 상대를 고려하고, 평소 말투를 고려했을 때 나는 뭐라고 대답할 지를 생성해줘. 
        다른 설명은 덧붙이지 말고, 한글로 대답을 만들어줘.
        """
    print("prompt:\n"+prompt)
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

# 실행 함수
# my_id : 123
def run(my_id: str, target_id: str, q: str):
    # 기존 페르소나 로드
    my_persona_path = f"{my_id}"
    my_persona = load_persona(my_persona_path)
    target_persona_path = f"{target_id}"
    target_persona = load_persona(target_persona_path)
    # 페르소나 분리 (내 페르소나와 타겟 페르소나)
    P_me = my_persona
    P_partner = target_persona
    
    # 번역 및 톤 제거
    a_eng = translate_to_english(q)
    a_eng_tone_removed = remove_tone_with_gpt(a_eng)

    # 사용자 응답 생성
    response = generate_response(P_partner, P_me, q, a_eng_tone_removed, q)

    return response

# 예시 사용
# train("user_id", "오늘 날씨 어때? 어디 놀러 갈래???", "그러게 날씨 진짜 좋네. 우리 한강 가자 이따 수업끝나구") #, "my_persona.txt")

# 예시 사용
# answer = run("my_id", "target_id", "오늘 저녁에 뭐 먹을래?") #, "my_persona.txt")
# print(f"Generated Answer: {answer}")
if __name__ == "__main__":
    train("1", "2", "내일 저녁에 같이 밥 먹자", "좋아. 그러자!")
    # print(load_persona(f"1"))
    response = run("1", "3", "내일 저녁에 시간되냐?")
    # 내일 저녁에 시간이 괜찮아!
    print("response: ", response)