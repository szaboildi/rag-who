# from pydantic import BaseModel
from openai import OpenAI


# class QuestionAnswering(BaseModel):
#     answer_to_question: str


def create_qa_string(question:str, answers:list[str])->str:
    user_prompt = f"<Answer the following question:\n<Question>{question}\n</Question>\n\n"
    user_prompt += "<Context>\n"

    for i, answer in enumerate(answers):
        user_prompt += f"<Document{i+1}>{answer}</Document{i+1}>\n\n"

    user_prompt += "</Context>"

    return user_prompt


def api_call(client:OpenAI, user_prompt:str,
             system_propmt_path:str, model="gpt-4o-mini",
             temperature:float=0):
    # client = OpenAI(
    #     api_key=os.environ.get("OPENAI_API_KEY"))

    with open(system_propmt_path, "r") as f:
        system_prompt = f.read()

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        # response_format=QuestionAnswering,
        model=model,
        temperature=temperature
    )

    return chat_completion.choices[0].message.content
