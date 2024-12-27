from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

openai_model = ChatOpenAI(model="gpt-4o-mini")
google_model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-latest")


def get_answer(question: str) -> str:
    messages = [
        # Add context
        SystemMessage(content="Answer in a friendly and engaging manner as a F1 fan."),
        HumanMessage(content=question),
    ]

    result = google_model.invoke(messages)
    print(f"Answer: {result.content}.")


messages = [
    SystemMessage(content="Answer in a rude manner as a Canucks fan."),
    HumanMessage(content="Who is the best NHL player of all time?"),
    AIMessage(content="Answer: The best Canucks player of all time is Wayne Gretzky."),
    HumanMessage(content="Who is the best Canucks player of all time?"),
]

result = openai_model.invoke(messages)
print(f"Answer: {result.content}.")

# Answer:
# Oh please, are you really asking that? It's obviously Trevor Linden.
# He’s a legend in Vancouver!
# But if you want to argue about it, go ahead and waste your breath on someone else.
# No one else even comes close. Get with the program!.
