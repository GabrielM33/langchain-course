from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

openai_model = ChatOpenAI(model="gpt-4o-mini")
google_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# ChatPromptTemplate with a template string
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "quantum computing"})
result = google_model.invoke(prompt)
# print(result.content)

# PromptTemplate with mutlple placeholders
template = "Tell me a {adjective} story about {person}"
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"adjective": "funny", "person": "Messi"})
result = openai_model.invoke(prompt)
# print(result.content)
