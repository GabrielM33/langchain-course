from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

openai_model = ChatOpenAI(model="gpt-4o-mini")
google_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

chain = (
    prompt_template | google_model | StrOutputParser() | uppercase_output | count_words
)

result = chain.invoke({"topic": "lawyers", "joke_count": 2})

print(result)
