from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

# Adding kwargs for model configuration
openai_model = ChatOpenAI(
    model="gpt-4o-mini",  # Updated model name
    temperature=0.7,  # Controls randomness (0-1)
    max_tokens=150,  # Maximum length of response
    streaming=False,  # Whether to stream the response
    verbose=False,  # Print debug information
)

google_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    max_tokens=150,
    temperature=0.7,
    verbose=True,
)

result = google_model.invoke("How many stars are there in the sky?")

print("Full result:")
print(result)

print("\n\nContent only:")
print(result.content)
