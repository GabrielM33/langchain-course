from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Simple template
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({"topic": "dogs"})
# print(prompt)

# Template with multiple variables
template = "Tell me a {adjective} story about {animal}"
prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke({"adjective": "funny", "animal": "whales"})
# print(prompt)

# Template with multiple messages
messages = [
    ("system", "You are a expert in the field of {topic}"),
    ("user", "Tell me a story about {topic}"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({"topic": "quantum computing"})
print(prompt)
