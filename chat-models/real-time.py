from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

chat_history = []

system_message = SystemMessage(content="You are a helpful assistant.")
chat_history.append(system_message)  # Add system message to chat history

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":  # Exit chat
        break
    chat_history.append(HumanMessage(content=query))  # Add user message to chat history

    # Get AI response using the chat history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message to chat history
    print(f"AI: {response}")  # Print AI response

    # Answer from gpt-4o-mini:
    # You: How many times did Juventus win the champions league?
    #
    # AI: As of October 2023, Juventus has won the UEFA Champions League (formerly known as the European Cup) twice.
    # Their victories came in the following years:
    #
    # 1. 1984-85
    # 2. 1995-96
    #
    # If you need more information about their performances in the tournament or any other details, feel free to ask!
    #
    # You: who is Juventus GOAT?
    #
    # AI: Determining the "Greatest of All Time" (GOAT) for Juventus can be subjective, as it often depends on personal opinions and criteria.
    # However, a few players are frequently mentioned in discussions about Juventus' greatest legends:
    #
    # 1. **Alessandro Del Piero**:
    # Often considered the face of Juventus, Del Piero spent 19 years at the club and is the all-time leading scorer.
    # His skill, loyalty, and leadership have made him a beloved figure among fans.
    #
    # 2. **Michel Platini**:
    # The French midfielder played for Juventus in the 1980s and was known for his incredible playmaking ability.
    # He won three Ballon d'Or awards while at the club and helped lead them to numerous titles.
    #
    # 3. **Pavel Nedvěd**:
    # The Czech midfielder was known for his work ethic, skill, and influential performances.
    # He won the Ballon d'Or in 2003 and was a key player during his time with the club.
    #
    # 4. **Roberto Baggio**:
    # Baggio is another legendary figure in Italian football who had a significant impact during his time at Juventus in the early 1990s.
    # His talent and charisma made him a fan favorite.
    #
    # 5. **Gaetano Scirea**:
    # A legendary defender and a key part of the Juventus team during the late 1970s and 1980s.
    # Scirea is remembered for his elegance on the field and sportsmanship.
    #
    # Different fans may have their preferences based on different eras and achievements, but these players are often at the top of the list when discussing Juventus' greatest players.
    #
    # You:  exit
    #
    # AI: It seems like you're looking to end the conversation.
    # If you have any more questions in the future or need assistance, feel free to return.
    # Have a great day!
