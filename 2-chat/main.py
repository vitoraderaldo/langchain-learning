from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv

# This code is using the conversation style of the LLM 
# that requires sending the entire conversation history to the model.

load_dotenv()


chat = ChatOpenAI(
  #verbose=True
)

memory = ConversationBufferMemory(
  memory_key="messages",
  return_messages=True, # Will return the messages specifying which entity produced: Human or AI
  chat_memory=FileChatMessageHistory("messages.json"), # Optional parameter to save the chat history
)

# summarized_memory = ConversationSummaryMemory.from_messages(
#     llm=chat, # This memory requires a LLM to ask it to summarize the conversation
#     chat_memory=FileChatMessageHistory("summarized-messages.json"),
#     return_messages=True,  # Will return the messages specifying which entity produced: Human or AI
#     memory_key="messages",  # Optional parameter to save the chat history
# )


prompt = ChatPromptTemplate(
  input_variables=["content", "messages"],
  messages=[
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{content}"),
  ],
)

chain = LLMChain(
  llm = chat,
  prompt=prompt,
  memory=memory,
  #verbose=True,
)

while True:
  content = input(">> ")

  result = chain({
    "content": content
  })

  print(result["text"])
