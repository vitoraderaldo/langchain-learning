from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

# This code is using the completion style of the LLM.

# Step 1: Load env variables
load_dotenv()

# Step 2: Load arguments
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="sum 2 numbers", type=str, help="Task to be performed")
parser.add_argument("--language", default="typescript", type=str, help="Language to be used")
args = parser.parse_args()


# Step 3: Setup LLM
llm = OpenAI()

# Step 4: Setup Code Generator Chain
code_prompt = PromptTemplate(
  template="Write a very short {language} function that will {task}",
  input_variables=["language", "task"]
)

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt,
  output_key="code",
)

# Step 5: Setup Testing Generator chain
test_prompt = PromptTemplate(
  template="Write a unit test for the following {language} code: {code}",
  input_variables=["language", "code"],
)

test_chain = LLMChain(
  llm=llm,
  prompt=test_prompt,
  output_key="test",
)

# Step 6: Wrap up chains in a SequentialChain
chain = SequentialChain(
  chains=[code_chain, test_chain],
  input_variables=["task", "language"],
  output_variables=["code", "test"],
)

result = chain({
  "language": args.language,
  "task": args.task
})


print("\n>>>>>>> GENERATED CODE <<<<<<<")
print(result["code"])

print("\n>>>>>>> GENERATED TEST <<<<<<<")
print(result["test"])
