from langchain_core.prompts import PromptTemplate

def describe_subject(subject: str, positive: bool) -> None:
  if positive:
    prompt=PromptTemplate.from_file(
      template_file="prompts/pros_subject.yaml",
    )
  else:
    prompt=PromptTemplate.from_file(
      template_file="prompts/cons_subject.yaml",
    )
  filled_prompt=prompt.invoke({"subject": subject})
  print("filled_prompt:", filled_prompt.to_string())

describe_subject("LangChain", positive=True)
describe_subject("LangChain", positive=False)