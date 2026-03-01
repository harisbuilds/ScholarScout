import os
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()

api_key = os.getenv("OPENAI_KEY")
if not api_key:
    raise ValueError("OPENAI_KEY not found in .env file. Please ensure it is set.")
client = OpenAI(api_key=api_key)

system_prompt = """
    You are a helpful assisstant who helps scholars find right scholarship and admission opportunity
    Use the following context to answer the user's question:\n\nContext:\n{context}. If you
    cannot find answer in the context please reply with 'I cannot find the information.'
"""

def get_llm_response(context: str, query: str) -> str:

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.format(context=context)
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    # Example usage
    with open("test_context.txt", "r") as f:
        sample_context = f.read()

    test_queries = [
        "What is ScholarScout, and what kind of programs does it help with?",
        "What is the eligibility criteria for the scholarship?",
        "What is the application deadline for the scholarship?",
    ]

    print("Sending request to LLM using OpenAI API...")
    for query in test_queries:
        start = time.time()
        reply = get_llm_response(context=sample_context, query=query)
        time_taken = time.time() - start
        print("--- Query ---")
        print(query)
        print("--- LLM Reply ---")
        print(reply)
        print("Time taken:", time_taken)
        print("\n")
