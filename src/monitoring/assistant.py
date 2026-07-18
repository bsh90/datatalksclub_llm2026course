from sys import argv
import os

from dotenv import load_dotenv
from openai import OpenAI

from utils.ingest import load_faq_data, build_index
from metrics import RAGWithMetrics

from db_save import save_conversation

def create_assistant():
    load_dotenv()

    documents = load_faq_data()
    index = build_index(documents)

    return RAGWithMetrics(
        index=index,
        llm_client=OpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1"),
        model='poolside/laguna-m.1:free',
    ) 

if __name__ == "__main__":
    assistant = create_assistant()

    query = "How do I join the course?"
    if len(argv) > 1:
        query = argv[1]

    answer = assistant.rag(query)
    save_conversation(assistant.last_call, query, "llm-zoomcamp")
    print(answer)
