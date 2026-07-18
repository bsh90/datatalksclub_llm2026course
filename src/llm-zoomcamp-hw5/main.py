from starter import rag, index
import os
from openai import OpenAI
from dotenv import load_dotenv
from rag_helper import RAGBase

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

import sqlite3
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True

provider = TracerProvider()
#provider.add_span_processor(
#    SimpleSpanProcessor(ConsoleSpanExporter())
#)
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

load_dotenv()
openrouter_client = OpenAI(
                            api_key=os.getenv("OPENROUTER_API_KEY"),
                            base_url="https://openrouter.ai/api/v1"
                            )
class RAGTraced(RAGBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, index=index, llm_client=openrouter_client, model= "poolside/laguna-m.1:free")

    def search(self, query):
        with tracer.start_as_current_span("search_operation") as span:
            search = super().search(query)
            span.set_attribute("search_key", "search_value")
            return search
    
    def llm(self, prompt):
        with tracer.start_as_current_span("llm_operation") as span:
            llm = super().llm(prompt)
            span.set_attribute("input_tokens", llm.usage.input_tokens)
            span.set_attribute("output_tokens", llm.usage.output_tokens)
            return llm

    def rag(self, query):
        with tracer.start_as_current_span("rag_operation") as span:
            search_results = self.search(query)
            prompt = super().build_prompt(query, search_results)
            response = self.llm(prompt)
            span.set_attribute("rag_key", response.output_text)
            return response.output_text
        
ragTraced = RAGTraced()

import pandas as pd
connection = sqlite3.connect('traces.db')

if __name__ == "__main__":
    query = "How does the agentic loop keep calling the model until it stops?"
    #answer = ragTraced.rag(query)

    search = pd.read_sql_query("SELECT * FROM spans WHERE name='search_operation'", connection).to_dict()
    llm = pd.read_sql_query("SELECT * FROM spans WHERE name='llm_operation'", connection).to_dict()
    rag = pd.read_sql_query("SELECT * FROM spans WHERE name='rag_operation'", connection).to_dict()

    print("search duration time: ", search['end_time'][0]-search['start_time'][0])
    print("llm duration time: ", llm['end_time'][0]-llm['start_time'][0])
    print("rag first input tokens: ", llm['input_tokens'][0])
    print("rag second input tokens: ", llm['input_tokens'][1])
    print("rag third input tokens: ", llm['input_tokens'][2])
    print("rag fourth input tokens: ", llm['input_tokens'][3])
    connection.close()

