INSTRUCTIONS = '''
You're a course teaching assistant. Answer the student's question using the search tool. 
Make multiple searches with different keywords before answering.
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='openai/gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        return self.index.search(
            query,
            num_results=num_results
        )


    def build_prompt(self, query, search_results):
        context = self.__build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

    def __build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append('Content: ' + doc['content'])
            lines.append('file name: ' + doc['filename'])
            lines.append('')

        return '\n'.join(lines).strip()
