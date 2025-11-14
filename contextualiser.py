from haystack.core.component import Component
from haystack import Document
from haystack import component, default_to_dict, default_from_dict
from tqdm import tqdm 
import openai
import re

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>

This is a document that has been split into chunks mostly of the size {chunk_size} words.
You will recieve a chunk and have contextualise it in the document. The context should be around {context_size} words.
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


@component
class ContextualTextPreProcessor:
    def __init__(self, chunk_size: int = 150, context_size: int = 50, model = "gpt-4o-mini", api_key=None):
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.api_key = api_key
        self.model = model
        self.client = None

    def warm_up(self):
        if self.client:
            return
        
        if self.api_key:
            self.client = openai.Client(api_key=self.api_key)
        else: 
            self.client = openai.Client()
            raise ValueError("OpenAI API key is missing! Set it as an environment variable or pass it explicitly.")

    def generate_context(self, doc: Document, og_docs: list[Document]) -> Document:
        """
        Generate a concise context for the chunk using GPT-4 mini.
        The context will help the model understand the chunk within its broader document.
        """
        chunk = doc.content
        og_file = doc.meta['file_path']

        # print(og_file, chunk)

        # Find the whole raw text of the document where the chunk is originally from
        og_pdf = None
        for og in og_docs:
            if og.meta['file_path'] == og_file:
                og_pdf = og.content

        if not og_pdf:
            print("System Bug did not found the og file from the chunk: {chunk}".format(doc.id))

        # Openai Message to contextualise: Should use prompt caching to speed and cost saving (eventually conside Anthropics)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": DOCUMENT_CONTEXT_PROMPT.format(doc_content=og_pdf, chunk_size=self.chunk_size, context_size=self.context_size)},
                {"role": "user", "content": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)}
            ]
        ) 
        
        return response
    
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], og_docs: list[Document]) -> list[Document]:
        """
        Process a list of documents, generating context and cleaning chunks.
        """
        if not self.client:
            raise RuntimeError("Forgot to Warmup the Contextualiser")
        processed_documents = []
        print("beginning the contextualiser process")
        for doc in tqdm(documents):
            response = self.generate_context(doc, og_docs)
            context = response.choices[0].message.content
            contextualised_content = context + doc.content
            new_doc = Document(doc.id, contextualised_content, meta=doc.meta)
            processed_documents.append(new_doc)

        return {"documents": processed_documents}

    # Unsure if this part is correct or needed
    def to_dict(self):
        return default_to_dict(self, chunk_size=self.chunk_size, context_size=self.context_size, model=self.model)

    @classmethod
    def from_dict(cls, data: dict) -> "ContextualTextPreProcessor":
        default_from_dict(cls, data)

        