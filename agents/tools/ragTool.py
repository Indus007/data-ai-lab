from langchain.embeddings import HuggingFaceEmbeddings, VertexAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool
from langchain.document_loaders import TextLoader
import pinecone
from langchain.vectorstores import FAISS, Chroma, Pinecone
from langchain.document_loaders import PyPDFLoader
from utils import printf
from .__common__ import chat_input_filename


class RagTool(BaseTool):
    name = "Retrieval augmented generation tool"
    description = "you must use this tool when user like to query with a pdf file."

    def _run(self, question: str) -> str:
        return self.getAnswer(question)

    def _arun(self, question: str):
        raise NotImplementedError('This tool does not support async')

    
    def getAnswer(self, question):
        vectorStore = self.prepareVectorStore()
        docs = vectorStore.similarity_search(query=question)
        content = ''
        for doc in docs:
            content = content.__add__(doc.page_content)
        # print(content)
        return content

    def prepareVectorStore(self):
        documents = self.prepareDocs()
        embeddings = self.prepareEmbeddings()
        vectorStore = FAISS.from_documents(documents=documents, embedding=embeddings)
        # # pincone also works perfectly fine
        printf('vector store initialized')
        return vectorStore

    def prepareDocs(self):
        loader = PyPDFLoader(chat_input_filename)   # file uploaded to server
        pdfDocs = loader.load()
        # pdfDocs = loader
        
        # pdfDocs = ['../data/vectordata/hotel_booking.pdf']
        # pdfDocs = [chat_input_filename]
        
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000, 
            chunk_overlap=200,
            length_function = len)
        docs = text_splitter.split_documents(pdfDocs)
        printf('docs prepared')
        # print(docs[0].page_content)
        return docs


    def prepareEmbeddings(self):
        # embeddings = HuggingFaceEmbeddings()  #hfembed doesn't work when llm initialized first, some bug
        embeddings = VertexAIEmbeddings()
        printf('embedding engine prepared')
        return embeddings

