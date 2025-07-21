from google.cloud import bigquery
from google.cloud import aiplatform
from langchain.sql_database import SQLDatabase
from sqlalchemy.engine import create_engine
from langchain.tools import BaseTool
from langchain.llms.vertexai import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from tools import CircumferenceTool
from utils import printf

import pandas as pd
import numpy as np

# from langchain import debug as debug
# debug = True

# file = '../data/output/data.csv'


class GoogleSqlTool(BaseTool):
    name = "SQL generation tool"
    description = "you must use this tool to generate sql or to create sequential queries or to build database queries. Returns output file name."

    def _run(self, question: str) -> str:
        return self.generateSQL(question)

    def _arun(self, question: str):
        raise NotImplementedError('This tool does not support async')

    def generateSQL(self, question):
        llm = self.prepareLLM()
        memory = self.prepareMemory()
        agent = self.prepareAgent(llm, memory)
        db = self.prepareDB()
        tables = db.get_usable_table_names()
        schemas = str(self.getSchema())
        prompt = self.preparePrompt(question, tables, schemas)
        # print(prompt)
        sqlquery = agent.run(prompt)
        result = self.runQuery(db, sqlquery)
        file = self.saveQueryResult(result)
        return file
        
    def runQuery(self, db, sqlquery):
        result=''
        if sqlquery != 'Please specify a valid question.' and sqlquery != 'Incorrect query. Please try again.' and sqlquery !='INVALID_QUERY':
            result = db.run(sqlquery)
            printf(result)
        else :
            printf('Not a valid question. Try again!')
        return result
    
    def saveQueryResult(self, result:str):
        file = '../data/output/data.csv'
        # print(result)
        result = eval(result)
        arr = np.array(result)
        # print(arr)
        pd.DataFrame(arr).to_csv(file, index=False)
        return file

    def prepareAgent(self, llm, memory):
        """build agent"""
        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=[CircumferenceTool()],
            llm=llm,
            verbose=True,
            max_iteration=3,
            early_stopping_method='generate',
            memory=memory
        )
        printf('sql agent created')
        return agent
    
    def prepareLLM(self):
        """Initialize llm"""
        aiplatform.init()
        llm=ChatVertexAI(model_name='gemini-1.5-pro-002', temperature=0.1, max_output_tokens=1024)
        # llm = ChatVertexAI(model_name='codechat-bison@001')
        # print(llm)
        printf('llm for sql initialized')
        return llm

    def prepareMemory(self):
        """memory to store last 5 chats"""
        memory = ConversationBufferWindowMemory(memory_key='chat_history', k=3, return_messages=True)
        printf('memory created')
        return memory

    def getSchema(self):
        client = bigquery.Client()
        table = client.get_table('performplus_poc_testclient.ng_platform_usage_team_weekly')
        # schema = '{}'.format(table.schema)
        schema = ["{0} {1} {2}".format(field.name, field.field_type, field.description) for field in table.schema]
        schema = ','.join(schema)
        # print(schema)
        return schema
        # print("Table schema: {}".format(table.schema))

    def preparePrompt(self, question, tables, schemas):
        """set-up prompt for the question"""
        
        template = """
        You are a GoogleSQL expert. Given an input question, create a syntactically correct GoogleSQL query.
        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        Also, pay attention to which column is in which table.
        Use single character aliases for tables while doing sql joins.
        Only use the following tables: {table_info}
        Here's the schema of the table {schemas}. The format of schema is - column name followed by column datatype followed by any available column description. Each column sggregated by comma. Details of a new column in same order follows after each comma.
        Please use the schema of the table. Make sure to use column description to understand the column better.
        
        Question: {question}
        
        Note that month_id value is integer and month_id format is yyyymm. 
        Make use of date columns of big query for sorting, for any queries related to time.
        If someone asks for aggregation on a STRING data type column, then CAST column as NUMERIC before you do the aggregation.
        If someone asks for average on a STRING data type column, then CAST column as NUMERIC before you do the average.
        Be mindful of years and months.

        """
        template = f"{template}. Always respond in json in the format as ```json ```"   #work for formatting but there's other errors.
        
        input_variables=["question", "table_info", "schemas"]
        prompt = PromptTemplate(input_variables=input_variables, template=template)
        prompt = prompt.format(question=question, table_info=tables, schemas=schemas)
        
        printf('prompt ready')
        return prompt


    def prepareDB(self):
        """Initilize & setup db for langchains"""
        
        # Dev project
        project = 'gcp-project-name'
        dataset = 'bq-poc-testclient'
        tables = ['test-table-1', 'test-table-2', 'test-table-3']


        dbConn = f'bigquery://{project}/{dataset}'
        engine = create_engine(dbConn)
        printf(dbConn)

        # db = SQLDatabase(engine=engine)
        db = SQLDatabase(engine=engine,include_tables=[x for x in tables])
        printf('db connection established & prepared')

        return db



