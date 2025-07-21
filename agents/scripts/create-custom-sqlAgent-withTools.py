# Last updated - June 2024
# This script creates a agent with various tools and initializes it for use.
# The script might be outdated but captures the basic essence of building a custom agent from scratch using langchain.

from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from typing import Tuple, Dict
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import MessagesPlaceholder
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS

from typing import Union
import re
import pprint

# from sql_analyzer.config import cfg
# from sql_analyzer.log_init import logger
# from sql_analyzer.sql.sql_tool import ExtendedSQLDatabaseToolkit
# from sql_analyzer.sql_db_factory import sql_db_factory


import os
from dotenv import load_dotenv
from google.cloud import aiplatform, bigquery
from langchain.llms.vertexai import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from tools import CircumferenceTool, ForecastingTool, GoogleSqlTool, RagTool



from sqlalchemy.engine import create_engine
from langchain.sql_database import SQLDatabase

FINAL_ANSWER_ACTION = "Final Answer:"


#----------------------------#----------------------------#----------------------------#----------------------------#

from langchain.tools.sql_database.tool import BaseSQLDatabaseTool
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from typing import Optional, List, Any
from json import dumps


class ListViewSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting view names."""

    name = "sql_db_list_views"
    description = "Input is an empty string, output is a comma separated list of views in the database."

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for a specific view."""
        return ", ".join(self.db._inspector.get_view_names())

    async def _arun(
        self,
        tool_input: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("ListTablesSqlDbTool does not support async")


class ListIndicesSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting view names."""

    name = "sql_db_list_indices"
    description = """Input is an a list of tables, output is a JSON string with the names of the indices, column names and wether the index is unique.

    Example Input: "table1, table2, table3, table4"
    """

    def _run(
        self,
        table_names: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the indices for all tables."""
        tables: List[str] = table_names.split(", ")
        indices_list: List[List[Any]] = []
        try:
            for table in tables:
                indices: List[Any] = self.db._inspector.get_indexes(table)
                indices_list.extend(indices)
            return dumps(indices_list)
        except Exception as e:
            return f"Error: {e}"

    async def _arun(
        self,
        table_names: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("ListTablesSqlDbTool does not support async")


class InfoViewSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database."""

    name = "sql_view_schema"
    description = """
    Input to this tool is a comma-separated list of views, output is the schema and sample rows for those views.    

    Example Input: "view1, view2, view3"
    """

    def _run(
        self,
        view_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for views in a comma-separated list."""
        try:
            views = view_names.split(", ")
            view_info = ""
            meta_tables = [
                tbl
                for tbl in self.db._metadata.sorted_tables
                if tbl.name in set(views)
                and not (self.db.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
            ]
            for i, view in enumerate(views):
                view_def = self.db._inspector.get_view_definition(view)
                view_info += view_def
                view_info += "\n\n/*"
                view_info += f"\n{self.db._get_sample_rows(meta_tables[i])}\n"
                view_info += "*/"
            # return view_info
            return self.db.get_table_info_no_throw()
        except Exception as e:
            """Format the error message"""
            return f"Error: {e}"

    async def _arun(
        self,
        table_name: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("SchemaSqlDbTool does not support async")


class ExtendedSQLDatabaseToolkit(SQLDatabaseToolkit):
    def get_tools(self) -> List[BaseTool]:
        base_tools = super(ExtendedSQLDatabaseToolkit, self).get_tools()
        db = base_tools[0].db
        base_tools.append(ListViewSQLDatabaseTool(db=db))
        base_tools.append(InfoViewSQLDatabaseTool(db=db))
        base_tools.append(ListIndicesSQLDatabaseTool(db=db))
        return base_tools



#----------------------------#----------------------------#----------------------------#----------------------------#

class ExtendedMRKLOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = self.includes_final_answer(text)
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    "Parsing LLM output produced both a final answer "
                    f"and a parse-able action: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation="Invalid Format: Missing 'Action:' after 'Thought:'",
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation="Invalid Format:"
                " Missing 'Action Input:' after 'Action:'",
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    def includes_final_answer(self, text):
        includes_answer = (
            FINAL_ANSWER_ACTION in text or FINAL_ANSWER_ACTION.lower() in text.lower()
        )
        return includes_answer

    @property
    def _type(self) -> str:
        return "mrkl"


def setup_memory() -> Tuple[Dict, ConversationBufferMemory]:
    """
    Sets up memory for the open ai functions agent.
    :return a tuple with the agent keyword pairs and the conversation memory.
    """
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    return agent_kwargs, memory


def setConfigs():
    """Initilize configurations or creds"""
    os.chdir('../config/')
    load_dotenv('.env')
    pprint('configs set')

def prepareTools():
    """build tools for llm"""
    tools = [CircumferenceTool(), GoogleSqlTool(), ForecastingTool(), RagTool()]
    return tools

def prepareMemory():
    """memory to store last 5 chats"""
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    pprint('memory created')
    return memory


def prepareLLM():
    """Initialize llm"""
    aiplatform.init()
    llm=ChatVertexAI(model_name='gemini-1.5-flash', temperature=0.0, max_output_tokens=1024, top_p=0.95, top_k=40)
    # print(llm)
    pprint('llm initialized')
    return llm


def prepareDB():
    """Initilize & setup db for langchains"""
    
    # Dev project
    project = 'gcp-project-name'
    dataset = 'bq-poc-testclient'
    tables = ['test-table-1', 'test-table-2', 'test-table-3']

    dbConn = f'bigquery://{project}/{dataset}'
    engine = create_engine(dbConn)
    pprint(dbConn)

    client = bigquery.Client()
    table = client.get_table('bq-poc-testclient.test-table-1')
    # schema = ["{0} {1} {2}".format(field.name, field.field_type, field.description) for field in table.schema]
    # schema = '{}'.format(table.schema)
    # schema = ["{0} {1} {2}".format(field.name, field.field_type, field.description) for field in table.schema]
    # schema = ','.join(schema)

    # db = SQLDatabase(engine=engine)
    db = SQLDatabase(engine=engine,include_tables=[x for x in tables])
    # db = SQLDatabase(engine=engine,include_tables=[x for x in tables],sample_rows_in_table_info=3, custom_table_info=custom_table_info) # explore, to provide schema of used tables
    pprint('db connection established & prepared')

    return db





def init_sql_db_toolkit(llm) -> SQLDatabaseToolkit:
    # db: SQLDatabase = sql_db_factory()
    db: SQLDatabase = prepareDB()
    print(llm)
    toolkit = ExtendedSQLDatabaseToolkit(db=db, llm=llm)
    return toolkit


def initialize_agent(llm, toolkit: SQLDatabaseToolkit) -> AgentExecutor:
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=setup_memory(),
    )
    return agent_executor


def agent_factory() -> AgentExecutor:
    setConfigs()
    llm=prepareLLM()
    sql_db_toolkit = init_sql_db_toolkit(llm)
    agent_executor = initialize_agent(llm, sql_db_toolkit)
    agent = agent_executor.agent
    agent.output_parser = ExtendedMRKLOutputParser()
    return agent_executor


if __name__ == "__main__":
    agent_executor = agent_factory()
    result = agent_executor.run("Describe all tables")
    # Include more questions here or modify to interate interactively
    pprint(result)