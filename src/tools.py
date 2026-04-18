from tavily import TavilyClient 
from langchain.tools import tool
import os 
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.environ['TAVILY_API_KEY']

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY API key not set. Please set it in .env")

tavily_client = TavilyClient(api_key = TAVILY_API_KEY)

@tool
def tavily_web_search(query: str, max_results: int = 3) -> str:
    '''
    Calls Tavily Web Search Client to search the web based on a query.
    '''
    try:
        results = tavily_client.search(query, max_results = max_results)

        # return the content of each result as a seperate item in a list and then join the result with a newline so each result is seperated by a newline
        content = [result['content'] for result in results.get('results', [])] 
        return "\n".join(content) if content else "No results found"
    except Exception as e:
        return f"Web search error: {str(e)}"