from flask import Flask, request, jsonify, send_file, send_from_directory
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os

app = Flask(__name__)

# Set the environment variable in Python script
os.environ["TAVILY_API_KEY"] = "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"

def llm_response(query):
    try:
        # Create the agent and tools
        memory = MemorySaver()
        search = TavilySearchResults(max_results=5)
        tools = [search]
        
        # Use LangChain's OpenAI wrapper
        model = ChatOpenAI(
            base_url="http://localhost:1234/v1",  # Assuming local API for LLMStudio
            api_key="lm-studio",
        )
        
        # Create the agent executor
        agent_executor = create_react_agent(model, tools)
        
        # Generate the response using the agent
        response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
        
        return response['output']
    except Exception as e:
        return f"An error occurred: {str(e)}"

def handle_input(query):
    try:
        # Run Tavily search first
        search = TavilySearchResults(max_results=2)
        search_results = search.invoke({"query": query})
        
        # Generate LLM response
        llm_results = llm_response(query)
        
        # Format the response
        formatted_response = {
            "search_results": search_results,
            "llm_response": llm_results
        }
        
        return formatted_response
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    response = handle_input(user_query)
    
    if "error" in response:
        html_response = f"""
        <div class="mb-2">
            <p class="font-bold">You: {user_query}</p>
        </div>
        <div class="mb-4">
            <p class="font-bold text-red-500">Error:</p>
            <p>{response['error']}</p>
        </div>
        """
    else:
        html_response = f"""
        <div class="mb-2">
            <p class="font-bold">You: {user_query}</p>
        </div>
        <div class="mb-4">
            <p class="font-bold">Agent:</p>
            <p>{response['llm_response']}</p>
        </div>
        <div class="mb-4">
            <p class="font-bold">Search Results:</p>
            <ul class="list-disc pl-5">
        """
        
        for result in response['search_results']:
            html_response += f"<li><a href='{result['url']}' target='_blank' class='text-blue-500 hover:underline'>{result['title']}</a></li>"
        
        html_response += "</ul></div>"
    
    return html_response

if __name__ == "__main__":
    app.run(debug=True)
