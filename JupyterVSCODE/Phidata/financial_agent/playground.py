import os
import logging
from dotenv import load_dotenv

from phi.agent import Agent
from phi.model.google import Gemini        # ✅ Use phidata Gemini wrapper
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app

# ============ Enable Debug Logging ============
logging.basicConfig(level=logging.DEBUG)

# ============ Load Keys ============
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")   # ✅ Consistent variable name

if not google_api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

# ============ Agents ============
# Web Search Agent
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Gemini(id="gemini-1.5-flash", api_key=google_api_key),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Finance Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Gemini(id="gemini-1.5-flash", api_key=google_api_key),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
            
        )
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# ============ Playground ============
app = Playground(
    agents=[finance_agent, websearch_agent],
    stream=False    # ✅ disable streaming for Gemini
).get_app()

if __name__ == "__main__":
    serve_playground_app(
        "playground:app",   # filename:variable
        host="127.0.0.1",
        port=7777,
        reload=True
    )

