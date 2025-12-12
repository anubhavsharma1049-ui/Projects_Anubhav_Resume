import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.llm.aws import BedrockLLM   
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

if not aws_access_key or not aws_secret_key:
    raise ValueError("❌ AWS credentials not found in .env file")
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=BedrockLLM(model="amazon.titan-text-express-v1", stream=True),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)


finance_agent = Agent(
    name="Finance AI Agent",
    model=BedrockLLM(model="amazon.titan-text-express-v1", stream=True),
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

# ============ Multi-Agent ============
multi_ai_agent = Agent(
    team=[websearch_agent, finance_agent],
    model=BedrockLLM(model="amazon.titan-text-express-v1", stream=True),
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response(
    "Summarize analyst recommendation and share the latest news for NVDA",
    stream=True   # ✅ enable streaming output
)
