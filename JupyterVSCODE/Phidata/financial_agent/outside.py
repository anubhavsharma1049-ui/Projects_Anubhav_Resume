import os
from phi.agent import Agent
from phi.model.google import Gemini

# Load your Google Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set. Run: export GOOGLE_API_KEY='your_key_here'")

# Initialize agent with Gemini model
agent = Agent(
    model=Gemini(id="gemini-1.5-flash", api_key=GOOGLE_API_KEY),
)

# Test streaming response
agent.print_response("Test streaming output", stream=True)
