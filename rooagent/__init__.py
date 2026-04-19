"""Package initialization for rooagent.

Importing `agent` may require external API keys during runtime (e.g. when the
ChatOpenAI client is configured). To avoid forcing test runners to have those
credentials set, import `agent` lazily and fail gracefully when initialization
raises an exception. This file will be restored after temporary test runs.
"""

try:
	from .agent import agent, tools
except Exception:
	# Allow importing rooagent without initializing the agent (useful for tests).
	agent = None
	tools = []