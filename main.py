from agents.extensions.models.litellm_model import LitellmModel
from agents import Agent, Runner
import asyncio
import os

OLLAMA_BASE = "http://localhost:11434"

model = LitellmModel(
    model="ollama/gpt-oss:20b",
    base_url=OLLAMA_BASE,
    api_key=None,
)

async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model
    )

    result = await Runner.run(
        agent,
        "Write a haiku about recursion in programming."
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
