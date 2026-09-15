import os
import streamlit as st
import requests
import yagmail

from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool


# Load Environment Variables

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

EMAIL = os.getenv("EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


# Page Configuration

st.set_page_config(
    page_title="FzAgent",
)

st.title("Travelgent")
st.caption("AI Travel Assistant powered by LangChain and Groq")

st.divider()


# LLM

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=2,
    api_key=GROQ_API_KEY
)


# Tool 1: Google Search

@tool
def serpapi_search(query: str):
    """Searches Google for travel-related information."""

    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": SERP_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" in results:
        return [
            {
                "title": r["title"],
                "link": r["link"],
                "snippet": r.get("snippet", "")
            }
            for r in results["organic_results"][:5]
        ]

    return "No search results found."


# Tool 2: Weather

@tool
def get_weather(city: str):
    """Gets current weather information for a city."""

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}"
        f"&appid={WEATHER_API_KEY}"
        f"&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:

        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]

        return f"Weather in {city}: {weather}, Temperature: {temp}°C"

    return f"Weather information for {city} was not found."


# Tool 3: Send Email

@tool
def send_email(recipient: str, content: str):
    """Sends the travel itinerary to the specified email address."""

    yag = yagmail.SMTP(
        EMAIL,
        EMAIL_PASSWORD
    )

    yag.send(
        to=recipient,
        subject="AI Travel Plan",
        contents=content
    )

    return "Trip itinerary sent successfully."


# Memory

memory = InMemorySaver()


# AI Agent

agent = create_agent(
    model=llm,

    tools=[
        serpapi_search,
        get_weather,
        send_email
    ],

    system_prompt="""
You are an AI Travel Assistant.

You can:
1. Search Google for travel information.
2. Check current weather.
3. Create travel itineraries.
4. Send travel itineraries through email.

Use the available tools whenever they are useful.

When the user asks for a travel plan, create a clear and useful itinerary.

If the user asks to send the itinerary by email,
use the send_email tool.

Always provide a clear final response.
""",

    checkpointer=memory
)


# Streamlit Input

user_input = st.text_area(
    "Enter your travel request",
    placeholder=(
        "Example: Plan a 3-day trip to Turkey "
        "and send the itinerary to my email."
    ),
    height=120
)

# Run Agent

if st.button("Run Agent"):

    if not user_input.strip():

        st.warning("Please enter a travel request.")

    else:

        with st.spinner("FzAgent is working..."):

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                },

                config={
                    "configurable": {
                        "thread_id": "fzagent-user"
                    }
                }
            )

        final_response = response["messages"][-1].content

        st.divider()

        st.subheader("Agent Response")

        st.write(final_response)