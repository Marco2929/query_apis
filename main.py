import gradio as gr
import os
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
import requests

# Initialize the OpenAI model
llm = OpenAI(model="gpt-4o-mini")


def getWeather(lat, lon, exclude=None, units="standard", lang="en", api_key=os.environ.get("WEATHER_API")):
    """This function calls the OpenWeatherMap One Call API to get weather data."""
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat,
        "lon": lon,
        "exclude": exclude,
        "units": units,
        "lang": lang,
        "appid": api_key,
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API call failed with status code {response.status_code}", "details": response.text}


def weather_agent(prompt):
    # Create a tool for the getWeather function
    weather_tool = FunctionTool.from_defaults(fn=getWeather)

    # Set up the agent worker
    agent_worker = FunctionCallingAgentWorker.from_tools(
        [weather_tool], llm=llm, verbose=True, allow_parallel_tool_calls=False
    )

    # Create the agent
    agent = agent_worker.as_agent()

    # Process the input prompt
    return agent.chat(prompt)


# Gradio interface
def gradio_weather_agent(prompt):
    try:
        response = weather_agent(prompt)
        return response.response
    except Exception as e:
        return {"error": str(e)}


# Create Gradio interface
demo = gr.Interface(
    fn=gradio_weather_agent,
    inputs=gr.Textbox(label="Prompt", placeholder="Ask about the weather..."),
    outputs=gr.Markdown(label="Response"),
    title="Wetter Chatbot",
    description="Interact with the weather agent to get weather information. Simply provide a prompt describing your query.",
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
