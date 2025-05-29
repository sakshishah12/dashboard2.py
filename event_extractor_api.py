from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime, timedelta
import json
import re

# Import functions from Generic file
from MAIN.generic import search_top_links,scrape_websites,refine_with_expert_reviewer_loop,expert_llm,reviewer_llm

app = FastAPI(title="Event Extractor API", description="Extracts structured information using expert-review loop.")


class EventExtractionResponse(BaseModel):
    location: str
    extracted_events: List[dict]

@app.get("/extract_events", response_model=EventExtractionResponse)
def extract_events(location: str = Query(..., description="Location to search for events, e.g., 'Manhattan, NY'")):
    search_query = f"list top websites with up-to-date event information in {location}, including city calendars, community centers, tourism boards, and social media event pages"
    start_date = datetime.now() + timedelta(days=1)
    end_date = start_date + timedelta(days=2)

    expert_prompt_template = """
    You are an expert event data extractor.

    Task: Extract structured **event information for ONLY events happening between {start_date} and {end_date}** from the given text.

    Use this feedback from reviewer: "{review_feedback}"

    Text:
    {raw_text}

    Extract a JSON list with the following fields for each event:
    - "event_name": title of the event
    - "date": in "Month Day, Year" format
    - "location": venue + city/state
    - "demand_level": High, Medium, or Low

    Only include events whose dates fall within the given range.
    """


    reviewer_prompt_template = """
You are reviewing extracted event data.

Instructions:
1. Validate JSON format.
2. Ensure each object includes: event_name, date, location, demand_level.
3. Only accept events between {start_date} and {end_date}.
4. Must be relevant to {location}.
5. If valid, return "APPROVED". Otherwise, give detailed feedback.

Data:
{extracted_output}
"""

    urls = search_top_links(search_query)
    raw_texts = scrape_websites(urls)

    combined_text = "\n\n".join(raw_texts)

    output_text, logs = refine_with_expert_reviewer_loop(
        raw_text=combined_text,
        expert_llm=expert_llm,
        reviewer_llm=reviewer_llm,
        expert_prompt_template=expert_prompt_template,
        reviewer_prompt_template=reviewer_prompt_template,
        max_iterations=3,
        additional_format_kwargs={
            "start_date": start_date.strftime("%B %d, %Y"),
            "end_date": end_date.strftime("%B %d, %Y"),
            "location": location
        }
    )

    try:
        if output_text.startswith("```json"):
            output_text = output_text.replace("```json", "").rstrip("```").strip()

        match = re.search(r"\[.*\]", output_text, re.DOTALL)
        json_data = json.loads(match.group(0) if match else output_text)
    except Exception as e:
        json_data = [{"error": "Invalid JSON", "details": str(e)}]

    return EventExtractionResponse(location=location, extracted_events=json_data)
