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


# === CONFIGURATION ===
TAVILY_API_KEY = ""
GOOGLE_API_KEY = ""

# Initialize clients
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
expert_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
reviewer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)



def search_top_links(query, limit=3):
    search_result = tavily_client.search(query=query, search_depth="basic")
    return [r.get("url") for r in search_result.get("results", []) if "url" in r][:limit]

def scrape_websites(urls):
    scraped_data = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            scraped_data.append(text)
        except Exception as e:
            print(f"ailed to scrape {url}: {e}")
    return scraped_data

def refine_with_expert_reviewer_loop(
    raw_text,
    expert_llm,
    reviewer_llm,
    expert_prompt_template,
    reviewer_prompt_template,
    max_iterations=3,
    reviewer_acceptance_check=lambda feedback: "APPROVED" in feedback.upper(),
    additional_format_kwargs={},
):
    review_feedback = "Initial attempt. Extract information as requested."
    logs = []

    for iteration in range(max_iterations):
        expert_prompt = expert_prompt_template.format(raw_text=raw_text, review_feedback=review_feedback,**additional_format_kwargs)
        expert_output = expert_llm.invoke(expert_prompt).content.strip()

        reviewer_prompt = reviewer_prompt_template.format(
            extracted_output=expert_output,
            **additional_format_kwargs
        )
        reviewer_feedback = reviewer_llm.invoke(reviewer_prompt).content.strip()

        logs.append({
            "iteration": iteration + 1,
            "expert_output": expert_output,
            "reviewer_feedback": reviewer_feedback
        })

        if reviewer_acceptance_check(reviewer_feedback):
            return expert_output, logs

        review_feedback = reviewer_feedback

    return expert_output, logs
