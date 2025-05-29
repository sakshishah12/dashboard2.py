from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from datetime import datetime
import json
import re

# Import helpers from shared logic
from MAIN.generic import (
    search_top_links,
    scrape_websites,
    refine_with_expert_reviewer_loop,
    expert_llm,
    reviewer_llm
)

app = FastAPI(title="Hotel Pricing Extractor API", description="Extracts competitor hotel pricing using expert-review loop.")

class HotelInfo(BaseModel):
    hotel_name: str
    hotel_location: str
    price_per_night_usd: float

class HotelPricingResponse(BaseModel):
    hotel_name: str
    hotel_location: str
    date_range: str
    competitors: List[HotelInfo]

@app.get("/extract_hotel_pricing", response_model=HotelPricingResponse)
def extract_hotel_pricing(
    hotel_name: str = Query(..., description="Main hotel name, e.g., 'Hilton Garden Inn'"),
    hotel_location: str = Query(..., description="Hotel location, e.g., 'Stony Brook, NY'"),
    start_date: str = Query(..., description="Start date, format YYYY-MM-DD"),
    end_date: str = Query(..., description="End date, format YYYY-MM-DD")
):
    try:
        # Format date
        start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%B %d, %Y")
        end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%B %d, %Y")
        date_range = f"{start} - {end}"
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    # Search query
    search_query = f"""Find top websites or sources that provide up-to-date competitive hotel pricing information for "{hotel_name}" and surrounding areas in "{hotel_location}"."""

    # Expert prompt template
    expert_prompt_template = f"""
You are a hotel pricing expert with access to raw scraped data.

Your task:
- Identify the top 5 real competitor hotels near "{hotel_name}", located in "{hotel_location}".
- Only include legitimate competitors (exclude hostels, dorms, or motels unless explicitly relevant).
- Base competitor selection and pricing on hotel category (e.g., luxury, boutique, mid-scale, budget).
- Ensure that prices are realistic for the market, season, and demand, and specific to the date range: {date_range}.

For each competitor, provide:
- "hotel_name": string
- "hotel_location": string (including city and state if available)
- "price_per_night_usd": number (integer or float)

Output format:
Return only a strict JSON array containing exactly 5 hotel objects. Do not add any explanation or commentary. Only include valid JSON.

Text:
{{raw_text}}
"""

    # Reviewer prompt template
    reviewer_prompt_template = f"""
You are a hotel pricing reviewer.

Instructions:
- Carefully review the JSON list of hotels and their prices below:
{{extracted_output}}
- Validate that pricing logic makes sense (e.g., star rating vs. price, luxury vs. budget).
- Detect and flag any unrealistic entries or outliers.
- Confirm that the dates are within the specified range: {date_range}.
- Ensure that all competitors are located near or in "{hotel_location}".
- Ensure output is a valid JSON array of exactly 5 hotel objects with fields: "hotel_name", "hotel_location", "price_per_night_usd".
- Do not include any explanation or text outside the JSON.

Output format:
Return only the strict JSON array of hotel objects, fully corrected and sorted by price if necessary.
"""

    urls = search_top_links(search_query)
    raw_texts = scrape_websites(urls)
    combined_text = "\n\n".join(raw_texts)

    output_text, _ = refine_with_expert_reviewer_loop(
        raw_text=combined_text,
        expert_llm=expert_llm,
        reviewer_llm=reviewer_llm,
        expert_prompt_template=expert_prompt_template,
        reviewer_prompt_template=reviewer_prompt_template,
        max_iterations=3
    )

    try:
        # Clean markdown-style output
        if output_text.startswith("```json"):
            output_text = output_text.replace("```json", "").rstrip("```").strip()

        match = re.search(r"\[.*\]", output_text, re.DOTALL)
        competitors = json.loads(match.group(0) if match else output_text)
    except Exception as e:
        competitors = [{"error": "Invalid JSON", "details": str(e)}]

    return HotelPricingResponse(
        hotel_name=hotel_name,
        hotel_location=hotel_location,
        date_range=date_range,
        competitors=competitors
    )
