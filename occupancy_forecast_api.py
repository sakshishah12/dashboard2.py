from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta
from generic import refine_with_expert_reviewer_loop, expert_llm, reviewer_llm
import json

app = FastAPI()

# === Request and Response Models ===
class BookingInput(BaseModel):
    hotel_name: str
    hotel_location: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    confirmed_bookings: List[dict]
    events_json: List[dict]

class ForecastOutput(BaseModel):
    forecast_logs: List[dict]
    final_forecast: str

# === Prompt Templates for Expert and Reviewer ===
expert_prompt_template = """
You are a hotel revenue management AI specializing in booking prediction.

Hotel: {hotel_name}
Location: {hotel_location}
Forecast Period: {date_range}

Confirmed Bookings Data:
{confirmed_bookings}

Event Data:
{events_json}

Task:
- Predict the **total bookings** for each day in the forecast period.
- Base your prediction on confirmed bookings and relevant event data.

Respond in JSON array format:
[
  {{"date": "YYYY-MM-DD", "predicted_total_bookings": <integer>, "reasoning": "..." }},
  ...
]
"""

reviewer_prompt_template = """
You are a reviewer evaluating the predicted total bookings.

Event Data: {events_json}

Instructions:
- Ensure each entry includes "date", "predicted_total_bookings", and "reasoning".
- Check for illogical or inconsistent predictions based on events and trends.
- Suggest corrections if needed.

Forecast to Review:
{extracted_output}

Respond with:
{{"feedback": "<review comments>", "corrected_forecast": [ ... ]}}
"""

# === Forecast Endpoint ===
@app.post("/forecast", response_model=ForecastOutput)
async def forecast_total_bookings(input_data: BookingInput):
    date_range = f"{input_data.start_date} to {input_data.end_date}"
    
    confirmed_bookings_str = json.dumps(input_data.confirmed_bookings, indent=2)
    events_json_str = json.dumps(input_data.events_json, indent=2)
    
    final_forecast, logs = refine_with_expert_reviewer_loop(
        raw_text="",
        expert_llm=expert_llm,
        reviewer_llm=reviewer_llm,
        expert_prompt_template=expert_prompt_template,
        reviewer_prompt_template=reviewer_prompt_template,
        max_iterations=3,
        reviewer_acceptance_check=lambda feedback: "APPROVED" in feedback.upper(),
        additional_format_kwargs={
            "hotel_name": input_data.hotel_name,
            "hotel_location": input_data.hotel_location,
            "date_range": date_range,
            "confirmed_bookings": confirmed_bookings_str,
            "events_json": events_json_str
        }
    )
    
@app.post("/forecast")
async def forecast_total_bookings(input_data: BookingInput):
    # ...
    final_forecast, logs = refine_with_expert_reviewer_loop(...)

    try:
        final_forecast_json = json.loads(final_forecast)
    except json.JSONDecodeError:
        final_forecast_json = {"error": "Invalid JSON format in forecast output."}

    return final_forecast_json
