#!/usr/bin/env python3
"""Fetch Exa news briefing and push to Home Assistant REST API."""
import random
from pathlib import Path
import sys
from datetime import datetime, timezone, date, timedelta

import requests

from exa_py import Exa

HA_URL = "http://10.27.81.207:8123"
HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIzZDMyMzcxZWVhNGM0NDg3ODlhMjZjYmQ5YmYwMTFkZCIsImlhdCI6MTc3NjExODgxMCwiZXhwIjoyMDkxNDc4ODEwfQ.ZKVSGYGQDXRxp3ans4LWGEeNd4kakdACOJaR3B0LkdY"

def fetch_briefing() -> dict | None:
    exa=Exa("2a346e12-05d4-48b3-8193-d2ae8fc32b60")

    result = exa.search(
        """Return up to 9 news stories broken down as follows. 
Prioritize local news over national, and national over global. 
Prioritize stories with significant impact or importance, and avoid filler or minor-interest items. 
If a major global event is dominating the news cycle, include only the single most factual, update-oriented result for it.

Alaska / Fairbanks News
  - 0 to 3 stories
  - Include only stories with meaningful impact or significance.
  - Do not include filler, crime blotter items, routine accidents, or minor local-interest stories.
  - It is preferable to return 1 or 2 Alaska stories than to fill the section with lower-significance items.

US News
  - 1 to 3 stories

Global News
  - 1 to 3 stories

Additional Significant News
  - Fill remaining slots with the most important distinct developments regardless of category.
""",
        category = "news",
        num_results = 9,
        output_schema = {
            "description": "Schema for a collection of news stories separated into categories",
            "type": "object",
            "required": ["global_stories","us_stories","local_stories"],
            "properties": {
                "global_stories": {
                    "type": "array",
                    "description": "List of global news stories",
                    "items": {
                        "type": "object",
                        "required": ["headline", "summary"],
                        "properties": {
                            "headline": {
                                "type": "string",
                                "description": "The main headline of the news story"
                            },
                            "summary": {
                                "type": "string",
                                "description": "A brief summary of the news story"
                            }
                        }
                    }
                },
                "us_stories": {
                    "type": "array",
                    "description": "List of US news stories",
                    "items": {
                        "type": "object",
                        "required": ["headline", "summary"],
                        "properties": {
                            "headline": {
                                "type": "string",
                                "description": "The main headline of the news story"
                            },
                            "summary": {
                                "type": "string",
                                "description": "A brief summary of the news story"
                            }
                        }
                    }
                },
                "local_stories": {
                    "type": "array",
                    "description": "List of local news stories",
                    "items": {
                        "type": "object",
                        "required": ["headline", "summary"],
                        "properties": {
                            "headline": {
                                "type": "string",
                                "description": "The main headline of the local news story"
                            },
                            "summary": {
                                "type": "string",
                                "description": "A brief summary of the local news story"
                            }
                        }
                    }
                }
            }
        },
        start_published_date = (datetime.now(tz=timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        system_prompt = "- do not include news from minor, niche, or unreliable sources.\n- DO NOT INCLUDE advice columns, Opinion pieces, repetitive punditry, sensationalist reaction pieces, investor class-actions, or marketing fluff.\n- If a major global event is dominating the\nnews cycle, include only the single most factual, update-oriented result for it.\n- Prioritize major operational milestones, scientific breakthroughs, and significant geopolitical actions. \n- Span different categories (e.g., Space, Science, International Affairs, Tech).\n- If an article primarily discusses events that occurred previously, include it only if it contains substantial new information. ",
        type = "deep",
        contents = {
            "max_age_hours": 24
        }
    )

    if result:
        result = result.output.content
        result['fetched_at'] = datetime.now(timezone.utc).isoformat()

    return result

def get_tdih_options():
    url = f"https://byabbe.se/on-this-day/{ date.today().month }/{ date.today().day }/events.json"
    resp = requests.get(url)
    if resp.status_code != 200:
        return ""
    events = resp.json().get("events", [])
    sample = random.sample(events, min(10, len(events)))
    lines = [f"- {e['year']}: {e['description']}" for e in sample]
    header = f"Historical events on {date.today().strftime('%B %d')}:"
    return header + "\n" + "\n".join(lines)    
    
    
def pick_random_category():
    lines = (Path(__file__).parent / "briefing_categories.txt").read_text().splitlines()
    lines = [l for l in lines if l.strip()]
    
    rng = random.Random(date.today().toordinal())
    rng.shuffle(lines)
    return lines[0].strip()
    
def build_llm_cache(stories: str) -> bool:
    daily_fact_category = pick_random_category()
    tdih = get_tdih_options()
    
    prompt = f"""Today is: {datetime.now().strftime('%A, %B %d')} (make sure to mention the day of the week)

{stories.strip()}
    
Fun fact category: {daily_fact_category}

{tdih}    
    """
    
    print("Updating LLM with cached prompt:")
    print(prompt)
    try:
        resp = requests.post(
            "http://localhost:11434/generate/cache_news",
            json={
                'news': prompt,
            },
            timeout = 10
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to update LLM cache with news: {e}")
        return False
    
def format_result(result: dict) -> str:
    """
    Format a news search result dict into a compact plain-text string
    suitable for passing to an LLM.

    Returns a string with national and local stories separated, each story
    rendered as a compact headline + summary block.
    """
    global_stories = result.get("global_stories", [])
    us_stories = result.get("us_stories", [])
    local_stories = result.get("local_stories", [])

    lines: list[str] = []
    
    total_stories = len(global_stories) + len(us_stories) + len(local_stories)

    def _render_stories(stories: list[dict]) -> list[str]:
        out = []
        for i, story in enumerate(stories, 1):
            headline = story.get("headline", "").strip()
            summary = story.get("summary", "").strip()
            out.append(f"{i}. {headline}")
            if summary:
                out.append(f"   {summary}")
        return out

    if global_stories:
        lines.append("## Global")
        lines.extend(_render_stories(global_stories))

    if us_stories:
        if lines:
            lines.append("")
        lines.append("## US")
        lines.extend(_render_stories(us_stories))

    if local_stories:
        if lines:
            lines.append("")
        lines.append("## Local")
        lines.extend(_render_stories(local_stories))

    return "\n".join(lines), total_stories


def main():
    result = fetch_briefing()
    if result:
        text_result, total_stories=format_result(result)
        if build_llm_cache(text_result):
            print(f"LLM Cache built with {total_stories}")
    else:
        print("Failed — HA sensor unchanged", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
