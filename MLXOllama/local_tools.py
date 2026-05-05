import json

from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx
import pytz

from rapidfuzz import process, fuzz, utils

from . import cache_utils

LOCAL_TOOLS = {
    "pause_music": {
        "type": "function",
        "function": {
            "name": "pause_music",
            "description": "Pauses the music. You MUST call this every time a user asks to pause or stop the music",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    "list_playlists": {
        "type": "function",
        "function": {
            "name": "list_playlists",
            "description": "Retrieves all playlists from the library. Call this tool first to find the 'id' of a playlist when the user refers to one by name, or to answer questions about available playlists. Returns a list of playlist objects with 'id' and 'name' properties.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    "get_current_playback": {
        "type": "function",
        "function": {
            "name": "get_current_playback",
            "description": "Retrieves the current playback state from Apple Music, including whether something is playing/paused/stopped, the current track details (title, artist, album), playback position, and duration. Use this when the user asks 'what's playing', 'what song is this', or needs context before controlling playback.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    # "list_albums": {
        # "type": "function",
        # "function": {
            # "name": "list_albums",
            # "description": "Retrieves all albums from the library. Call this tool first to find the exact title and artist for a requested album, or to provide information about an album",
            # "parameters": {
                # "type": "object",
                # "properties": {
                    # 'offset': {
                        # 'type': "integer",'description': "The offset to use when listing albums",
                    # },
                    # 'limit': {
                        # 'type': 'integer','description': "The number of albums to return in the result set",
                    # },
                # },
                # "required": []
            # }
        # }
    # },
    "search_music": {
        "type": "function",
        "function": {
            "name": "search_music",
            "description": "Searches the library for albums, artists, or specific tracks. Use this as the primary tool for finding music when the exact title is unknown or to verify metadata. Searches by a single criteria (track, album, artist) ONLY. IMPORTANT: Do not proivde multiple criteria when searching. If user provides both track and either artist and/or album, search by track NAME ONLY.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term (e.g., 'Dark Side', 'Pink Floyd', or 'Wish You Were Here')."
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max results to return."
                    }
                },
                "required": ["query"]
            }
        }
    },
    "play_track": {
        "type": "function",
        "function": {
            "name": "play_track",
            "description": "Plays a single track by its persistent ID. If the ID is unknown, call search_tracks first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "string", "description": "The persistent ID of the track."}
                },
                "required": ["track_id"]
            }
        }
    },
    "play_playlist": {
        "type": "function",
        "function": {
            "name": "play_playlist",
            "description": "Plays a playlist by ID. Use list_playlists first to resolve a name to an ID. Support optional shuffling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_id": {"type": "string", "description": "The unique ID of the playlist."},
                    "shuffle": {"type": "boolean", "default": False, "description": "Whether to shuffle the playlist."}
                },
                "required": ["playlist_id"]
            }
        }
    },
    "play_album": {
        "type": "function",
        "function": {
            "name": "play_album",
            "description": "Plays a specific album. Requires exact artist and album names, as returned by search_music. Use search_music first to get the exact names",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {"type": "string", "description": "The name of the artist."},
                    "album_name": {"type": "string", "description": "The title of the album."},
                    "shuffle": {"type": "boolean", "default": False}
                },
                "required": ["artist_name", "album_name"]
            }
        }
    },
    "play_artist": {
        "type": "function",
        "function": {
            "name": "play_artist",
            "description": "Plays all tracks by a specific artist. Defaults to shuffle mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {"type": "string", "description": "The name of the artist."},
                    "shuffle": {"type": "boolean", "default": True}
                },
                "required": ["artist_name"]
            }
        }
    },
    "get_current_time": {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in the user's timezone. Always use this when the query involves now, today, tomorrow, current time, scheduling, deadlines, recency, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Optional IANA timezone name e.g. 'America/Anchorage'. Defaults to user's known timezone."
                    }
                },
                "required": []
            }
        }
    },
    "local_entity_state": {
        "type": "function",
        "function": {
            "name": "local_entity_state",
            "description": "Retrieve the current state and key attributes of one or more entities. Use this tool whenever a question involves the current status, value, location, health, on/off state, brightness, temperature, position, or any real-time property of an entity. Prefer fetching multiple relevant entities in a single call. NOTE: This tool name DOES NOT have a prefix. Call it exactly as 'local_entity_state'",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "String"},
                        "description": "List of entity IDs to fetch (e.g. ['light.kitchen_ceiling', 'sensor.backyard_temperature']). Use a list even for one entity."
                    }
                },
                "required": ['entity_ids'],
            }
        }
    },
    "get_current_weather": {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Retrieves real-time weather forecasts from the NWS for a specific sector. Use for current, daily, or weekend forecasts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": { "type": "number", "description": "Decimal latitude (e.g., 64.837)." },
                    "longitude": { "type": "number", "description": "Decimal longitude (e.g., -147.716)." },
                    "location": { "type": "string", "description": "City/State name for non-local scans." },
                    "timeframe": {
                        "type": "string",
                        "enum": ["today", "tomorrow", "this weekend", "7 day"],
                        "description": "The temporal window for the sensor sweep. Defaults to 'today'."
                    }
                },
                "required": ["location"]
            }
        }
    }
}

async def get_current_time(timezone: str | None = None) -> str:
    """
    Returns current date and time as a string.

    Args:
        timezone: Optional IANA timezone name (e.g. 'America/Anchorage', 'Europe/London', 'UTC')
                 If None, uses the system's local timezone (or fallback to UTC).

    Returns:
        Formatted string like: "2026-03-17 14:55 AKDT"
    """
    if timezone:
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            # Fallback when invalid timezone name is passed
            tz = pytz.UTC
            timezone = "UTC (fallback - invalid timezone)"
    else:
        # Use system local time if no timezone specified
        tz = datetime.now().astimezone().tzinfo
        # or strictly: tz = pytz.utc  ← choose one philosophy

    now = datetime.now(tz)

    # Most readable format for agents & humans
    return now.strftime("%Y-%m-%d %-I:%M:%S %p %Z")

async def local_entity_state(entity_ids: str | list[str]) -> str:
    from . import cache_utils

    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]

    result = {}
    missing = []

    for eid in entity_ids:
        state = cache_utils.state_cache.get(eid)
        if state is not None:
            result[eid] = state
        else:
            missing.append(eid)

    # Include missing entities so LLM knows what failed
    if missing:
        result["_missing"] = missing

    # Return compact JSON (LLMs prefer this over pretty-printed)
    return json.dumps(result, separators=(",", ":"))


async def get_current_weather(latitude=None, longitude=None, location=None, timeframe="today"):
    headers = {"User-Agent": "Starfleet-Command-AVO-Assistant/1.0 (israel@MacStudio)"}
    now = datetime.now() # Mar 19, 2026 (Thursday)

    async with httpx.AsyncClient(timeout=20.0) as client:

        # 1. Sensor Selection
        if latitude and longitude:
            # Use direct coordinates (Highest Precision / Lowest Latency)
            lat, lon = latitude, longitude
        elif location:
            # 1. Geocode
            geo_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
            geo_res = await client.get(geo_url, headers=headers)
            if not geo_res.json(): return "Sector not found."
            lat, lon = geo_res.json()[0]["lat"], geo_res.json()[0]["lon"]

        #  Make sure lat/lon are numbers
        lat = float(lat)
        lon = float(lon)

        # 2. Get NWS Grid
        pts_res = await client.get(
            f"https://api.weather.gov/points/{lat:.3f},{lon:.3f}",
            headers=headers,
            follow_redirects=True
        )
        if pts_res.status_code != 200:
            resp = {
                'result': "ERROR",
                'content': pts_res.text,
            }
            return json.dumps(resp)

        forecast_url = pts_res.json()["properties"]["forecast"]

        # 3. Get Full Forecast (Next 7 days)
        f_res = await client.get(forecast_url, headers=headers)
        periods = f_res.json()["properties"]["periods"]

        # 4. Temporal Logic: Define "This Weekend"
        # Since today is Thursday (weekday 3), Friday is +1, Sat is +2, Sun is +3
        days_to_friday = (4 - now.weekday()) % 7
        friday_date = (now + timedelta(days=days_to_friday)).date()
        sunday_date = friday_date + timedelta(days=2)

        if timeframe == "this weekend":
            return json.dumps([p for p in periods if friday_date <=
                                datetime.fromisoformat(p["startTime"]).date() <= sunday_date])

        if timeframe == "today":
            return json.dumps([p for p in periods if
                                datetime.fromisoformat(p["startTime"]).date() == now.date()])

        return json.dumps(periods[:6]) # Default to 3 days (day/night pairs)

################ MUSIC PLAYBACK USING apple-music-custom ########################
from .config import ITUNES_URL

async def get_current_playback(server_base_url: str = ITUNES_URL) -> dict[str, Any]:
    """Get the current playback state from the apple-music-custom server.

    Returns detailed info about what's currently playing (or paused/stopped),
    including track metadata, artist, album, playback position, state, etc.

    This is the best endpoint for "what's playing right now?" queries.
    """
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                f"{server_base_url.rstrip('/')}/now_playing"
            )
            resp.raise_for_status()
            data = resp.json()

            # Optional: Add a human-friendly summary for the LLM
            if data and isinstance(data, dict):
                state = data.get("state", "stopped")
                track = data.get("track", {})

                if track and state in ("playing", "paused"):
                    summary = f"{state.capitalize()}: {track.get('name')} by {track.get('artist')} " \
                              f"from {track.get('album')}"
                else:
                    summary = "Nothing is currently playing."

                data["friendly_summary"] = summary  # helpful for LLM reasoning

            return json.dumps(data)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return json.dumps(
                {"state": "stopped", "friendly_summary": "Nothing is currently playing."}
            )
        return json.dumps({"error": f"HTTP error: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to get current playback: {str(e)}"})


async def _set_shuffle(enable: bool):
    """
    Internal helper to set shuffle mode via PUT /shuffle.
    The body must be 'mode=songs' to enable or 'mode=off' to disable.
    """
    mode = "songs" if enable else "off"
    async with httpx.AsyncClient() as client:
        await client.put(f"{ITUNES_URL}/shuffle", data={"mode": mode})

async def pause_music():
    """
    PUT /pause.
    Pauses the music.
    """
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/pause")
        response.raise_for_status()
        return response.text

async def list_playlists():
    """
    GET /playlists.
    Returns all playlists. Use this to map a playlist name to its required 'id'.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ITUNES_URL}/playlists")
        response.raise_for_status()
        return response.text

async def list_albums(offset: int = 0, limit: int = 100):
    """
    GET /albums.
    Returns all albums.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ITUNES_URL}/library/albums", params={"offset": offset, 'limit': limit,})
        response.raise_for_status()
        return response.text

async def search_music(query: str, server_base_url: str = "http://localhost:8181"):
    """Unified search across cached albums/artists + server track search.

    Returns candidates in a consistent format so the LLM can pick the best match.
    """
    results = {
        'albums': [],
        'artists': [],
        'tracks': [],
    }

    # 1. Fuzzy search on your local album cache
    if cache_utils.itunes_cache.get('albums'):
        album_matches = process.extract(
            query,
            cache_utils.itunes_cache['albums'].keys(),
            scorer=fuzz.WRatio,
            processor=utils.default_process,
        )
        for title, score, _ in album_matches:
            if score > 55:
                item = dict(cache_utils.itunes_cache['albums'][title])
                item['confidence'] = round(score)
                results['albums'].append(item)

    # 2. Fuzzy search on your local artist cache
    if cache_utils.itunes_cache.get('artists'):
        artist_matches = process.extract(
            query,
            cache_utils.itunes_cache['artists'].keys(),
            scorer=fuzz.WRatio,
            processor=utils.default_process,
        )
        for name, score, _ in artist_matches:
            if score > 55:
                item = dict(cache_utils.itunes_cache['artists'][name])
                item['confidence'] = round(score)
                results['artists'].append(item)

    # 3. Server search for tracks (the missing piece)
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{server_base_url.rstrip('/')}/library/search",
                params={"q": query}
            )
            resp.raise_for_status()
            track_data = resp.json()   # Expecting a list of track dicts

            # Add type + confidence so everything is consistent
            for track in track_data['tracks']:  # adjust limit as needed
                item = dict(track)
                item['confidence'] = 80
                results['tracks'].append(item)

    except httpx.HTTPError as e:
        print(f"Track search to server failed: {e}")
        # results['tracks'] stays empty — LLM can still use album/artist matches
    except Exception as e:
        print(f"Unexpected error in track search: {e}")

    return json.dumps(results)



async def play_track(track_id: str):
    """
    PUT /library/tracks/:id/play.
    Plays a specific track by its persistent ID.
    """
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/library/tracks/{track_id}/play")
        response.raise_for_status()
        return response.text

async def play_playlist(playlist_id: str, shuffle: bool = False):
    """
    PUT /playlists/:id/play.
    Starts a playlist. Configures shuffle mode before initiating playback.
    """
    await _set_shuffle(shuffle)
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/playlists/{playlist_id}/play")
        response.raise_for_status()
        return response.text

async def play_album(artist_name: str, album_name: str, shuffle: bool = False):
    """
    PUT /library/albums/:artist/:album/play.
    Plays an album. Requires exact artist and album names for the URI path.
    """
    await _set_shuffle(shuffle)
    # quote() ensures spaces/special characters are safe for the URI path
    safe_artist = quote(artist_name, safe='')
    safe_album = quote(album_name, safe='')
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/library/albums/{safe_artist}/{safe_album}/play")
        response.raise_for_status()
        return response.text

async def play_artist(artist_name: str, shuffle: bool = True):
    """
    PUT /library/artists/:artist/play.
    Queues all tracks by an artist. Defaults to shuffle enabled for variety.
    """
    await _set_shuffle(shuffle)
    safe_artist = quote(artist_name, safe='')
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/library/artists/{safe_artist}/play")
        response.raise_for_status()
        return response.text
