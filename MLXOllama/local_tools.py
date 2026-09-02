import json
import inspect
import re
import types

from datetime import datetime, timedelta, date
from enum import Enum
from typing import (
    Any,
    Annotated,
    Literal,
    Mapping,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID
from urllib.parse import quote

import httpx
import pytz

from rapidfuzz import process, fuzz, utils

from . import cache_utils

LOCAL_TOOLS = {}

def _json_schema(annotation: Any) -> dict[str, Any]:
    """Generate a JSON Schema dict from a Python type annotation.

    Supported:
      - str, int, float, bool, None
      - Optional[T] / T | None
      - Union[T1, T2, ...]
      - Literal[...]
      - list[T], set[T], tuple[T], tuple[T, ...]
      - dict[K, V], Mapping[K, V]
      - Sequence[T]
      - Annotated[T, ...]
      - Enum subclasses
      - NewType
      - Any
      - date, datetime, UUID

    The generated schemas intentionally use a conservative subset of JSON
    Schema suitable for LLM function-calling backends.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    # Resolve typing.NewType(...)
    # NewType objects expose their underlying type as __supertype__.
    if hasattr(annotation, "__supertype__"):
        return _json_schema(annotation.__supertype__)

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Annotated[T, ...] -> schema for T.
    if origin is Annotated:
        return _json_schema(args[0])

    # Literal[...]
    if origin is Literal:
        literal_values = list(args)
        schema: dict[str, Any] = {"enum": literal_values}

        # Only add "type" when every literal has the same exact Python type.
        if literal_values:
            first_type = type(literal_values[0])
            if all(type(value) is first_type for value in literal_values):
                if first_type is str:
                    schema["type"] = "string"
                elif first_type is bool:
                    schema["type"] = "boolean"
                elif first_type is int:
                    schema["type"] = "integer"
                elif first_type is float:
                    schema["type"] = "number"
                elif first_type is type(None):
                    schema["type"] = "null"

        return schema

    # Optional[T], T | None, or a general Union.
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        has_none = len(non_none) != len(args)

        # Optional[T] / T | None.
        #
        # Preserve nullability explicitly when None is actually part of the
        # allowed value set. This is different from requiredness, which is
        # determined separately from the function parameter's default value.
        if len(non_none) == 1:
            base_schema = _json_schema(non_none[0])

            if has_none:
                # JSON Schema can express this cleanly with anyOf.
                #
                # Example:
                #   str | None ->
                #   {"anyOf": [{"type": "string"}, {"type": "null"}]}
                return {
                    "anyOf": [
                        base_schema,
                        {"type": "null"},
                    ]
                }

            return base_schema

        # General Union.
        union_schema = {
            "anyOf": [_json_schema(arg) for arg in non_none]
        }

        if has_none:
            union_schema["anyOf"].append({"type": "null"})

        return union_schema

    # Enum classes.
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        values = [member.value for member in annotation]

        schema = {"enum": values}

        if values:
            first_type = type(values[0])
            if all(type(value) is first_type for value in values):
                if first_type is str:
                    schema["type"] = "string"
                elif first_type is bool:
                    schema["type"] = "boolean"
                elif first_type is int:
                    schema["type"] = "integer"
                elif first_type is float:
                    schema["type"] = "number"

        return schema

    # list[T] / set[T] / Sequence[T]
    if origin in (list, set, Sequence):
        schema = {"type": "array"}

        if args:
            schema["items"] = _json_schema(args[0])

        return schema

    # tuple[T, ...]
    #
    # Intentionally do not use prefixItems for fixed heterogeneous tuples.
    # That is valid modern JSON Schema, but not worth the compatibility risk
    # in an LLM tool-calling stack.
    if origin is tuple:
        schema = {"type": "array"}

        if args:
            if len(args) == 2 and args[1] is Ellipsis:
                schema["items"] = _json_schema(args[0])
            else:
                # Conservative fallback for tuple[T1, T2, ...].
                schema["items"] = {}

        return schema

    # Bare containers.
    if annotation in (list, set, tuple, Sequence):
        return {"type": "array"}

    if annotation in (dict, Mapping):
        return {"type": "object"}

    # dict[K, V] / Mapping[K, V]
    if origin in (dict, Mapping):
        schema = {"type": "object"}

        if len(args) == 2:
            key_type, value_type = args

            # JSON object keys are strings. For dict[str, V], describe the
            # values precisely via additionalProperties.
            if key_type is str:
                schema["additionalProperties"] = _json_schema(value_type)
            else:
                # Python allows non-string dict keys, but JSON objects don't.
                # Don't pretend the key constraint is enforceable here.
                schema["additionalProperties"] = _json_schema(value_type)

        return schema

    # Bare scalar types.
    if annotation is str:
        return {"type": "string"}

    if annotation is bool:
        return {"type": "boolean"}

    if annotation is int:
        return {"type": "integer"}

    if annotation is float:
        return {"type": "number"}

    if annotation is type(None):
        return {"type": "null"}

    # Date/time-like values.
    #
    # Keep these as plain strings rather than relying on JSON Schema "format",
    # since format validation is not consistently honored by tool backends.
    if annotation in (date, datetime, UUID):
        return {"type": "string"}

    # Unknown types: don't lie.
    #
    # {} means "unconstrained value", which is preferable to claiming that
    # an arbitrary Python type is a string.
    return {}

_PARAM_REGEX = re.compile(r"^\s{0,4}(\w+)\s*(?:\([^)]*\))?:\s*(.*)$")
def _parameter_descriptions(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from an Args/Parameters docstring section."""
    descriptions = {}
    in_parameters = False
    current = None

    for line in inspect.cleandoc(docstring or "").splitlines():
        clean_line = line.strip().lower().rstrip(":")

        # Detect the start of the parameters section
        if clean_line in {"args", "arguments", "parameters"}:
            in_parameters = True
            continue

        if in_parameters:
            # Break on known next-sections
            if clean_line in {"returns", "raises", "yields"}:
                break
            # Break on unknown next-sections (unindented text ending in a colon)
            if line and not line[0].isspace() and line.rstrip().endswith(":"):
                break

            match = _PARAM_REGEX.match(line)
            if match:
                current = match.group(1)
                descriptions[current] = match.group(2).strip()
            elif current and line.strip():
                descriptions[current] += f" {line.strip()}"

    return descriptions


def local_tool(func):
    """Register a local function as an OpenAI-compatible tool."""
    signature = inspect.signature(func)
    descriptions = _parameter_descriptions(func.__doc__)

    # Resolve postponed annotations / forward references.
    #
    # Falls back to the raw signature annotation if resolution fails, so one
    # problematic annotation doesn't prevent the tool from registering.
    try:
        type_hints = get_type_hints(func)
    except (NameError, TypeError, ValueError):
        type_hints = {}

    properties = {}
    required = []

    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Prefer resolved type hints, falling back to the raw annotation.
        annotation = type_hints.get(name, parameter.annotation)

        # Handle missing type hints explicitly.
        if annotation is inspect.Parameter.empty:
            annotation = str

        schema = _json_schema(annotation)

        if name in descriptions:
            schema["description"] = descriptions[name]

        if parameter.default is not inspect.Parameter.empty:
            if parameter.default is not None:
                schema["default"] = parameter.default
        else:
            required.append(name)

        properties[name] = schema

    # Ensure there is always a description.
    doc_text = inspect.cleandoc(func.__doc__ or "")
    description_part = re.split(
        r"(?i)\n(?:parameters|args|arguments|returns|yields|raises)"
        r"\s*(?:\n[-=]+)?\s*\n",
        doc_text,
    )[0]
    description = " ".join(description_part.split())

    if not description:
        description = f"Executes the {func.__name__} function."

    LOCAL_TOOLS[func.__name__] = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return func


@local_tool
async def get_current_time(timezone: str | None = None) -> str:
    """
    Get the current date and time in the user's timezone. Always use this when the query involves now, today, tomorrow, current time, scheduling, deadlines, recency, etc.

    PARAMETERS
    ----------
    timezone (str | None): Optional IANA timezone name e.g. 'America/Anchorage'. Defaults to the user's known timezone.

    RETURNS
    -------
    Formatted string like: "2026-03-17 14:55 AKDT"
    """
    # Default to system local time
    tz = datetime.now().astimezone().tzinfo
    if timezone:
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            pass

    now = datetime.now(tz)

    # Most readable format for agents & humans
    return now.strftime("%Y-%m-%d %-I:%M:%S %p %Z")

@local_tool
async def local_entity_state(entity_ids: list[str]) -> str:
    """Retrieve the current state and key attributes of one or more entities. Use this tool whenever a question involves the current status, value, location, health, on/off state, brightness, temperature, position, or any real-time property of an entity. Prefer fetching multiple relevant entities in a single call. NOTE: This tool name DOES NOT have a prefix. Call it exactly as 'local_entity_state'

    PARAMETERS
    ----------
    entity_ids (list[str]): List of entity IDs to fetch (e.g. ['light.kitchen_ceiling', 'sensor.backyard_temperature']). Use a list even for one entity.

    RETURNS
    -------
    A compact JSON object containing the current state of each entity.
    """
    from . import cache_utils

    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]

    result:dict[str,Any] = {}
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


@local_tool
async def get_current_weather(
        latitude: float | None = None,
        longitude: float | None = None,
        *,
        location: str | None = None,
        timeframe: Literal["today", "tomorrow", "this weekend", "7 day"] = "today",
):
    """Retrieves real-time weather forecasts from the NWS for a specific sector. Use for current, daily, or weekend forecasts. Provide either latitude and longitude together, or location. Never provide only one coordinate or combine coordinates with location. Prefer latitude and longitude when known.

    Provide either latitude and longitude together, or location. Never provide
    only one coordinate or combine coordinates with location. Prefer latitude and
    longitude when they are known because coordinates provide the most precise
    forecast. Do not provide a partial coordinate pair or both coordinate and
    location inputs.

    PARAMETERS
    ----------
    latitude (float | None): Decimal latitude (e.g., 64.837). Use together with longitude when known.
    longitude (float | None): Decimal longitude (e.g., -147.716). Use together with latitude when known.
    location (str | None): City/State name for non-local scans. Use when coordinates are unavailable.
    timeframe (Literal): The forecast period to retrieve. Defaults to 'today'.

    RETURNS
    -------
    A JSON-encoded weather forecast.
    """
    headers = {"User-Agent": "Starfleet-Command-AVO-Assistant/1.0 (israel@MacStudio)"}
    now = datetime.now() # Mar 19, 2026 (Thursday)

    async with httpx.AsyncClient(timeout=20.0) as client:

        # 1. Sensor Selection
        if latitude is not None or longitude is not None:
            if latitude is None or longitude is None:
                return "Both latitude and longitude are required when using coordinates."
            # Use direct coordinates (Highest Precision / Lowest Latency)
            lat, lon = latitude, longitude
        elif location:
            # 1. Geocode
            geo_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
            geo_res = await client.get(geo_url, headers=headers)
            if not geo_res.json(): return "Sector not found."
            lat, lon = geo_res.json()[0]["lat"], geo_res.json()[0]["lon"]
        else:
            return "Provide either both latitude and longitude or a location."

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

@local_tool
async def get_current_playback() -> str:
    """Retrieves the current playback state from Apple Music, including whether something is playing/paused/stopped, the current track details (title, artist, album), playback position, and duration. Use this when the user asks 'what's playing', 'what song is this', or needs context before controlling playback.

    PARAMETERS
    ----------
    None.

    RETURNS
    -------
    The current playback state as a JSON object.
    """
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                f"{ITUNES_URL.rstrip('/')}/now_playing"
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
    Internal helper to set shuffle mode via PUT /shuffle. The body must be 'mode=songs' to enable or 'mode=off' to disable.

    PARAMETERS
    ----------
    enable (bool): Whether shuffle mode should be enabled.

    RETURNS
    -------
    None.
    """
    mode = "songs" if enable else "off"
    async with httpx.AsyncClient() as client:
        await client.put(f"{ITUNES_URL}/shuffle", data={"mode": mode})

@local_tool
async def pause_music():
    """
    Pauses the music. You MUST call this every time a user asks to pause or stop the music

    PARAMETERS
    ----------
    None.

    RETURNS
    -------
    The server response.
    """
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/pause")
        response.raise_for_status()
        return response.text

@local_tool
async def list_playlists():
    """
    Retrieves all playlists from the library. Call this tool first to find the 'id' of a playlist when the user refers to one by name or to answer questions about available playlists. Returns a list of playlist objects with 'id' and 'name' properties.

    PARAMETERS
    ----------
    None.

    RETURNS
    -------
    A list of playlist objects.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ITUNES_URL}/playlists")
        response.raise_for_status()
        return response.text

async def list_albums(offset: int = 0, limit: int = 100):
    """
    Retrieves all albums from the library.

    PARAMETERS
    ----------
    offset (int): The offset to use when listing albums.
    limit (int): The number of albums to return in the result set.

    RETURNS
    -------
    A list of album objects.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ITUNES_URL}/library/albums", params={"offset": offset, 'limit': limit,})
        response.raise_for_status()
        return response.text

@local_tool
async def search_music(query: str, limit: int = 10):
    """Searches the library for albums, artists, or specific tracks. Use this as the
    primary tool for finding music when the exact title is unknown or to verify metadata.
    IMPORTANT:Search using exactly one criterion: a track name, album name, or artist name.
    Do not combine criteria in a single query. If the user provides multiple pieces
    of information, search using only the track name.

    PARAMETERS
    ----------
    query (str): The search term (e.g., 'Dark Side', 'Pink Floyd', or 'Wish You Were Here').
    limit (int): Max results to return.

    RETURNS
    -------
    Matching albums, artists, and tracks.
    """
    results = {
        'albums': [],
        'artists': [],
        'tracks': [],
    }

    if cache_utils.itunes_cache.get('albums'):
        album_matches = process.extract(
            query,
            cache_utils.itunes_cache['albums'].keys(),
            scorer=fuzz.WRatio,
            processor=utils.default_process,
            limit=limit,
        )
        for title, score, _ in album_matches:
            if score > 55:
                item = dict(cache_utils.itunes_cache['albums'][title])
                item['confidence'] = round(score)
                results['albums'].append(item)

    if cache_utils.itunes_cache.get('artists'):
        artist_matches = process.extract(
            query,
            cache_utils.itunes_cache['artists'].keys(),
            scorer=fuzz.WRatio,
            processor=utils.default_process,
            limit=limit,
        )
        for name, score, _ in artist_matches:
            if score > 55:
                item = dict(cache_utils.itunes_cache['artists'][name])
                item['confidence'] = round(score)
                results['artists'].append(item)

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "http://localhost:8181/library/search",
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


@local_tool
async def play_track(track_id: str):
    """
    Plays a single track by its persistent ID. If the ID is unknown, call search_music first.

    PARAMETERS
    ----------
    track_id (str): The persistent ID of the track.

    RETURNS
    -------
    The server response.
    """
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/library/tracks/{track_id}/play")
        response.raise_for_status()
        return response.text

@local_tool
async def play_playlist(playlist_id: str, shuffle: bool = False):
    """
    Plays a playlist by ID. Use list_playlists first to resolve a name to an ID. Support optional shuffling.

    PARAMETERS
    ----------
    playlist_id (str): The unique ID of the playlist.
    shuffle (bool): Whether to shuffle the playlist.

    RETURNS
    -------
    The server response.
    """
    await _set_shuffle(shuffle)
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/playlists/{playlist_id}/play")
        response.raise_for_status()
        return response.text

@local_tool
async def play_album(artist_name: str, album_name: str, shuffle: bool = False):
    """
    Plays a specific album. Requires exact artist and album names, as returned by search_music. Use search_music first to get the exact names.

    PARAMETERS
    ----------
    artist_name (str): The name of the artist.
    album_name (str): The title of the album.
    shuffle (bool): Whether to shuffle the album tracks.

    RETURNS
    -------
    The server response.
    """
    await _set_shuffle(shuffle)
    # quote() ensures spaces/special characters are safe for the URI path
    safe_artist = quote(artist_name, safe='')
    safe_album = quote(album_name, safe='')
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/library/albums/{safe_artist}/{safe_album}/play")
        response.raise_for_status()
        return response.text

@local_tool
async def play_artist(artist_name: str, shuffle: bool = True):
    """
    Plays all tracks by a specific artist. Defaults to shuffle mode.

    PARAMETERS
    ----------
    artist_name (str): The name of the artist.
    shuffle (bool): Whether to shuffle the album tracks.

    RETURNS
    -------
    The server response.
    """
    await _set_shuffle(shuffle)
    safe_artist = quote(artist_name, safe='')
    async with httpx.AsyncClient() as client:
        response = await client.put(f"{ITUNES_URL}/library/artists/{safe_artist}/play")
        response.raise_for_status()
        return response.text
