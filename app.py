"""
Agentic AI-Based Travel Planning Assistant (LangChain + Streamlit)
===================================================================

Run locally :  streamlit run app.py
Deploy      :  push this folder to GitHub -> https://share.streamlit.io -> select app.py

Structure of this file
----------------------
1. Configuration & data loading   (JSON datasets, flexible key detection)
2. Core business logic            (flights, hotels, places, weather, budget)
3. LangChain tools                (thin wrappers around the core logic)
4. Agent construction             (tool-calling / ReAct agent via LangChain + LangGraph)
5. Offline planner                (deterministic pipeline, works without an LLM key)
6. Streamlit user interface
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from langchain_core.tools import tool

# ----------------------------------------------------------------------------
# 1. CONFIGURATION & DATA LOADING
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CURRENCY = "₹"
APP_VERSION = "2.6 (fixed None-total crash in budget rendering)"

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 71: "Light snow", 73: "Snow",
    75: "Heavy snow", 80: "Rain showers", 81: "Rain showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm",
}

CITY_ALIASES = {
    "new delhi": "delhi", "bangalore": "bengaluru", "bombay": "mumbai",
    "madras": "chennai", "panaji": "goa", "panjim": "goa",
}


def _money(value: Any) -> str:
    """Format a number as ₹1,234, or 'N/A' when the value is None."""
    return f"{CURRENCY}{value:,.0f}" if isinstance(value, (int, float)) else "N/A"


def _first(record: dict, keys: list[str], default: Any = None) -> Any:
    """Return the first non-empty value in `record` matching any alias in `keys`."""
    lowered = {str(k).strip().lower().replace(" ", "_"): v for k, v in record.items()}
    for key in keys:
        if lowered.get(key) not in (None, ""):
            return lowered[key]
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert numbers or strings such as '₹4,800' to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group())
    return default


def _to_minutes(value: Any) -> int:
    """Convert '2h 30m', '2:30', 150 or 2.5 (hours) to minutes. Returns 0 if unknown."""
    if isinstance(value, (int, float)):
        return int(value * 60) if value <= 24 else int(value)
    if isinstance(value, str):
        hours = re.search(r"(\d+)\s*h", value, re.I)
        mins = re.search(r"(\d+)\s*m", value, re.I)
        if hours or mins:
            return (int(hours.group(1)) if hours else 0) * 60 + (int(mins.group(1)) if mins else 0)
        if ":" in value:
            parts = value.split(":")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]) * 60 + int(parts[1])
        return _to_minutes(_to_float(value))
    return 0


def _norm_city(name: Any) -> str:
    """Normalise a city name for matching."""
    text = str(name or "").strip().lower()
    return CITY_ALIASES.get(text, text)


def _city_match(query_norm: str, candidate: Any) -> bool:
    """True if the normalised query city matches the candidate city."""
    cand = _norm_city(candidate)
    if not query_norm or not cand:
        return False
    return query_norm == cand or query_norm in cand or cand in query_norm


def _load_records(path: Path) -> list[dict]:
    """Load a JSON file and return a list of records (handles wrapped dicts)."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, list):
                return value
        return [raw]
    return raw


def _normalise_flight(rec: dict, idx: int) -> dict:
    departure = str(
        _first(rec, ["departure_time", "departure", "dep_time", "depart"], "")
    )
    arrival = str(
        _first(rec, ["arrival_time", "arrival", "arr_time"], "")
    )

    # Read duration from the dataset if available.
    duration = _to_minutes(
        _first(rec, ["duration", "duration_minutes", "duration_mins",
                     "flight_duration"], 0)
    )

    # Calculate duration from timestamps when missing or invalid.
    if duration <= 0 and departure and arrival:
        try:
            dep_dt = datetime.fromisoformat(departure)
            arr_dt = datetime.fromisoformat(arrival)

            duration = int((arr_dt - dep_dt).total_seconds() / 60)

            # Handle flights arriving after midnight.
            if duration < 0:
                duration += 24 * 60

        except (ValueError, TypeError):
            duration = 0

    return {
        "id": str(_first(rec, ["id", "flight_id", "flight_no",
                               "flight_number"], idx)),
        "airline": str(_first(rec, ["airline", "carrier",
                                    "airline_name", "name"], "Unknown")),
        "source": str(_first(rec, ["source", "from", "origin",
                                   "departure_city", "from_city"], "")),
        "destination": str(_first(rec, ["destination", "to", "dest",
                                        "arrival_city", "to_city"], "")),
        "price": _to_float(_first(rec, ["price", "fare", "cost",
                                        "price_inr"], 0)),
        "departure_time": departure,
        "arrival_time": arrival,
        "duration_minutes": duration,
        "stops": _first(rec, ["stops", "total_stops"], 0),
        "date": str(_first(rec, ["date", "departure_date",
                                 "flight_date"], "")),
    }


def _normalise_hotel(rec: dict, idx: int) -> dict:
    rating = _to_float(_first(rec, ["rating", "review_score", "user_rating", "stars"], 0))
    return {
        "id": str(_first(rec, ["id", "hotel_id"], idx)),
        "name": str(_first(rec, ["name", "hotel_name", "hotel"], "Unknown Hotel")),
        "city": str(_first(rec, ["city", "location", "destination"], "")),
        "rating": rating,
        "stars": _to_float(_first(rec, ["stars", "star_rating", "category"], 0)),
        "price_per_night": _to_float(_first(rec, ["price_per_night", "price", "rate", "cost_per_night"], 0)),
        "amenities": _first(rec, ["amenities", "facilities"], []),
    }


def _normalise_place(rec: dict, idx: int) -> dict:
    return {
        "id": str(_first(rec, ["id", "place_id"], idx)),
        "name": str(_first(rec, ["name", "place", "place_name", "title"], "Unknown Place")),
        "city": str(_first(rec, ["city", "location", "destination"], "")),
        "category": str(_first(rec, ["type", "category", "kind", "tags"], "general")),
        "rating": _to_float(_first(rec, ["rating", "review_score", "user_rating"], 0)),
        "description": str(_first(rec, ["description", "about", "summary"], "")),
        "latitude": _first(rec, ["latitude", "lat"], None),
        "longitude": _first(rec, ["longitude", "lon", "lng"], None),
    }


def _sample_dataset() -> dict[str, list[dict]]:
    """Small built-in dataset used only when data/*.json files are absent."""
    cities = ["Delhi", "Mumbai", "Bengaluru", "Goa", "Jaipur", "Chennai"]
    airlines = [("IndiGo", 1.0), ("Air India", 1.25), ("SpiceJet", 0.9)]
    flights = []
    for i, src in enumerate(cities):
        for j, dst in enumerate(cities):
            if i == j:
                continue
            base = 3200 + 450 * ((i * 3 + j * 5) % 7)
            duration = 80 + 20 * ((i + j) % 6)
            for k, (name, mult) in enumerate(airlines):
                dep_hour = 6 + (i + j + k * 4) % 14
                flights.append({
                    "airline": name, "source": src, "destination": dst,
                    "price": round(base * mult, -1), "departure_time": f"{dep_hour:02d}:00",
                    "arrival_time": f"{(dep_hour + duration // 60) % 24:02d}:{duration % 60:02d}",
                    "duration_minutes": duration + k * 10,
                })
    hotels_raw = [
        ("Goa", "Sea View Resort", 4, 4.4, 3200), ("Goa", "Palm Grove Inn", 3, 4.0, 1900),
        ("Goa", "Candolim Luxe", 5, 4.7, 7800), ("Jaipur", "Pink City Haveli", 4, 4.5, 3500),
        ("Jaipur", "Amber Budget Stay", 3, 3.9, 1500), ("Jaipur", "Rajmahal Palace", 5, 4.8, 9000),
        ("Mumbai", "Marine Drive Hotel", 4, 4.3, 5200), ("Mumbai", "Bandra Comfort", 3, 4.0, 2800),
        ("Mumbai", "Gateway Grand", 5, 4.6, 11000), ("Delhi", "Connaught Residency", 4, 4.2, 4100),
        ("Delhi", "Old Delhi Lodge", 3, 3.8, 1700), ("Delhi", "Imperial Suites", 5, 4.7, 10500),
        ("Bengaluru", "Garden City Inn", 4, 4.3, 3600), ("Bengaluru", "Tech Park Stay", 3, 4.0, 2300),
        ("Bengaluru", "Lalbagh Grand", 5, 4.6, 8200), ("Chennai", "Marina Bay Hotel", 4, 4.2, 3300),
        ("Chennai", "Temple Town Lodge", 3, 3.9, 1800), ("Chennai", "Coromandel Palace", 5, 4.6, 7600),
    ]
    places_raw = [
        ("Goa", "Baga Beach", "beach", 4.5, "Lively beach with water sports and shacks"),
        ("Goa", "Candolim Market", "shopping", 4.1, "Popular market for souvenirs"),
        ("Goa", "Basilica of Bom Jesus", "heritage", 4.6, "UNESCO-listed baroque church"),
        ("Goa", "Old Goa Heritage Walk", "heritage", 4.4, "Walk through Portuguese-era churches"),
        ("Goa", "Calangute Water Sports", "adventure", 4.3, "Jet skiing, parasailing, banana boat"),
        ("Goa", "Fort Aguada", "heritage", 4.5, "17th-century fort with sea views"),
        ("Goa", "Anjuna Flea Market", "shopping", 4.2, "Wednesday flea market"),
        ("Goa", "Dudhsagar Falls", "nature", 4.7, "Four-tiered waterfall"),
        ("Jaipur", "Amber Fort", "heritage", 4.7, "Hilltop fort-palace"),
        ("Jaipur", "City Palace", "heritage", 4.5, "Royal residence and museum"),
        ("Jaipur", "Hawa Mahal", "heritage", 4.4, "Palace of Winds"),
        ("Jaipur", "Johari Bazaar", "shopping", 4.2, "Jewellery and textiles market"),
        ("Jaipur", "Nahargarh Sunset Point", "nature", 4.5, "Panoramic city views"),
        ("Jaipur", "Chokhi Dhani", "food", 4.3, "Rajasthani village-themed dining"),
        ("Mumbai", "Gateway of India", "heritage", 4.5, "Iconic waterfront arch"),
        ("Mumbai", "Marine Drive", "nature", 4.6, "Seaside promenade"),
        ("Mumbai", "Elephanta Caves", "heritage", 4.4, "Rock-cut cave temples"),
        ("Mumbai", "Colaba Causeway", "shopping", 4.1, "Street shopping"),
        ("Mumbai", "Juhu Beach", "food", 4.0, "Beach with street food"),
        ("Mumbai", "Sanjay Gandhi National Park", "nature", 4.3, "Forest park with Kanheri caves"),
        ("Delhi", "Red Fort", "heritage", 4.5, "Mughal fort"),
        ("Delhi", "India Gate", "heritage", 4.6, "War memorial"),
        ("Delhi", "Qutub Minar", "heritage", 4.5, "Tallest brick minaret"),
        ("Delhi", "Chandni Chowk", "food", 4.4, "Historic market and street food"),
        ("Delhi", "Lotus Temple", "culture", 4.5, "Bahai house of worship"),
        ("Delhi", "Dilli Haat", "shopping", 4.2, "Craft bazaar"),
        ("Bengaluru", "Lalbagh Botanical Garden", "nature", 4.5, "Historic botanical garden"),
        ("Bengaluru", "Bangalore Palace", "heritage", 4.3, "Tudor-style palace"),
        ("Bengaluru", "Cubbon Park", "nature", 4.4, "Central green space"),
        ("Bengaluru", "Commercial Street", "shopping", 4.1, "Busy shopping street"),
        ("Chennai", "Marina Beach", "beach", 4.3, "Long urban beach"),
        ("Chennai", "Kapaleeshwarar Temple", "culture", 4.6, "Dravidian temple"),
        ("Chennai", "Fort St. George", "heritage", 4.2, "First British fort in India"),
        ("Chennai", "Mahabalipuram Shore Temple", "heritage", 4.6, "UNESCO shore temple"),
    ]
    return {
        "flights": [_normalise_flight(r, i) for i, r in enumerate(flights)],
        "hotels": [_normalise_hotel({"city": c, "name": n, "stars": s, "rating": r, "price_per_night": p}, i)
                   for i, (c, n, s, r, p) in enumerate(hotels_raw)],
        "places": [_normalise_place({"city": c, "name": n, "type": t, "rating": r, "description": d}, i)
                   for i, (c, n, t, r, d) in enumerate(places_raw)],
    }


@lru_cache(maxsize=1)
def get_data() -> dict[str, Any]:
    """Load flights/hotels/places from data/*.json (or fall back to built-in samples)."""
    sample = _sample_dataset()
    result: dict[str, Any] = {"using_sample": []}
    spec = {"flights": _normalise_flight, "hotels": _normalise_hotel, "places": _normalise_place}
    for name, normaliser in spec.items():
        path = DATA_DIR / f"{name}.json"
        try:
            records = _load_records(path)
            result[name] = [normaliser(r, i) for i, r in enumerate(records) if isinstance(r, dict)]
            if not result[name]:
                raise ValueError("empty dataset")
        except (OSError, ValueError, json.JSONDecodeError):
            result[name] = sample[name]
            result["using_sample"].append(name)
    return result


# ----------------------------------------------------------------------------
# 2. CORE BUSINESS LOGIC
# ----------------------------------------------------------------------------
def _flight_date(flight: dict) -> date | None:
    """Calendar date of a flight, or None when the dataset only stores a time (e.g. '11:30')."""
    for field in ("date", "departure_time"):
        text = str(flight.get(field, "") or "").strip()
        if not text:
            continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            match = re.search(r"\d{4}-\d{2}-\d{2}", text)
            if match:
                return date.fromisoformat(match.group())
    return None


def _clock(timestamp: Any) -> str:
    """Format '2025-02-08T03:08:00' as '03:08'; return other values (e.g. '17:00') unchanged."""
    text = str(timestamp or "").strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%H:%M")
    except ValueError:
        return text


def find_flights(source: str, destination: str, sort_by: str = "cheapest",
                 start_date: str = "", limit: int = 5, nearest_if_unmatched: bool = False) -> dict:
    """Search flights by route and (optionally) departure date.

    - Flights with a real calendar date must match `start_date` exactly.
    - Flights with only a departure time are treated as a regular daily schedule and
      flagged as 'schedule_based' (fare/availability not live-verified).
    - If nothing matches the date and `nearest_if_unmatched` is True, the flights on the
      closest dataset date are returned as a 'reference_fare' (clearly labelled).
    """
    src, dst = _norm_city(source), _norm_city(destination)
    route = [f for f in get_data()["flights"]
             if _city_match(src, f["source"]) and _city_match(dst, f["destination"])]
    if not route:
        return {"count": 0, "status": "no_route",
                "message": f"No flights found from {source} to {destination} in the dataset."}

    candidates = route
    reference_day = None
    if start_date:
        try:
            target = date.fromisoformat(start_date)
        except ValueError:
            return {"count": 0, "status": "invalid_date", "message": f"Invalid date '{start_date}'."}
        candidates, other_dates = [], set()
        for flight in route:
            flight_day = _flight_date(flight)
            if flight_day is None or flight_day == target:
                candidates.append(flight)
            else:
                other_dates.add(flight_day.isoformat())
        if not candidates:
            if not nearest_if_unmatched:
                return {"count": 0, "status": "date_mismatch", "available_dates": sorted(other_dates),
                        "message": f"The route {source} to {destination} exists, but no flight is dated "
                                   f"{start_date}. Dataset dates: {', '.join(sorted(other_dates))}."}
            dated = [(abs((_flight_date(f) - target).days), f) for f in route if _flight_date(f)]
            closest = min(gap for gap, _ in dated)
            candidates = [f for gap, f in dated if gap == closest]
            reference_day = _flight_date(candidates[0]).isoformat()

    big = 10 ** 6
    if sort_by == "fastest":
        candidates = sorted(candidates, key=lambda f: (f["duration_minutes"] or big, f["price"]))
        reason = "Selected as the fastest available option."
    else:
        candidates = sorted(candidates, key=lambda f: (f["price"], f["duration_minutes"] or big))
        reason = "Selected as the cheapest available option."

    selected = candidates[0]
    if reference_day:
        status = "reference_fare"
        note = (f"No flight is dated {start_date}; showing the closest dataset flight (dated {reference_day}) "
                "as a REFERENCE FARE. The real fare on your date may differ.")
    elif _flight_date(selected) is not None:
        status, note = "date_confirmed", ""
    else:
        status = "schedule_based"
        note = ("Dataset has departure times only (no calendar dates), so this is treated as a regular "
                "daily schedule. Fare and seat availability are not live-verified.")
    return {"count": len(candidates), "status": status, "sort_by": sort_by, "selected": selected,
            "alternatives": candidates[1:limit], "reason": reason, "note": note}


def find_hotels(city: str, min_rating: float = 0, max_price_per_night: float = 0, limit: int = 3) -> dict:
    """Filter hotels by city/rating/price and rank by rating (desc) then price (asc)."""
    target = _norm_city(city)
    hotels = [h for h in get_data()["hotels"] if _city_match(target, h["city"])]
    filtered = [h for h in hotels if h["rating"] >= min_rating
                and (max_price_per_night <= 0 or h["price_per_night"] <= max_price_per_night)]
    if not filtered:
        return {"count": 0, "message": f"No hotels in {city} match rating>={min_rating} and "
                                       f"price<={max_price_per_night or 'any'}."}
    filtered.sort(key=lambda h: (-h["rating"], h["price_per_night"]))
    return {"count": len(filtered), "selected": filtered[0], "alternatives": filtered[1:limit],
            "reason": f"Highest rated ({filtered[0]['rating']}) among {len(filtered)} matching hotels."}



def find_places(
    city: str,
    category: str = "",
    min_rating: float = 0,
    limit: int = 10
) -> dict:
    """Discover attractions in a city, optionally filtered by category."""

    target = _norm_city(city)

    # Get places matching the city and minimum rating.
    places = [
        p for p in get_data()["places"]
        if _city_match(target, p["city"])
        and p["rating"] >= min_rating
    ]

    # Apply category filtering strictly.
    if category:
        wanted = [
            c.strip().lower()
            for c in re.split(r"[,/|]", category)
            if c.strip()
        ]

        places = [
            p for p in places
            if any(
                word == p["category"].strip().lower()
                for word in wanted
            )
        ]

    if not places:
        return {
            "count": 0,
            "places": [],
            "message": (
                f"No places found in {city}"
                + (f" matching category '{category}'." if category else ".")
            )
        }

    # Sort by rating, highest first.
    places.sort(key=lambda p: -p["rating"])

    # Remove duplicate names while keeping the highest-rated record.
    unique = []
    seen = set()

    for place in places:
        key = place["name"].strip().lower()

        if key not in seen:
            seen.add(key)
            unique.append(place)

    return {
        "count": len(unique[:limit]),
        "places": unique[:limit]
    }



def _shift_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:  # 29 Feb
        return day.replace(year=day.year - years, day=28)


def _request_daily(url: str, lat: float, lon: float, start: date, end: date) -> dict:
    response = requests.get(url, params={
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
    }, timeout=15)
    response.raise_for_status()
    return response.json()["daily"]


def fetch_weather(city: str, start_date: str, end_date: str) -> dict:
    """Day-wise weather for a city via Open-Meteo (forecast, or historical estimate for far dates)."""
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if end < start:
            return {"error": "end_date is before start_date."}
        geo = requests.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=10).json()
        if not geo.get("results"):
            return {"error": f"Could not geocode '{city}'."}
        place = geo["results"][0]
        lat, lon = place["latitude"], place["longitude"]

        today = date.today()
        if start >= today and end <= today + timedelta(days=15):
            url, q_start, q_end, source = FORECAST_URL, start, end, "live forecast"
        elif end <= today - timedelta(days=7):
            url, q_start, q_end, source = ARCHIVE_URL, start, end, "historical observation"
        else:
            years = 1
            while _shift_years(end, years) > today - timedelta(days=7):
                years += 1
            q_start, q_end = _shift_years(start, years), _shift_years(end, years)
            url, source = ARCHIVE_URL, f"historical estimate (same dates in {q_start.year})"

        daily = _request_daily(url, lat, lon, q_start, q_end)
        days = []
        for i, code in enumerate(daily["weather_code"]):
            days.append({
                "date": (start + timedelta(days=i)).isoformat(),
                "condition": WMO_CODES.get(int(code) if code is not None else -1, "Unknown"),
                "temp_max_c": daily["temperature_2m_max"][i],
                "temp_min_c": daily["temperature_2m_min"][i],
                "rain_mm": daily["precipitation_sum"][i],
            })
        return {"city": place.get("name", city), "source": source, "days": days}
    except (requests.RequestException, KeyError, ValueError) as exc:
        return {"error": f"Weather lookup failed ({type(exc).__name__})"}


def compute_budget(
    flight_price_per_person: float | None,
    hotel_price_per_night: float,
    nights: int,
    travelers: int = 1,
    daily_local_expense: float = 1500,
    round_trip: bool = False,
    return_flight_price_per_person: float | None = None,
    flight_is_reference: bool = False
) -> dict:
    """
    Estimate trip costs without inventing missing flight fares.

    A round trip is complete only when both outbound and return fares exist.
    Missing return fares are never replaced with the outbound fare.
    """

    nights = max(int(nights), 0)
    travelers = max(int(travelers), 1)
    rooms = math.ceil(travelers / 2)

    # Calculate known airfare only
    if flight_price_per_person is None:
        flight_total = None
        basis = "unavailable"
    else:
        per_person = flight_price_per_person

        if round_trip and return_flight_price_per_person is not None:
            per_person += return_flight_price_per_person

        flight_total = round(per_person * travelers)
        basis = "reference" if flight_is_reference else "actual"

    # A round trip is incomplete if its return fare is missing
    return_missing = round_trip and return_flight_price_per_person is None

    hotel_total = hotel_price_per_night * nights * rooms
    local_total = daily_local_expense * (nights + 1) * travelers

    # Existing non-flight subtotal
    known_subtotal = round(hotel_total + local_total)

    # Known costs, including outbound airfare if available
    known_trip_subtotal = (
        known_subtotal + flight_total
        if flight_total is not None
        else known_subtotal
    )

    # A complete total is available only when the required airfare is known
    total_is_incomplete = flight_total is None or return_missing

    grand_total = (
        round(known_trip_subtotal)
        if not total_is_incomplete
        else None
    )

    return {
        "flight_basis": basis,
        "flight_actual": flight_total if basis == "actual" else None,
        "flight_reference": flight_total if basis == "reference" else None,
        "flight": flight_total,
        "hotel": round(hotel_total),
        "food_and_local_travel": round(local_total),
        "known_subtotal": known_subtotal,
        "known_trip_subtotal": round(known_trip_subtotal),
        "return_fare_missing": return_missing,
        "total_is_incomplete": total_is_incomplete,
        "confirmed_total": (
            grand_total if basis == "actual" and not total_is_incomplete else None
        ),
        "estimated_total": (
            grand_total if basis == "reference" and not total_is_incomplete else None
        ),
        "total": grand_total,
        "total_is_estimate": basis == "reference",
        "assumptions": (
            f"{travelers} traveler(s), {rooms} room(s), {nights} night(s), "
            f"{'round-trip' if round_trip else 'one-way'} flights, "
            f"{CURRENCY}{daily_local_expense:,.0f}/person/day local expenses"
        ),
    }


# ----------------------------------------------------------------------------
# 3. LANGCHAIN TOOLS
# ----------------------------------------------------------------------------
@tool
def flight_search_tool(source: str, destination: str, sort_by: str = "cheapest", start_date: str = "",
                       use_nearest_date_if_unavailable: bool = False) -> str:
    """Search flights between two cities. sort_by is 'cheapest' or 'fastest'.
    Pass start_date (YYYY-MM-DD) to check the requested departure date. The result has a 'status':
    date_confirmed, schedule_based (daily schedule, fare not live-verified), reference_fare
    (closest dataset date, only when use_nearest_date_if_unavailable=true), date_mismatch or no_route."""
    return json.dumps(find_flights(source.strip(), destination.strip(), sort_by, start_date=start_date,
                                   nearest_if_unmatched=use_nearest_date_if_unavailable), ensure_ascii=False)


@tool
def hotel_recommendation_tool(city: str, min_rating: float = 0, max_price_per_night: float = 0) -> str:
    """Recommend hotels in a city. Use min_rating (0-5) and max_price_per_night (0 = no limit).
    Returns the best-rated hotel plus alternatives."""
    return json.dumps(find_hotels(city.strip(), min_rating, max_price_per_night), ensure_ascii=False)


@tool
def places_discovery_tool(city: str, category: str = "", min_rating: float = 0) -> str:
    """Discover tourist attractions in a city. Optional category keywords (comma separated,
    e.g. 'beach, heritage') and minimum rating. Returns places ranked by rating."""
    return json.dumps(find_places(city.strip(), category, min_rating), ensure_ascii=False)


@tool
def weather_lookup_tool(city: str, start_date: str, end_date: str) -> str:
    """Get day-wise weather for a city. Dates must be ISO format YYYY-MM-DD."""
    return json.dumps(fetch_weather(city.strip(), start_date, end_date), ensure_ascii=False)


@tool
def budget_estimation_tool(
    flight_price_per_person: float | None,
    hotel_price_per_night: float | None,
    nights: int,
    travelers: int = 1,
    daily_local_expense: float = 1500,
    round_trip: bool = False,
    return_flight_price_per_person: float | None = None,
    flight_is_reference: bool = False
) -> str:
    """
    Estimate trip costs.

    Pass 0 for hotel_price_per_night when no hotel is available.
    Pass null for flight fares only when unavailable.
    Set flight_is_reference=true for unconfirmed or historical fares.
    For round trips, pass the return fare only when it is actually available.
    """

    # Treat an unavailable hotel price as zero.
    hotel_price = (
        0.0 if hotel_price_per_night: float,
        else float(hotel_price_per_night)
    )

    result = compute_budget(
        flight_price_per_person,
        hotel_price,
        nights,
        travelers,
        daily_local_expense,
        round_trip,
        return_flight_price_per_person,
        flight_is_reference
    )

    return json.dumps(result, ensure_ascii=False)


TOOLS = [flight_search_tool, hotel_recommendation_tool, places_discovery_tool,
         weather_lookup_tool, budget_estimation_tool]

# ----------------------------------------------------------------------------
# 4. AGENT CONSTRUCTION
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert AI Travel Planning Agent. Today's date is __TODAY__.
All prices are in Indian Rupees (₹).

WORKFLOW (always follow it, calling tools - never invent data):

1. Understand the request:
   - Origin and destination
   - Start and return dates
   - Number of travelers
   - Budget
   - Interests
   - Whether round-trip flights are enabled

   Treat the user's selected trip settings as authoritative.
   Never change a selected round-trip request into a one-way request.

   Number of days = end_date - start_date + 1.
   Nights = days - 1.

2. FLIGHT SEARCH:
   Call flight_search_tool with source, destination, sort_by and start_date (YYYY-MM-DD).

   For ROUND-TRIP requests:
   - ALWAYS search destination -> origin.
   - Use the end_date as the return flight date.
   - Search the return flight independently.
   - If the return result is date_mismatch, retry with
     use_nearest_date_if_unavailable=true.
   - If the return result is no_route, report that no return route exists.
   - Never invent a return flight or return fare.

   For date_mismatch, retry the relevant search with
   use_nearest_date_if_unavailable=true.
   Clearly label a different-date result as a REFERENCE FARE.

   If status is schedule_based, say the fare/date comes from a
   daily schedule and is not live-verified.

   If no flight is found, report it honestly.

3. HOTEL SEARCH:
   Always call hotel_recommendation_tool independently of flight results.
   Respect the user's minimum rating and maximum price limits.
   Never invent hotel availability or prices.

4. PLACES:
   Call places_discovery_tool using the user's interests as category keywords.
   If no results are found, retry without a category.
   Never claim generic attractions match the selected interest.

5. WEATHER:
   Call weather_lookup_tool for the destination and travel dates.

6. BUDGET:
   Call budget_estimation_tool using the selected flight and hotel prices.

   IMPORTANT ROUND-TRIP RULES:
   - If the user selected round-trip, ALWAYS pass round_trip=true.
   - Never pass round_trip=false when the user selected round-trip.
   - Pass travelers exactly as specified by the user.
   - Pass return_flight_price_per_person when a return fare is found.
   - If no return flight is found, pass return_flight_price_per_person=null.
   - Never treat a missing return flight as free.
   - If either flight leg is unavailable, do not present the result
     as a complete round-trip cost.
   - Clearly label any amount excluding a missing flight leg as
     a known subtotal or incomplete estimate.
   - For reference_fare or schedule_based flights, pass
     flight_is_reference=true.

   For one-way requests, pass round_trip=false.
When calling budget_estimation_tool, always pass a numeric value
for hotel_price_per_night. If no hotel is available or the hotel
price is unknown, pass 0. Never pass null for hotel_price_per_night.
7. MISSING DATA:
   If a tool returns no results, report that honestly and continue
   with the available information.
   Never treat missing prices as zero.
   Never claim the budget is respected when a cost is missing.
   Never present an incomplete amount as a complete trip total.

8. BUDGET LIMIT:
   Always deliver the available flight, hotel, weather, itinerary
   and budget information, even if the budget is exceeded.
   Add a short Budget note explaining the overage.
   Suggest cheaper options only by calling the tools again and
   recalculating the budget.

9. ACCURACY:
   Use only facts returned by the tools.
   Never mention trains, buses, other cities or outside information.
   Never perform your own cost arithmetic.
   Copy budget values from budget_estimation_tool.
   For reference_fare flights, show the actual dataset date and
   explain that it differs from the requested date.
   Never present a reference fare as a confirmed booking.
   Only give reasons supported by tool results or the user's request.

10. INTERESTS AND ITINERARY:
    Give every day 2-3 different attractions from places_discovery_tool.
    Never repeat an attraction.
    If insufficient distinct places exist, give fewer and explain why.
    Never pad the itinerary with invented attractions.
    Do not claim the itinerary matches an interest when no matching
    places were returned.

FINAL ANSWER FORMAT (Markdown):

# Your <N>-Day Trip to <Destination> (<start> - <end>)

**Flight Selected:** airline, fare type, price and departure/arrival.
For round trips, show outbound and return separately.
Clearly report unavailable return flights.

**Hotel Recommended:** name, price/night and rating.

**Weather:** one line per day.

**Itinerary:** Day 1..N with available attractions.

**Budget:** distinguish actual, reference and unavailable fares.
For an incomplete round trip, show the known subtotal and clearly
state that the complete round-trip total is unavailable.

**Why we selected this:** 2-4 short bullets based only on tool results.

After the Markdown, output ONE fenced ```json block with exactly
this shape:

{
  "trip_summary": {
    "origin": "",
    "destination": "",
    "start_date": "",
    "end_date": "",
    "travelers": 1
  },
  "flight": {
    "airline": "",
    "price": 0,
    "departure_time": "",
    "arrival_time": ""
  },
  "hotel": {
    "name": "",
    "price_per_night": 0,
    "rating": 0
  },
  "weather": [
    {
      "date": "",
      "condition": "",
      "temp_max_c": 0
    }
  ],
  "itinerary": [
    {
      "day": 1,
      "date": "",
      "activities": [""]
    }
  ],
  "budget": {
    "flight": null,
    "hotel": 0,
    "food_and_local_travel": 0,
    "total": null
  },
  "reasoning": ""
}

Use null for unavailable flight or total values.
Never use zero to represent missing prices.
"""


def get_secret(name: str) -> str:
    """Read a key from Streamlit secrets or environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml present
        pass
    return os.environ.get(name, "")


def build_llm(provider: str, model: str, api_key: str, temperature: float = 0.2):
    """Create a chat model for the chosen provider (imports are lazy)."""
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)
    if provider == "Groq (free tier)":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, temperature=temperature)
    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider}")


def build_agent(llm):
    """Build a tool-calling ReAct-style agent (LangChain 1.x, with LangGraph fallback)."""
    prompt = SYSTEM_PROMPT.replace("__TODAY__", date.today().isoformat())
    try:
        from langchain.agents import create_agent
        return create_agent(llm, TOOLS, system_prompt=prompt)
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(llm, TOOLS, prompt=prompt)


def _message_text(content: Any) -> str:
    """Flatten message content (some providers return a list of parts)."""
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def run_agent(agent, query: str) -> tuple[str, list[dict]]:
    """Invoke the agent; return (final_text, reasoning_steps)."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]},
                          config={"recursion_limit": 40})
    steps: list[dict] = []
    for msg in result["messages"]:
        msg_type = getattr(msg, "type", "")
        if msg_type == "ai" and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                steps.append({"kind": "call", "tool": call["name"], "payload": call["args"]})
        elif msg_type == "tool":
            steps.append({"kind": "result", "tool": getattr(msg, "name", "tool"),
                          "payload": _message_text(msg.content)})
    return _message_text(result["messages"][-1].content), steps


def split_markdown_and_json(text: str) -> tuple[str, dict | None]:
    """Separate the human-readable part from the trailing ```json block."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if not match:
        return text, None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return text, None
    return (text[:match.start()] + text[match.end():]).strip(), data


# ----------------------------------------------------------------------------
# 5. OFFLINE (NO-LLM) PLANNER - same tools, deterministic orchestration
# ----------------------------------------------------------------------------


def build_offline_plan(source: str, destination: str, start: date, end: date, travelers: int,
                       interests: list[str], flight_pref: str, min_rating: float,
                       round_trip: bool, daily_expense: float, max_hotel_price: float = 0,
                       use_nearest_date: bool = True) -> dict:
    """Run the tools in a fixed order and assemble the structured itinerary."""

    days_count = (end - start).days + 1
    nights = days_count - 1

    flights = find_flights(
        source, destination, flight_pref,
        start_date=start.isoformat(),
        nearest_if_unmatched=use_nearest_date
    )
    flight = flights.get("selected")

    return_flights = (
        find_flights(
            destination, source, flight_pref,
            start_date=end.isoformat(),
            nearest_if_unmatched=use_nearest_date
        ) if round_trip else None
    )
    return_flight = return_flights.get("selected") if return_flights else None

    hotels = find_hotels(destination, min_rating, max_hotel_price)

    places = (
        find_places(destination, ", ".join(interests))
        if interests
        else find_places(destination)
    )

    # Add general attractions only when no specific interest is selected.
    if not interests and places.get("count", 0) < days_count * 2:
        extra = find_places(destination, limit=days_count * 3)

        seen = {
            p["name"].strip().lower()
            for p in places.get("places", [])
        }

        merged = places.get("places", []) + [
            p for p in extra.get("places", [])
            if p["name"].strip().lower() not in seen
        ]

        places = {
            "count": len(merged),
            "places": merged
        }

    weather = fetch_weather(destination, start.isoformat(), end.isoformat())
    hotel = hotels.get("selected")

    outbound_price = flight["price"] if flight else None
    return_price = return_flight["price"] if return_flight else None

    legs = [flight] + ([return_flight] if round_trip else [])
    statuses = [flights.get("status")] + (
        [return_flights.get("status")] if round_trip else []
    )

    all_legs_available = all(leg is not None for leg in legs)
    all_confirmed = all(status == "date_confirmed" for status in statuses)

    # Count only available airfare; never invent a missing return fare.
    budget = compute_budget(
        flight_price_per_person=outbound_price,
        hotel_price_per_night=hotel["price_per_night"] if hotel else 0,
        nights=nights,
        travelers=travelers,
        daily_local_expense=daily_expense,
        round_trip=round_trip,
        return_flight_price_per_person=return_price,
        flight_is_reference=not all_confirmed
    )

    if not all_legs_available:
        budget["confirmed_total"] = None
        budget["estimated_total"] = None
        budget["total"] = None
        budget["total_is_incomplete"] = True
        budget["return_fare_missing"] = round_trip and return_flight is None

    picks = places.get("places", [])[:days_count * 3]

    itinerary = [
        {
            "day": d + 1,
            "date": (start + timedelta(days=d)).isoformat(),
            "activities": []
        }
        for d in range(days_count)
    ]

    for i, place in enumerate(picks):
        itinerary[i % days_count]["activities"].append(
            f"{place['name']} ({place['category']}, ★{place['rating']})"
        )

    reasoning = [
        flights.get("reason") or flights.get("message", ""),
        (
            return_flights.get("reason") or return_flights.get("message", "")
        ) if return_flights else "",
        hotels.get("reason") or hotels.get("message", ""),
        (
            "Attractions are ranked by rating and distributed evenly across days."
            if picks else
            "No attractions match the selected interests in this city."
        )
    ]

    return {
        "trip_summary": {
            "origin": source,
            "destination": destination,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "travelers": travelers,
            "round_trip": round_trip
        },
        "flight": flight,
        "flight_status": flights.get("status", ""),
        "flight_note": flights.get("note") or flights.get("message", ""),
        "return_flight": return_flight,
        "return_flight_status": return_flights.get("status", "") if return_flights else "",
        "return_flight_note": (
            return_flights.get("note") or return_flights.get("message", "")
        ) if return_flights else "",
        "hotel": hotel,
        "weather": weather.get("days", []),
        "weather_note": weather.get("error") or weather.get("source"),
        "itinerary": itinerary,
        "budget": budget,
        "reasoning": " ".join(r for r in reasoning if r)
    }



def render_plan_markdown(plan: dict) -> str:
    """Convert the structured plan into a human-readable Markdown itinerary."""
    s, f, rf, h, b = plan["trip_summary"], plan["flight"], plan.get("return_flight"), plan["hotel"], plan["budget"]
    n_days = len(plan["itinerary"])
    lines = [f"# Your {n_days}-Day Trip to {s['destination']} ({s['start_date']} → {s['end_date']})", "",
             "### ✈️ Flights"]

    def flight_line(label: str, flight: dict | None, origin: str, dest: str, note: str, status: str = "") -> None:
        if flight:
            badge = {"date_confirmed": "✅ confirmed for your date",
                     "reference_fare": "⚠️ reference fare (unverified)",
                     "schedule_based": "⚠️ schedule fare (date unverified)"}.get(status, "")
            lines.append(f"- {label}: {flight['airline']} ({_money(flight['price'])}, {badge}) – "
                         f"{origin} → {dest}, departs {_clock(flight['departure_time'])}"
                         + (f", arrives {_clock(flight['arrival_time'])}" if flight.get("arrival_time") else ""))
            if note:
                lines.append(f"  - _{note}_")
        else:
            lines.append(f"- {label}: Unavailable – {note or 'no matching flight in the dataset'}")

    flight_line("Outbound", f, s["origin"], s["destination"], plan.get("flight_note", ""),
                plan.get("flight_status", ""))
    if s.get("round_trip"):
        flight_line("Return", rf, s["destination"], s["origin"], plan.get("return_flight_note", ""),
                    plan.get("return_flight_status", ""))

    lines.append("\n### 🏨 Hotel Recommendation")
    lines.append("_Prices and availability come from the dataset and are not live-verified._")
    lines.append(f"- {h['name']} ({_money(h['price_per_night'])}/night, ★{h['rating']})"
                 if h else "- No hotel matched your filters.")

    lines.append(f"\n### 🌤️ Weather ({plan.get('weather_note') or 'n/a'})")
    if plan["weather"]:
        for i, w in enumerate(plan["weather"], 1):
            tmax = "n/a" if w.get("temp_max_c") is None else f"{w['temp_max_c']}°C"
            tmin = "n/a" if w.get("temp_min_c") is None else f"{w['temp_min_c']}°C"
            rain = "n/a" if w.get("rain_mm") is None else f"{w['rain_mm']} mm"
            lines.append(f"- Day {i}: {w['condition']} ({tmax} / {tmin}, rain {rain})")
    else:
        lines.append("- Weather unavailable.")

    lines.append("\n### 🗺️ Itinerary")
    lines.append("_Attractions come from the dataset; please verify timings and availability._")
    for day in plan["itinerary"]:
        acts = "; ".join(day["activities"]) or "Free day / local exploration"
        lines.append(f"- **Day {day['day']} ({day['date']}):** {acts}")

    lines.append("\n### 💰 Budget Summary")
    basis = b["flight_basis"]
    if basis == "actual":
        lines.append(f"- ✅ **Actual flight fare** (confirmed for your date): {_money(b['flight_actual'])}")
    else:
        lines.append("- ✅ **Actual flight fare:** not available for the requested date")
    if basis == "reference":
        lines.append(f"- ⚠️ **Reference flight fare** (historical dataset fare, unverified): "
                     f"{_money(b['flight_reference'])}")
    elif basis == "unavailable":
        lines.append("- ⚠️ **Reference flight fare:** none found in the dataset")
    lines += [f"- Hotel: {_money(b['hotel'])}",
              f"- Food & local travel: {_money(b['food_and_local_travel'])}"]

    # Check for an incomplete total FIRST, so a None total is never formatted.
    if b.get("total_is_incomplete"):
        missing = "return flight" if b.get("return_fare_missing") else "required flight"
        lines.append(f"\n**Known subtotal (hotel + food + available flights): "
                     f"{_money(b.get('known_trip_subtotal', b['known_subtotal']))}**")
        lines.append(f"\n_Complete trip total unavailable: no {missing} was found in the dataset._")
    elif basis == "actual":
        lines.append(f"\n**Total Trip Cost (confirmed): {_money(b['confirmed_total'])}**")
    else:
        lines.append(f"\n**Estimated Total (includes unverified reference fare): {_money(b['estimated_total'])}**")
        lines.append("\n_A confirmed total is not available because no flight is confirmed for the requested date._")

    lines.append(f"\n_Assumptions: {b['assumptions']}_")
    lines.append(f"\n### 🧠 Why we selected this\n{plan['reasoning']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 6. STREAMLIT USER INTERFACE
# ----------------------------------------------------------------------------
PROVIDER_DEFAULTS = {
    "OpenAI": ("gpt-4o-mini", "OPENAI_API_KEY"),
    "Groq (free tier)": ("openai/gpt-oss-120b", "GROQ_API_KEY"),
    "Google Gemini": ("gemini-2.5-flash", "GOOGLE_API_KEY"),
}


MODEL_LIST_URLS = {
    "Groq (free tier)": "https://api.groq.com/openai/v1/models",
    "OpenAI": "https://api.openai.com/v1/models",
}
NON_CHAT_KEYWORDS = ("whisper", "guard", "tts", "embed", "safeguard", "orpheus", "playai", "moderation",
                     "dall", "image", "audio", "realtime", "transcribe", "compound", "davinci", "babbage")
PREFERRED_MODELS = {
    "Groq (free tier)": ["gpt-oss-120b", "llama-3.3-70b", "llama-4", "gpt-oss-20b", "llama-3.1-8b"],
    "OpenAI": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
}


@st.cache_data(ttl=600, show_spinner=False)
def list_provider_models(provider: str, api_key: str) -> list[str]:
    """Fetch the chat models currently available to this API key (Groq / OpenAI)."""
    response = requests.get(MODEL_LIST_URLS[provider], headers={"Authorization": f"Bearer {api_key}"},
                            timeout=10)
    response.raise_for_status()
    ids = sorted(m["id"] for m in response.json().get("data", []))
    return [i for i in ids if not any(word in i.lower() for word in NON_CHAT_KEYWORDS)]


def choose_model(provider: str, api_key: str, default_model: str) -> str:
    """Show a live model dropdown when possible, else a free-text box."""
    if provider in MODEL_LIST_URLS and api_key:
        try:
            models = list_provider_models(provider, api_key)
            if models:
                index = 0
                for wanted in PREFERRED_MODELS.get(provider, []):
                    match = next((i for i, m in enumerate(models) if wanted in m), None)
                    if match is not None:
                        index = match
                        break
                return st.sidebar.selectbox("Model (live list from your account)", models, index=index)
        except requests.RequestException:
            st.sidebar.warning("Could not fetch the model list (check the API key). Enter a model manually.")
    return st.sidebar.text_input("Model", default_model)


def sidebar_settings() -> dict:
    """Render sidebar controls and return the chosen configuration."""
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.radio("Planner mode", ["AI Agent (LangChain)", "Offline (no LLM)"],
                            help="Offline mode runs the same tools in a fixed order - no API key needed.")
    config: dict[str, Any] = {"mode": mode}
    if mode.startswith("AI"):
        provider = st.sidebar.selectbox("LLM provider", list(PROVIDER_DEFAULTS))
        default_model, secret_name = PROVIDER_DEFAULTS[provider]
        config["provider"] = provider
        config["api_key"] = st.sidebar.text_input(f"API key ({secret_name})", value=get_secret(secret_name),
                                                  type="password")
        config["model"] = choose_model(provider, config["api_key"], default_model)

    data = get_data()
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Data status")
    for name in ("flights", "hotels", "places"):
        origin = "built-in sample" if name in data["using_sample"] else "data/ folder"
        st.sidebar.caption(f"{name}: {len(data[name])} records ({origin})")

    with st.sidebar.expander("Upload your JSON datasets"):
        uploads = {n: st.file_uploader(f"{n}.json", type="json", key=f"up_{n}")
                   for n in ("flights", "hotels", "places")}
        if st.button("Save & reload data") and any(uploads.values()):
            DATA_DIR.mkdir(exist_ok=True)
            for name, file in uploads.items():
                if file is not None:
                    (DATA_DIR / f"{name}.json").write_bytes(file.getvalue())
            get_data.cache_clear()
            st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"App version {APP_VERSION}")
    st.sidebar.caption(f"Running file: {Path(__file__).resolve()}")
    return config


def show_agent_trace(steps: list[dict]) -> None:
    """Display the agent's tool calls and results."""
    with st.expander(f"🔍 Agent reasoning trace ({len(steps)} steps)"):
        for i, step in enumerate(steps, 1):
            if step["kind"] == "call":
                st.markdown(f"**{i}. Called `{step['tool']}`**")
                st.code(json.dumps(step["payload"], indent=2, ensure_ascii=False), language="json")
            else:
                st.markdown(f"**{i}. Result from `{step['tool']}`**")
                st.code(step["payload"][:1500], language="json")


def execute_with_agent(query: str, config: dict) -> None:
    """Run the LangChain agent and display its answer."""
    if not config.get("api_key"):
        st.error("Please enter an API key in the sidebar (or use Offline mode).")
        return
    try:
        with st.spinner("The agent is planning your trip (calling tools)..."):
            agent = build_agent(build_llm(config["provider"], config["model"], config["api_key"]))
            final_text, steps = run_agent(agent, query)
    except Exception as exc:  # network / auth / model errors
        message = str(exc)
        hint = ""
        if "model_not_found" in message or "does not exist" in message:
            hint = " -> Pick another model from the sidebar list (this one is unavailable to your key)."
        elif "429" in message or "rate" in message.lower():
            hint = " -> Rate limit reached. Wait a minute and retry, or choose a smaller model."
        elif "401" in message or "invalid_api_key" in message:
            hint = " -> The API key looks invalid. Re-copy it from your provider console."
        st.error(f"Agent error: {message}{hint}")
        return
    markdown_part, plan_json = split_markdown_and_json(final_text)
    st.markdown(markdown_part)
    show_agent_trace(steps)
    if plan_json:
        st.download_button("⬇️ Download itinerary (JSON)", json.dumps(plan_json, indent=2, ensure_ascii=False),
                           "itinerary.json", "application/json")
        with st.expander("Structured JSON output"):
            st.json(plan_json)


def show_budget_warning(budget: dict, budget_limit: float) -> None:
    """Compare the budget with the user's limit (0 = no limit), respecting estimate vs confirmed."""
    if not budget_limit:
        return
    known_no_flight = budget["known_subtotal"]
    known_with_flight = budget.get("known_trip_subtotal", known_no_flight)
    total = budget["total"]
    label = "Estimated total (with reference fare)" if budget.get("total_is_estimate") else "Total"

    if known_no_flight > budget_limit:
        st.warning(f"Known expenses without flights ({_money(known_no_flight)}) already exceed your budget of "
                   f"{_money(budget_limit)}.")
    elif total is None and known_with_flight > budget_limit:
        st.warning(f"Known expenses so far, including available flights ({_money(known_with_flight)}), "
                   f"already exceed your budget of {_money(budget_limit)}. "
                   "The complete total is unavailable because a flight leg is missing.")
    elif total is None:
        st.info("Known expenses are within your budget, but the final cost can't be confirmed "
                "until all flights are found.")
    elif total > budget_limit:
        st.warning(f"{label} ({_money(total)}) exceeds your budget of {_money(budget_limit)}.")
    else:
        st.success(f"{label} ({_money(total)}) is within your budget of {_money(budget_limit)}.")



def trip_form_tab(config: dict) -> None:
    """Form-based trip planner."""

    data = get_data()

    origins = sorted({
        f["source"] for f in data["flights"] if f["source"]
    }) or ["Delhi"]

    destinations = sorted({
        f["destination"] for f in data["flights"] if f["destination"]
    }) or ["Goa"]

    categories = sorted({
        c.strip().lower()
        for p in data["places"]
        for c in re.split(r"[,/|]", p["category"])
        if c.strip()
    })

    flight_days = sorted({
        d.isoformat()
        for d in (_flight_date(f) for f in data["flights"])
        if d
    })

    if flight_days:
        shown = ", ".join(flight_days[:5]) + (
            " ..." if len(flight_days) > 5 else ""
        )
        st.caption(f"📅 Flights in your dataset are dated: {shown}")

    # Origin sits outside the form so destination updates when it changes.
    source = st.selectbox("From", origins)

    reachable = sorted({
        f["destination"]
        for f in data["flights"]
        if f["destination"]
        and _city_match(_norm_city(source), f["source"])
    }) or destinations

    with st.form("trip_form"):

        destination = st.selectbox(
            "To (cities with flights from your origin)",
            reachable
        )

        c3, c4, c5 = st.columns(3)

        start = c3.date_input(
            "Start date",
            date.today() + timedelta(days=7)
        )

        end = c4.date_input(
            "End date",
            date.today() + timedelta(days=9)
        )

        travelers = c5.number_input(
            "Travelers", 1, 10, 1
        )

        c6, c7, c8 = st.columns(3)

        # Flight preference
        flight_pref = c6.radio(
            "Flight preference",
            ["cheapest", "fastest"],
            horizontal=True
        )

        flight_pref = flight_pref.strip().lower()

        min_rating = c7.slider(
            "Minimum hotel rating",
            0.0, 5.0, 3.5, 0.5
        )
        max_hotel_price = st.number_input(
    f"Maximum hotel price per night ({CURRENCY}, 0 = no limit)",
    min_value=0,
    max_value=100000,
    value=0,
    step=500
)

        budget_limit = c8.number_input(
            f"Max total budget ({CURRENCY}, 0 = none)",
            0, 1_000_000, 0, 1000
        )

        interests = st.multiselect(
            "Interests",
            categories[:40]
        )

        c9, c10 = st.columns(2)

        daily_expense = c9.number_input(
            f"Daily local expense per person ({CURRENCY})",
            0, 20000, 1500, 100
        )

        round_trip = c10.checkbox(
            "Round-trip flights"
        )

        use_nearest = st.checkbox(
            "If no flight exists on my date, show the closest dataset flight "
            "as a reference fare",
            value=True
        )

        notes = st.text_area(
            "Extra notes (optional)",
            placeholder="e.g. vegetarian food, family with kids..."
        )

        submitted = st.form_submit_button(
            "🚀 Plan my trip",
            use_container_width=True
        )

    # Validation starts outside the form.
    if not submitted:
        return

    if end < start:
        st.error("End date must be on or after the start date.")
        return

    days = (end - start).days + 1

    if not 1 <= days <= 14:
        st.warning(
            "Please choose a trip between 1 and 14 days "
            "(3-7 days recommended)."
        )
        return

    if _norm_city(source) == _norm_city(destination):
        st.error("Origin and destination must be different.")
        return

    # Offline planner: no LLM/API call.
   
    # Offline planner: no LLM/API call.
    if config["mode"].startswith("Offline"):

        plan = build_offline_plan(
            source,
            destination,
            start,
            end,
            int(travelers),
            interests,
            flight_pref,
            min_rating,
            round_trip,
            float(daily_expense),
            max_hotel_price=float(max_hotel_price),
            use_nearest_date=use_nearest
        )

        st.markdown(render_plan_markdown(plan))

        show_budget_warning(
            plan["budget"],
            budget_limit
        )

        st.download_button(
            "⬇️ Download itinerary (JSON)",
            json.dumps(
                plan,
                indent=2,
                ensure_ascii=False
            ),
            "itinerary.json",
            "application/json"
        )

        with st.expander("Structured JSON output"):
            st.json(plan)

    # AI Agent planner.
    else:

        query = (
            f"Plan a {days}-day trip from {source} to {destination}, "
            f"{start.isoformat()} to {end.isoformat()}, "
            f"for {int(travelers)} traveler(s). "
            f"Prefer the {flight_pref} flight. "
            f"Hotel minimum rating {min_rating}. "
            f"Maximum hotel price per night: "
            f"{CURRENCY}{max_hotel_price}. "
            f"{'Interests: ' + ', '.join(interests) + '. ' if interests else ''}"
            f"{'Total budget limit: ' + CURRENCY + str(budget_limit) + '. ' if budget_limit else ''}"
            f"Daily local expense {CURRENCY}{daily_expense} per person. "
            f"{'Use round-trip flights. ' if round_trip else 'One-way flight cost only.'} "
            f"{notes}"
        )

        execute_with_agent(query, config)


def free_form_tab(config: dict) -> None:
    """Natural-language trip request handled by the agent."""

    if not config["mode"].startswith("AI"):
        st.info(
            "Switch to **AI Agent (LangChain)** mode in the sidebar "
            "to use free-form requests."
        )
        return

    query = st.text_area(
        "Describe your trip",
        height=120,
        placeholder=(
            "Plan a 3-day trip from Delhi to Goa from 12 to 14 Feb. "
            "I like beaches and heritage, budget around ₹20,000."
        )
    )

    if st.button("Ask the agent", use_container_width=True) and query.strip():
        execute_with_agent(query.strip(), config)



def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="AI Travel Planner", page_icon="🧳", layout="wide")
    st.markdown("""
<style>
.stApp { background-color: #292238; color: #F5F0FA; }
[data-testid="stSidebar"] { background-color: #211A30; }
h1, h2, h3 { color: #E5D5F2 !important; }
.stApp p, .stApp label { color: #F5F0FA; }
.stButton > button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {
    background-color: #A77BC8 !important; color: #FFFFFF !important;
    border: 2px solid #A77BC8 !important; border-radius: 12px !important; font-weight: 600 !important;
}
.stButton > button p, [data-testid="stFormSubmitButton"] button p,
[data-testid="stDownloadButton"] button p { color: #FFFFFF !important; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover { background-color: #895DAA !important; border-color: #895DAA !important; }
.stTextInput input, .stNumberInput input { background-color: #403451; color: white; }
[data-baseweb="select"] > div { background-color: #403451; color: white; }
[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

    st.title("🧳 Agentic AI Travel Planning Assistant")
    st.caption("LangChain agent + flight, hotel, places, live weather (Open-Meteo) and budget tools.")
    config = sidebar_settings()
    if get_data()["using_sample"]:
        st.info(f"Using built-in sample data for: {', '.join(get_data()['using_sample'])}. "
                "Put your JSON files in a `data/` folder (or upload them in the sidebar) to use your own.")
    tab_form, tab_chat = st.tabs(["🧭 Trip planner", "💬 Free-form request"])
    with tab_form:
        trip_form_tab(config)
    with tab_chat:
        free_form_tab(config)


if __name__ == "__main__":
    main()