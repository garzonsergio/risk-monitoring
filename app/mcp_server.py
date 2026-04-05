import asyncio
import os
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import mcp.types as types

load_dotenv()

# Import your existing tools directly — no duplication
from app.agent.agent import (
    search_knowledge_base,
    check_live_river_level,
    check_all_levels,
    check_active_rainfall,
    check_full_network_status,
    check_precipitation_by_date,
    check_river_levels_by_date,
)

server = Server("antioquia-risk-monitor")

# ── Register tools ────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Tell Claude Desktop which tools are available."""
    return [
        Tool(
            name="check_full_network_status",
            description=(
                "Check the full live status of ALL stations across Antioquia — "
                "river levels (sn_*) AND precipitation (sp_*, sm_*). "
                "Use for broad questions about overall network status or alerts."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="check_active_rainfall",
            description=(
                "Check which stations are currently reporting active rainfall "
                "across Antioquia right now. Scans all sp_* and sm_* stations live."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="check_all_levels",
            description=(
                "Check live alert status of ALL river level stations (sn_*). "
                "Returns which rivers are in yellow, orange, or red alert."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="check_live_river_level",
            description=(
                "Check the current live level of ONE specific river station "
                "and compare against its alert thresholds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "station_codigo": {
                        "type": "string",
                        "description": "Station code, e.g. sn_1030",
                    }
                },
                "required": ["station_codigo"],
            },
        ),
        Tool(
            name="check_precipitation_by_date",
            description=(
                "Check rainfall data for ALL sp_* and sm_* stations on a specific "
                "past date with hourly breakdown. Use for questions about yesterday "
                "or a specific date."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date_str": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format, e.g. 2026-03-30",
                    }
                },
                "required": ["date_str"],
            },
        ),
        Tool(
            name="check_river_levels_by_date",
            description=(
                "Check river level history for all sn_* stations on a specific "
                "past date, flagging any that exceeded alert thresholds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date_str": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format, e.g. 2026-03-30",
                    }
                },
                "required": ["date_str"],
            },
        ),
        Tool(
            name="search_knowledge_base",
            description=(
                "Search historical summaries, alert thresholds, and station "
                "metadata stored in the vector knowledge base. Use for questions "
                "about thresholds, station locations, or historical patterns."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    }
                },
                "required": ["query"],
            },
        ),
    ]

# ── Execute tools ─────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool when Claude Desktop calls it."""
    try:
        if name == "check_full_network_status":
            result = check_full_network_status.invoke({})
        elif name == "check_active_rainfall":
            result = check_active_rainfall.invoke({})
        elif name == "check_all_levels":
            result = check_all_levels.invoke({})
        elif name == "check_live_river_level":
            result = check_live_river_level.invoke({
                "station_codigo": arguments["station_codigo"]
            })
        elif name == "check_precipitation_by_date":
            result = check_precipitation_by_date.invoke({
                "date_str": arguments["date_str"]
            })
        elif name == "check_river_levels_by_date":
            result = check_river_levels_by_date.invoke({
                "date_str": arguments["date_str"]
            })
        elif name == "search_knowledge_base":
            result = search_knowledge_base.invoke({
                "query": arguments["query"]
            })
        else:
            result = f"Unknown tool: {name}"

    except Exception as e:
        result = f"Tool {name} failed: {str(e)}"

    return [TextContent(type="text", text=str(result))]

# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())