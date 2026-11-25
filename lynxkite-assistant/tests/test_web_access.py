"""The web access tools must call the Firecrawl v2 API."""

import asyncio
import json
import os

import pytest
import requests
from lynxkite_assistant import assistant
from lynxkite_assistant.instructions import INTERNET_ACCESS_INFO, SYSTEM_PROMPT

TOOLS = [assistant.internet_search, assistant.scrape_web_page, assistant.map_web_page]


class FakeResponse:
    text = '{"success": true}'


@pytest.fixture
def posts(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json})
        return FakeResponse()

    monkeypatch.setattr(assistant, "web_access_url", "https://example.test")
    monkeypatch.setattr(assistant.requests, "post", fake_post)
    return calls


def test_internet_search_uses_v2_with_limit_5(posts):
    assert assistant.internet_search("lynxkite") == '{"success": true}'
    assert posts == [
        {
            "url": "https://example.test/v2/search",
            "json": {"query": "lynxkite", "limit": 5},
        }
    ]


def test_internet_search_limit_can_be_overridden(posts):
    assistant.internet_search("lynxkite", {"limit": 20})
    assert posts[0]["json"] == {"query": "lynxkite", "limit": 20}


def test_scrape_web_page_uses_v2(posts):
    assistant.scrape_web_page("https://lynxkite.com", {"formats": ["markdown"]})
    assert posts == [
        {
            "url": "https://example.test/v2/scrape",
            "json": {"url": "https://lynxkite.com", "formats": ["markdown"]},
        }
    ]


def test_map_web_page_uses_v2(posts):
    assistant.map_web_page("https://lynxkite.com")
    assert posts == [
        {
            "url": "https://example.test/v2/map",
            "json": {"url": "https://lynxkite.com"},
        }
    ]


@pytest.mark.parametrize("tool", TOOLS)
def test_tools_send_api_key(tool, monkeypatch):
    sent_headers = []

    def fake_post(url, json=None, headers=None, timeout=None):
        sent_headers.append(headers)
        return FakeResponse()

    monkeypatch.setattr(assistant, "web_access_url", "https://example.test")
    monkeypatch.setattr(assistant, "headers", {"Authorization": "Bearer key"})
    monkeypatch.setattr(assistant.requests, "post", fake_post)
    tool("https://lynxkite.com")
    assert sent_headers == [{"Authorization": "Bearer key"}]


@pytest.mark.parametrize("tool", TOOLS)
def test_tools_return_error_instead_of_raising(tool, monkeypatch):
    # The agent sees the return value. An exception would end its run.
    def failing_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(assistant, "web_access_url", "https://example.test")
    monkeypatch.setattr(assistant.requests, "post", failing_post)
    result = tool("https://lynxkite.com")
    assert result.startswith("Error ")
    assert "connection refused" in result


class AgentCreated(Exception):
    pass


def agent_options(monkeypatch, tmp_path, web_access_url) -> dict:
    """Runs assistant_stream until it creates the agent and returns the agent's options."""
    options = {}

    def fake_create_deep_agent(**kwargs):
        options.update(kwargs)
        raise AgentCreated()

    monkeypatch.setattr(assistant, "web_access_url", web_access_url)
    monkeypatch.setattr(
        assistant.deepagents, "create_deep_agent", fake_create_deep_agent
    )
    req = assistant.AssistantCompletionRequest(
        workspace=str(tmp_path / "test.lynxkite"), messages=[]
    )
    with pytest.raises(AgentCreated):
        asyncio.run(assistant.assistant_stream(req, skill_root=str(tmp_path)))
    return options


def test_agent_gets_web_tools_when_url_is_set(monkeypatch, tmp_path):
    options = agent_options(monkeypatch, tmp_path, "https://example.test")
    assert options["tools"] == TOOLS
    assert options["system_prompt"] == SYSTEM_PROMPT + INTERNET_ACCESS_INFO


def test_agent_gets_no_web_tools_without_url(monkeypatch, tmp_path):
    options = agent_options(monkeypatch, tmp_path, None)
    assert options["tools"] == []
    assert options["system_prompt"] == SYSTEM_PROMPT


live = pytest.mark.skipif(
    not (
        os.environ.get("LYNXKITE_TEST_WEB_ACCESS")
        and os.environ.get("LYNXKITE_WEB_ACCESS_URL")
    ),
    reason="Calls a real web access service and uses credits."
    " Set LYNXKITE_TEST_WEB_ACCESS=1 and LYNXKITE_WEB_ACCESS_URL to run it.",
)


@live
def test_live_search_returns_results_under_data_web():
    # The agent instructions promise this response shape.
    response = json.loads(assistant.internet_search("LynxKite"))
    results = response["data"]["web"]
    assert 0 < len(results) <= 5
    assert all("url" in r for r in results)


@live
def test_live_map_returns_links_as_objects():
    # The agent instructions promise this response shape.
    response = json.loads(assistant.map_web_page("https://www.firecrawl.dev"))
    links = response["links"]
    assert links
    assert all("url" in link for link in links)
