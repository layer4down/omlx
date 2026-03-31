# SPDX-License-Identifier: Apache-2.0
"""
Tests for structuredContent support in MCP tool results.

This addresses issue #469: MCP web search returns no output.
Web search MCP servers return results in structuredContent field.
"""

from unittest.mock import MagicMock


def test_extract_content_with_structured_content():
    """Test that _extract_content falls back to structuredContent when content is empty."""
    from omlx.mcp.client import MCPClient
    from omlx.mcp.types import MCPServerConfig, MCPTransport

    config = MCPServerConfig(
        name="test",
        transport=MCPTransport.STDIO,
        command="python",
    )
    client = MCPClient(config)

    mock_result = MagicMock()
    mock_result.content = []
    mock_result.structuredContent = {"results": ["Result 1", "Result 2"]}

    result = client._extract_content(mock_result)

    assert result == {"results": ["Result 1", "Result 2"]}


def test_extract_content_with_text_content():
    """Test that _extract_content still works with regular text content."""
    from omlx.mcp.client import MCPClient
    from omlx.mcp.types import MCPServerConfig, MCPTransport

    config = MCPServerConfig(
        name="test",
        transport=MCPTransport.STDIO,
        command="python",
    )
    client = MCPClient(config)

    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="Tool output")]

    result = client._extract_content(mock_result)

    assert result == "Tool output"


def test_extract_content_empty_no_structured():
    """Test that _extract_content returns None when both content and structuredContent are empty."""
    from omlx.mcp.client import MCPClient
    from omlx.mcp.types import MCPServerConfig, MCPTransport

    config = MCPServerConfig(
        name="test",
        transport=MCPTransport.STDIO,
        command="python",
    )
    client = MCPClient(config)

    mock_result = MagicMock()
    mock_result.content = []
    mock_result.structuredContent = None

    result = client._extract_content(mock_result)

    assert result is None


def test_extract_content_priority_text_over_structured():
    """Test that text content takes priority over structuredContent."""
    from omlx.mcp.client import MCPClient
    from omlx.mcp.types import MCPServerConfig, MCPTransport

    config = MCPServerConfig(
        name="test",
        transport=MCPTransport.STDIO,
        command="python",
    )
    client = MCPClient(config)

    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="Text content")]
    mock_result.structuredContent = {"results": ["Structured"]}

    result = client._extract_content(mock_result)

    assert result == "Text content"
