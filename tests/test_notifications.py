import asyncio
from MLXOllama.local_tools import get_notifications


def test_get_notifications():
    """Ensure get_notifications returns a valid response without asserting specific content."""
    response = asyncio.run(get_notifications())
    assert response is not None
    assert isinstance(response, dict)


if __name__ == "__main__":
    result = asyncio.run(get_notifications())
    print("Notification response:", result)
    assert result is not None
    assert isinstance(result, dict)
    print("✅ Successfully received a valid response.")
