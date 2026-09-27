"""USD per million tokens (input, output). Anthropic first-party API list prices."""

PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_READ_MULTIPLIER = 0.1


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> float:
    price_in, price_out = PRICES.get(model, PRICES["claude-opus-5"])
    return (
        input_tokens * price_in
        + cache_read_tokens * price_in * CACHE_READ_MULTIPLIER
        + output_tokens * price_out
    ) / 1_000_000
