import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

def init_sentry():
    """
    Initialize Sentry SDK for Agent & API Tracing.
    Will gracefully no-op if SENTRY_DSN is not configured.
    """
    dsn = os.getenv("SENTRY_DSN")
    environment = os.getenv("ENVIRONMENT", "development")

    if not dsn:
        print("[Sentry] No SENTRY_DSN detected. Tracing running in offline/dry-run mode.")
        return False

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        integrations=[FastApiIntegration()],
        traces_sample_rate=1.0,
        send_default_pii=False,
    )
    print(f"[Sentry] Initialized Sentry tracing in '{environment}' environment.")
    return True

def record_compression_span(original_tokens: int, compressed_tokens: int, mode: str):
    """
    Record custom span/tags for prompt compression in Sentry.
    """
    tokens_saved = max(0, original_tokens - compressed_tokens)
    ratio = round((tokens_saved / original_tokens) * 100, 2) if original_tokens > 0 else 0

    sentry_sdk.set_tag("compression.mode", mode)
    sentry_sdk.set_tag("compression.tokens_saved", tokens_saved)
    sentry_sdk.set_tag("compression.savings_ratio_pct", ratio)
