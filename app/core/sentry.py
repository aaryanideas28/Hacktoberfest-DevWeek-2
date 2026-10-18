import os

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    sentry_sdk = None
    FastApiIntegration = None
    SENTRY_AVAILABLE = False

def init_sentry():
    """
    Initialize Sentry SDK for Agent & API Tracing.
    Will gracefully no-op if SENTRY_DSN is not configured or sentry_sdk is missing.
    """
    if not SENTRY_AVAILABLE:
        print("[Sentry] sentry-sdk package not installed. Tracing disabled.")
        return False

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
        auto_enabling_integrations=False,
    )
    print(f"[Sentry] Initialized Sentry tracing in '{environment}' environment.")
    return True


def record_compression_span(original_tokens: int, compressed_tokens: int, mode: str, tokens_saved: int = 0):
    """
    Record custom span/tags for prompt compression in Sentry.
    """
    if not SENTRY_AVAILABLE or not sentry_sdk.is_initialized():
        return

    saved = tokens_saved if tokens_saved > 0 else max(0, original_tokens - compressed_tokens)
    ratio = round((saved / original_tokens) * 100, 2) if original_tokens > 0 else 0

    with sentry_sdk.start_span(op="prompt.compress", description=f"Mode: {mode}"):
        sentry_sdk.set_tag("compression.mode", mode)
        sentry_sdk.set_tag("compression.tokens_saved", saved)
        sentry_sdk.set_tag("compression.savings_ratio_pct", ratio)

