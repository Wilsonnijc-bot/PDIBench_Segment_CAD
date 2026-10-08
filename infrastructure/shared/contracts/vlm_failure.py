"""Typed, secret-free failure evidence used for VLM queue decisions."""
import urllib.error


class VLMTimeoutError(RuntimeError):
    failure = {'category': 'vlm_timeout', 'code': 'no_response_timeout', 'defer': True}


def failure_record(error):
    return dict(getattr(error, 'failure', {}))


def is_request_timeout(error):
    return isinstance(error, TimeoutError) or (
        isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError))


def raise_timeout(record):
    if record.get('failure', {}).get('category') == 'vlm_timeout':
        raise VLMTimeoutError('VLM request timed out without a response')
