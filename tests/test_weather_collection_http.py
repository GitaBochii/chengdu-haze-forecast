from scripts.collect_weather_training_fields import worker_http_client


def test_transient_http_retry_is_bounded_and_read_only():
    with worker_http_client() as client:
        retry = client.get_adapter('https://example.invalid').max_retries
        assert retry.total == 3
        assert retry.allowed_methods == {'HEAD', 'GET'}
        assert retry.is_retry('GET', 503)
        assert retry.is_retry('HEAD', 429)
        assert not retry.is_retry('GET', 404)
        assert not retry.is_retry('POST', 503)
        assert retry.respect_retry_after_header
