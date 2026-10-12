from scripts.audit_weather_training_batch import pending_rows, summarize


def test_resume_keeps_unknown_distinct_from_pending():
    planned = [{'url': 'a', 'issue_utc': 'x'}, {'url': 'b', 'issue_utc': 'x'}, {'url': 'c', 'issue_utc': 'y'}]
    done = {'a': {'state': 'available'}, 'b': {'state': 'unknown'}}
    assert pending_rows(planned, done, 1) == [planned[2]]
    result = summarize(planned, done)
    assert result['available'] == 1
    assert result['unknown'] == 1
    assert result['pending'] == 1
    assert result['complete_origins'] == 0
