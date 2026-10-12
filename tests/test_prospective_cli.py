from scripts.prospective_validation import status


def test_empty_status_does_not_create_fake_records(tmp_path):
    value=status(tmp_path)
    assert value['predictions']==0 and value['scores']==0
    assert value['enhanced_candidate']=='not_ready'
    assert value['pollution_input']=='not_connected'
    assert not (tmp_path/'ledger.sqlite3').exists()
