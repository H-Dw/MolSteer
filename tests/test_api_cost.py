"""Money accounting must separate cached input and include reasoning only once."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('api_cost', Path(__file__).resolve().parents[1]/'scripts/report_api_cost.py')
billing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(billing)
PRICE = {'prompt':'0.0000014','completion':'0.0000044','input_cache_read':'0.00000026'}


def test_cached_input_replaces_standard_rate():
    assert billing.token_cost(1000, 200, 600, PRICE) == pytest.approx(0.001596)
    assert billing.token_cost(None, 200, 0, PRICE) is None
    with pytest.raises(ValueError):
        billing.token_cost(100, 20, 101, PRICE)


def test_modules_reasoning_pending_and_missing_cost_are_explicit(tmp_path):
    folder = tmp_path/'attempt_01'
    folder.mkdir()
    path = folder/'api_requests.jsonl'
    events = [
        {'agent':'molthinker.mathematics','event':'request_started','request_id':'one'},
        {'agent':'molthinker.mathematics','event':'request_completed','request_id':'one',
         'usage':[{'input_tokens':9999,'output_tokens':9999,'provider_receipt':{'id':'gen-one',
            'usage':{'prompt_tokens':1000,'completion_tokens':200,'cost':0.003,
                     'prompt_tokens_details':{'cached_tokens':600},
                     'completion_tokens_details':{'reasoning_tokens':150}}}}]},
        {'agent':'molreader','event':'request_started','request_id':'two'},
        {'agent':'molreader','event':'request_failed','request_id':'two','status_code':502},
        {'agent':'molthinker.researcher','event':'request_started','request_id':'three'},
    ]
    path.write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')
    modules, total = billing.summarize([path], PRICE)
    assert total['input_tokens'] == 1000
    assert total['output_tokens'] == 200
    assert total['reasoning_tokens'] == 150
    assert total['total_tokens'] == 1200
    assert total['billed_usd'] == 0.003
    assert total['pending'] == 1 and total['failed'] == 1
    assert total['missing_cost_receipts'] == 2
    executor = next(m for m in modules if m['module']=='molexecutor')
    assert executor['input_tokens'] == 0 and executor['started'] == 0
