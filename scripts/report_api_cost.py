"""Per-module GLM token receipts, OpenRouter price estimates and actual billing."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
MODEL = 'z-ai/glm-5.3'
API = 'https://openrouter.ai/api/v1/'
ROLES = ('molreader', 'molthinker.biology', 'molthinker.mathematics',
         'molthinker.researcher', 'molthinker', 'molexecutor', 'molmonitor', 'flowr_root')
TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'total_tokens', 'cached_tokens', 'reasoning_tokens')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def get(endpoint, key=None):
    headers = {'Authorization':'Bearer '+key} if key else {}
    with urlopen(Request(API+endpoint, headers=headers), timeout=30) as response:
        return json.load(response)['data']


def credentials():
    from molsteer.agents.config import load_config, SecretStore
    return SecretStore(load_config()).get('openrouter')


def snapshot(root):
    key = credentials()
    model = next(m for m in get('models') if m['id'] == MODEL)
    endpoints = get('models/'+MODEL+'/endpoints')
    result = {'observed_at_utc':datetime.now(timezone.utc).isoformat(), 'model':MODEL,
        'pricing_per_token':model['pricing'], 'model_pricing_source':API+'models',
        'provider_pricing_source':API+'models/'+MODEL+'/endpoints',
        'provider_endpoints':[{k:e.get(k) for k in ('provider_name','tag','pricing')}
                              for e in endpoints['endpoints']]}
    for endpoint, fields in {
        'credits':('total_credits','total_usage'),
        'key':('limit','limit_remaining','usage','usage_daily','limit_reset','is_free_tier'),
    }.items():
        try:
            payload = get(endpoint, key)
            result[endpoint] = {k:payload.get(k) for k in fields}
        except Exception as error:
            result[endpoint] = {'error_type':type(error).__name__, 'status_code':getattr(error,'code',None)}
    write(root/'api_preflight.json', result)
    return result


def token_cost(prompt, completion, cached, pricing):
    """Reasoning is already part of completion; cached input replaces prompt rate."""
    if prompt is None or completion is None:
        return None
    if not 0 <= cached <= prompt:
        raise ValueError('Cached tokens must be a subset of prompt tokens')
    if cached and 'input_cache_read' not in pricing:
        return None
    p = Decimal(str(pricing['prompt']))
    c = Decimal(str(pricing['completion']))
    cache = Decimal(str(pricing.get('input_cache_read', 0)))
    return float((Decimal(prompt)-Decimal(cached))*p+Decimal(cached)*cache+Decimal(completion)*c)


def summarize(paths, pricing, generation_metadata=None):
    metadata = generation_metadata or {}
    modules = {role:{'module':role, 'started':0, 'completed':0, 'failed':0, 'pending':0,
                    **{k:0 for k in TOKEN_FIELDS}, 'list_price_usd':0., 'billed_usd':0.,
                    'missing_token_receipts':0, 'missing_cost_receipts':0, 'requests':[]}
               for role in ROLES}
    events = defaultdict(dict)
    for path in paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            events[(path.parent.name, row['agent'], row['request_id'])][row['event']] = row
    for (attempt, role, rid), group in events.items():
        if role not in modules:
            modules[role] = {**modules['flowr_root'], 'module':role, 'requests':[]}
        module = modules[role]
        module['started'] += int('request_started' in group)
        final = group.get('request_completed') or group.get('request_failed')
        status = 'completed' if 'request_completed' in group else 'failed' if final else 'pending'
        module[status] += 1
        usage_rows = final.get('usage', []) if final else []
        if not usage_rows:
            module['missing_token_receipts'] += 1
            module['missing_cost_receipts'] += 1
            module['requests'].append({'attempt':attempt, 'request_id':rid, 'status':status,
                                      'status_code':final.get('status_code') if final else None,
                                      'token_usage':None, 'billed_usd':None})
            continue
        for usage in usage_rows:
            receipt = usage.get('provider_receipt', {})
            provider_usage = receipt.get('usage', {})
            gid = receipt.get('id')
            stats = metadata.get(gid, {})
            prompt = provider_usage.get('prompt_tokens', usage.get('input_tokens'))
            completion = provider_usage.get('completion_tokens', usage.get('output_tokens'))
            cached = provider_usage.get('prompt_tokens_details', {}).get('cached_tokens', usage.get('cached_tokens', 0))
            reasoning = provider_usage.get('completion_tokens_details', {}).get('reasoning_tokens', usage.get('reasoning_tokens', 0))
            cost = provider_usage.get('cost')
            if cost is None:
                cost = stats.get('total_cost')
            estimate = token_cost(prompt, completion, cached, pricing)
            request = {'attempt':attempt, 'request_id':rid, 'status':status, 'generation_id':gid,
                'provider':receipt.get('provider') or stats.get('provider_name'),
                'served_model':receipt.get('model') or stats.get('model'),
                'input_tokens':prompt, 'output_tokens':completion,
                'cached_tokens':cached, 'reasoning_tokens':reasoning,
                'list_price_usd':estimate, 'billed_usd':cost}
            module['requests'].append(request)
            if prompt is None or completion is None:
                module['missing_token_receipts'] += 1
            else:
                values = {'input_tokens':prompt, 'output_tokens':completion, 'total_tokens':prompt+completion,
                          'cached_tokens':cached, 'reasoning_tokens':reasoning}
                for k,v in values.items():
                    module[k] += v
            if estimate is not None:
                module['list_price_usd'] += estimate
            if cost is None:
                module['missing_cost_receipts'] += 1
            else:
                module['billed_usd'] += cost
    fields = ('started','completed','failed','pending',*TOKEN_FIELDS,'list_price_usd',
              'billed_usd','missing_token_receipts','missing_cost_receipts')
    return list(modules.values()), {k:sum(m[k] for m in modules.values()) for k in fields}


def report(root, query_generations=False):
    preflight = read(root/'api_preflight.json')
    paths = sorted(root.glob('attempt_*/api_requests.jsonl'))
    previous = root/'api_cost_report.json'
    metadata = read(previous).get('generation_metadata', {}) if previous.exists() else {}
    if query_generations:
        key = credentials()
        for path in paths:
            for line in path.read_text(encoding='utf-8').splitlines():
                for usage in json.loads(line).get('usage', []):
                    gid = usage.get('provider_receipt', {}).get('id')
                    if not gid or gid in metadata:
                        continue
                    try:
                        stats = get('generation?id='+quote(gid, safe=''), key)
                        metadata[gid] = {k:stats.get(k) for k in ('id','provider_name','model',
                            'native_tokens_prompt','native_tokens_completion','native_tokens_reasoning',
                            'native_tokens_cached','total_cost','is_byok','finish_reason')}
                    except Exception as error:
                        metadata[gid] = {'error_type':type(error).__name__, 'status_code':getattr(error,'code',None)}
    modules, totals = summarize(paths, preflight['pricing_per_token'], metadata)
    result = {'observed_at_utc':datetime.now(timezone.utc).isoformat(), 'model':MODEL, 'currency':'USD',
        'price_snapshot_at':preflight['observed_at_utc'], 'pricing_per_token':preflight['pricing_per_token'],
        'pricing_source':preflight['model_pricing_source'], 'provider_endpoints':preflight['provider_endpoints'],
        'modules':modules, 'totals':totals, 'generation_metadata':metadata,
        'formula':'(input_tokens-cached_tokens)*prompt_price + cached_tokens*cache_read_price + output_tokens*completion_price',
        'notes':['Reasoning tokens are included in output tokens; do not charge them twice.',
                 'Actual cost comes from OpenRouter usage.cost or generation.total_cost.',
                 'Price estimates use the saved models endpoint; provider routing may change actual charges.',
                 'Missing or failed request receipts are unknown, never assumed free.',
                 'FLOWR inference and deterministic execution use no LLM tokens.']}
    if query_generations:
        try:
            after = get('credits', key)
            before = preflight.get('credits', {})
            if 'total_usage' in before:
                delta = after['total_usage']-before['total_usage']
                result['account_reconciliation'] = {'before':before,
                    'after':{k:after[k] for k in ('total_credits','total_usage')},
                    'usage_delta_usd':delta, 'known_request_cost_usd':totals['billed_usd'],
                    'unattributed_difference_usd':delta-totals['billed_usd'],
                    'scope':'Account may include other tasks and delayed billing.'}
        except Exception as error:
            result['account_reconciliation'] = {'error_type':type(error).__name__}
    write(root/'api_cost_report.json', result)
    lines = ['# 5i0b API token 与费用', '', f"模型：{MODEL}；币种：USD；定价快照：{result['price_snapshot_at']}。", '',
        '| 模块 | 完成调用 | 失败 | 待完成 | 输入 token | 缓存 token | 输出 token | 推理 token（含于输出） | 挂牌价估算 USD | 回执费用 USD |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for module in modules:
        lines.append('| '+str(module['module'])+' | '+' | '.join(str(module[k]) for k in
            ('completed','failed','pending','input_tokens','cached_tokens','output_tokens','reasoning_tokens'))+
            f" | {module['list_price_usd']:.9f} | {module['billed_usd']:.9f} |")
    lines += ['', f"已回执费用合计：${totals['billed_usd']:.9f}；挂牌价估算合计：${totals['list_price_usd']:.9f}。",
              f"缺少 token 回执：{totals['missing_token_receipts']}；缺少费用回执：{totals['missing_cost_receipts']}。缺失部分费用未知。", '',
              '计算：(输入−缓存)×输入单价 + 缓存×缓存读取单价 + 输出×输出单价。推理 token 已包含在输出中。', '',
              '[OpenRouter GLM-5.3 定价](https://openrouter.ai/z-ai/glm-5.3)。原始定价、逐请求回执和供应商信息保存在 JSON 报告。']
    (root/'api_cost_report.zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-root', required=True, type=Path)
    parser.add_argument('--snapshot-only', action='store_true')
    parser.add_argument('--query-generations', action='store_true')
    args = parser.parse_args()
    root = args.test_root.resolve()
    if not root.is_relative_to(REPO/'test'):
        parser.error('Billing artifacts must remain under test')
    if args.snapshot_only:
        value = snapshot(root)
        print(json.dumps({'status':'price_snapshot_saved', 'pricing':value['pricing_per_token'],
                          'credits':value.get('credits'), 'key':value.get('key')}))
    else:
        print(json.dumps(report(root, args.query_generations)))


if __name__ == '__main__':
    main()
