"""행정안전부 한파쉼터 공식 API DSSP-IF-10804. 별도 발급된 키 필요."""
import re
import pandas as pd
import requests
from safetydata_api import _rows, _error

URL = 'https://www.safetydata.go.kr/V2/api/DSSP-IF-10804'

def _float(v):
    try:
        x = float(str(v).strip().replace(',', ''))
        return x if pd.notna(x) else None
    except (TypeError, ValueError):
        return None

def _total(payload):
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in ('totalcount', 'total_count', 'totalcnt'):
                try: return int(v)
                except (ValueError, TypeError): pass
            if isinstance(v, (dict, list)):
                t = _total(v)
                if t is not None: return t
    elif isinstance(payload, list):
        for v in payload[:3]:
            t = _total(v)
            if t is not None: return t
    return None

def fetch_cold_shelters(key, rows_per_page=1000, max_pages=100):
    if not key: raise ValueError('SAFETYDATA_COLD_SERVICE_KEY가 없습니다.')
    results = []
    seen = set()
    with requests.Session() as session:
        for page in range(1, max_pages + 1):
            resp = session.get(URL, params={'serviceKey': key, 'pageNo': page,
                               'numOfRows': rows_per_page, 'returnType': 'json'}, timeout=45)
            # 플랫폼 인증 오류는 HTTP 200에 XML로 올 수 있음
            if 'xml' in resp.headers.get('Content-Type', '').lower() or '<OpenAPI' in resp.text[:300]:
                code = re.search(r'<returnReasonCode>\s*(\d+)\s*</returnReasonCode>', resp.text)
                raise ValueError(f'API 인증/호출 오류 코드 {code.group(1) if code else "알 수 없음"}. 한파쉼터 전용 키와 승인 여부를 확인하세요.')
            resp.raise_for_status()
            try: payload = resp.json()
            except ValueError as exc: raise ValueError('한파쉼터 API가 JSON을 반환하지 않았습니다.') from exc
            err = _error(payload)
            if err: raise ValueError(f'한파쉼터 API 오류: {err}')
            rows = _rows(payload)
            if not rows:
                if page == 1: raise ValueError('한파쉼터 API 응답에서 목록을 찾지 못했습니다.')
                break
            signature = repr((rows[0], rows[-1]))
            if signature in seen: raise ValueError('API가 같은 페이지를 반복했습니다. 페이지 매개변수를 확인하세요.')
            seen.add(signature)
            results.extend(rows)
            total = _total(payload)
            if total is not None and len(results) >= total: break
            if len(rows) < rows_per_page: break
        else:
            raise ValueError('전국 한파쉼터 페이지 제한에 도달했습니다. 수집 완료 여부를 확인하세요.')
    return pd.DataFrame(results)

def normalize_cold_shelters(raw):
    if raw.empty: raise ValueError('한파쉼터 API 응답이 비어 있습니다.')
    expected = {'REARE_NM', 'LAT', 'LOT'}
    if not expected.issubset(raw.columns):
        raise ValueError('한파쉼터 공식 필드가 없습니다. 실제 필드: ' + ', '.join(map(str, raw.columns[:25])))
    out = []
    for row in raw.to_dict('records'):
        name = str(row.get('REARE_NM') or '').strip()
        lat, lon = _float(row.get('LAT')), _float(row.get('LOT'))
        if not name or lat is None or lon is None or not (33 <= lat <= 39.5 and 124 <= lon <= 132):
            continue
        address = str(row.get('RONA_DADDR') or row.get('DADDR') or '').strip()
        out.append({'이름': name, '주소': address, '종류': '한파쉼터', 'lat': lat, 'lon': lon})
    if not out:
        preview = raw[['REARE_NM', 'LAT', 'LOT']].head(2).to_dict('records')
        raise ValueError(f'유효한 한파쉼터 좌표가 없습니다. 예시: {preview}')
    return pd.DataFrame(out).drop_duplicates(['이름', 'lat', 'lon']).reset_index(drop=True)
