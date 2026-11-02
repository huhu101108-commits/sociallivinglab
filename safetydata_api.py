"""행정안전부 무더위쉼터 API. No hardcoded key, no old shelter CSV fallback."""
import requests
import pandas as pd

URL="https://www.safetydata.go.kr/V2/api/DSSP-IF-10942"
FIELDS={
    "name":("SHLTR_NM","SHELTER_NM","RSTR_NM","FACLT_NM","FCLT_NM","FCLT_NM","SHLTR_NAME","무더위쉼터명","쉼터명","시설명"),
    "address":("RN_DTL_ADRES","DTL_ADRES","ROAD_NM_ADDR","RDNMADR","ROAD_ADDR","ADDR","DTL_ADDR","LOTNO_ADDR","도로명주소","주소","소재지도로명주소"),
    "lat":("LA","LAT","LATITUDE","YCORD","YCRD","Y_COORD","위도","lat","latitude"),
    "lon":("LO","LOT","LON","LONGITUDE","XCORD","XCRD","X_COORD","경도","lon","longitude"),
}
def _find(d,options):
    for key in options:
        for actual,value in d.items():
            if str(actual).lower()==key.lower() and value is not None:
                return value
    return None

def _rows(node):
    if isinstance(node,list):
        if node and all(isinstance(x,dict) for x in node):
            return node
        for x in node:
            found=_rows(x)
            if found:return found
    if isinstance(node,dict):
        for k in ("data","DATA","items","item","list","body","response","result","records"):
            if k in node:
                found=_rows(node[k])
                if found:return found
        for value in node.values():
            if isinstance(value,(dict,list)):
                found=_rows(value)
                if found:return found
    return []

def _error(payload):
    if not isinstance(payload,dict):return None
    for field in ("resultCode","RESULT_CODE","code","status"):
        val=payload.get(field)
        if val is not None and str(val).upper() not in ("0","00","200","SUCCESS","OK"):
            return f"{field}={val}"
    return None

def fetch_shelters(key,rows_per_page=500,max_pages=150):
    if not key:raise ValueError("SAFETYDATA_SERVICE_KEY가 설정되지 않았습니다.")
    all_rows=[]
    session=requests.Session()
    seen=set()
    for page in range(1,max_pages+1):
        r=session.get(URL,params={"serviceKey":key,"pageNo":page,
            "numOfRows":rows_per_page,"returnType":"json"},timeout=25)
        r.raise_for_status()
        try: payload=r.json()
        except ValueError as exc:raise ValueError("API가 JSON을 반환하지 않았습니다. 인증키/응답 형식을 확인하세요.") from exc
        err=_error(payload)
        if err:raise ValueError(f"API 오류: {err}")
        rows=_rows(payload)
        if not rows:
            if page==1:raise ValueError("API 응답에서 시설 목록을 찾지 못했습니다. 응답 구조 확인이 필요합니다.")
            break
        signature=str(rows[:2])+str(rows[-2:])
        if signature in seen:raise ValueError("API가 같은 페이지를 반복 반환했습니다. 페이지 매개변수를 확인하세요.")
        seen.add(signature)
        all_rows.extend(rows)
        if len(rows)<rows_per_page:break
    else:
        raise ValueError("페이지 제한에 도달했습니다. 전국 데이터 수집 완료 여부를 확인하세요.")
    return pd.DataFrame(all_rows)

def _degree(value):
    """Only accept WGS84-like degree values, never silently treat projected X/Y as degrees."""
    try:
        v=float(str(value).strip().replace(",", ""))
        return v if pd.notna(v) else None
    except (TypeError,ValueError):
        return None

def normalize_shelters(raw):
    if raw.empty:raise ValueError("무더위쉼터 API가 빈 목록을 반환했습니다.")
    normalized=[]
    for row in raw.to_dict("records"):
        name=_find(row,FIELDS["name"])
        if name is None or not str(name).strip():continue
        lat=lon=None
        for latkey,lonkey in (("LA","LO"),("LAT","LON"),("LATITUDE","LONGITUDE"),("YCORD","XCORD")):
            la=_degree(_find(row,(latkey,)))
            lo=_degree(_find(row,(lonkey,)))
            if la is not None and lo is not None and 33<=la<=39.5 and 124<=lo<=132:
                lat,lon=la,lo
                break
        if lat is None:continue
        normalized.append({"이름":str(name).strip(),
            "주소":str(_find(row,FIELDS["address"]) or "").strip(),
            "종류":"무더위쉼터","lat":lat,"lon":lon})
    if not normalized:
        preview=[]
        for field in ("RSTR_NM","LA","LO","YCORD","XCORD"):
            if field in raw.columns:
                preview.append(f"{field}={raw[field].head(2).astype(str).tolist()}")
        raise ValueError("좌표를 가진 쉼터를 찾지 못했습니다. 좌표 예시: "+"; ".join(preview)+". 좌표계 확인이 필요합니다.")
    return pd.DataFrame(normalized).drop_duplicates(["이름","lat","lon"]).reset_index(drop=True)
