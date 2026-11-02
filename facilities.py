"""Parse user-supplied official facility datasets without inventing sites."""
import io
import pandas as pd

CATEGORIES = ("무더위쉼터","한파쉼터","민방위 대피시설","응급의료기관","기타 안전시설")
def parse_facilities(blob, default_category="기타 안전시설"):
    df=None
    for enc in ("utf-8-sig","cp949","euc-kr"):
        try:
            df=pd.read_csv(io.BytesIO(blob),encoding=enc,low_memory=False)
            break
        except UnicodeDecodeError: continue
    if df is None or df.empty: raise ValueError("시설 CSV가 비어 있습니다.")
    df.columns=df.columns.astype(str).str.strip()
    def pick(*cols): return next((c for c in cols if c in df.columns),None)
    la=pick("위도","위도(도)","lat","latitude","LATITUDE")
    lo=pick("경도","경도(도)","lon","longitude","LONGITUDE")
    nm=pick("시설명","쉼터명칭","쉼터명","무더위쉼터명","이름","name","기관명")
    ad=pick("주소","도로명주소","소재지도로명주소","상세주소","소재지지번주소")
    cat=pick("시설유형","시설종류","분류","유형","category")
    if not all((la,lo,nm)): raise ValueError("시설명·위도·경도 열이 필요합니다.")
    out=pd.DataFrame({"이름":df[nm].fillna("").astype(str).str.strip(),
       "주소":df[ad].fillna("").astype(str) if ad else "",
       "종류":df[cat].fillna(default_category).astype(str) if cat else default_category,
       "lat":pd.to_numeric(df[la],errors="coerce"),
       "lon":pd.to_numeric(df[lo],errors="coerce")})
    out=out.dropna(subset=["lat","lon"])
    out=out[out.lat.between(33,39.5)&out.lon.between(124,132)]
    out=out[out["이름"].ne("")]
    return out.drop_duplicates(["이름","lat","lon","종류"]).reset_index(drop=True)
