"""Validate official shelter CSVs; never fabricate coordinates or open status."""
import io
import numpy as np
import pandas as pd


def parse_shelters(blob):
    df = None
    for enc in ('utf-8-sig','cp949','euc-kr'):
        try:
            df = pd.read_csv(io.BytesIO(blob),encoding=enc,low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    if df is None or df.empty:
        raise ValueError('쉼터 CSV를 읽을 수 없습니다.')
    df.columns = df.columns.astype(str).str.strip()
    def pick(*names):
        return next((n for n in names if n in df.columns),None)
    lat = pick('위도','위도(도)','lat','latitude','LATITUDE')
    lon = pick('경도','경도(도)','lon','longitude','LONGITUDE')
    name = pick('쉼터명칭','쉼터명','시설명','무더위쉼터명','이름','name')
    address = pick('상세주소','도로명주소','소재지도로명주소','주소','소재지지번주소')
    hours = pick('운영시간','이용가능시간','평일운영시간','운영시간대')
    if not (lat and lon and name):
        raise ValueError('CSV에는 쉼터명(또는 시설명), 위도, 경도 열이 필요합니다.')
    out=pd.DataFrame({'이름':df[name].astype(str).str.strip(),
                      '주소':df[address].fillna('').astype(str) if address else '',
                      '운영정보':df[hours].fillna('운영정보 확인 필요').astype(str) if hours else '운영정보 확인 필요',
                      'lat':pd.to_numeric(df[lat],errors='coerce'),
                      'lon':pd.to_numeric(df[lon],errors='coerce')})
    out=out.dropna(subset=['lat','lon'])
    out=out[out.lat.between(35.5,36.2)&out.lon.between(128.3,128.9)]
    out=out[(out['이름']!='')&(out['이름']!='nan')]
    return out.drop_duplicates(['이름','lat','lon']).reset_index(drop=True)


def distances_km(df,lat,lon):
    lat1=np.radians(float(lat));lon1=np.radians(float(lon))
    lat2=np.radians(df['lat'].astype(float));lon2=np.radians(df['lon'].astype(float))
    a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371.0*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
