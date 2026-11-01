"""KMA official village forecast; separate from the experimental ML model."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
import pandas as pd

KST=ZoneInfo('Asia/Seoul')
URL='https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
# KMA Lambert grid transform from latitude/longitude.
def grid(lat=35.879,lon=128.602):
    import math
    re=6371.00877/5.0; slat1=math.radians(30); slat2=math.radians(60)
    olon=math.radians(126); olat=math.radians(38)
    sn=math.log(math.cos(slat1)/math.cos(slat2))/math.log(math.tan(math.pi*.25+slat2*.5)/math.tan(math.pi*.25+slat1*.5))
    sf=math.tan(math.pi*.25+slat1*.5)**sn*math.cos(slat1)/sn
    ro=re*sf/math.tan(math.pi*.25+olat*.5)**sn
    ra=re*sf/math.tan(math.pi*.25+math.radians(lat)*.5)**sn
    theta=math.radians(lon)-olon
    if theta>math.pi:theta-=2*math.pi
    if theta< -math.pi:theta+=2*math.pi
    theta*=sn
    return int(ra*math.sin(theta)+43+.5),int(ro-ra*math.cos(theta)+136+.5)

def latest_base(now=None):
    now=now or datetime.now(KST)
    # Published at 02,05,08,11,14,17,20,23; allow 20 minutes for availability.
    for offset in range(2):
        day=(now-timedelta(days=offset)).date()
        for hour in (23,20,17,14,11,8,5,2):
            base=datetime(day.year,day.month,day.day,hour,tzinfo=KST)
            if base+timedelta(minutes=20)<=now:
                return base
    raise ValueError('예보 발표 시각을 계산할 수 없습니다.')

def fetch_forecast(key):
    if not key: raise ValueError('기상청 단기예보 API 인증키가 없습니다.')
    base=latest_base();nx,ny=grid()
    params={'serviceKey':key,'pageNo':1,'numOfRows':1000,'dataType':'JSON',
            'base_date':base.strftime('%Y%m%d'),'base_time':base.strftime('%H%M'),'nx':nx,'ny':ny}
    resp=requests.get(URL,params=params,timeout=20);resp.raise_for_status()
    try: payload=resp.json()
    except ValueError: raise ValueError('API가 JSON 대신 오류 메시지를 반환했습니다. 인증키와 승인 상태를 확인하세요.')
    header=payload.get('response',{}).get('header',{})
    if str(header.get('resultCode'))!='00':raise ValueError(str(header.get('resultMsg','API 오류')))
    body=payload['response']['body']; items=body.get('items',{}).get('item',[])
    if not items:raise ValueError('조회된 예보 항목이 없습니다.')
    df=pd.DataFrame(items)
    df['date']=pd.to_datetime(df['fcstDate']+df['fcstTime'],format='%Y%m%d%H%M')
    p=df.pivot_table(index='date',columns='category',values='fcstValue',aggfunc='first').sort_index()
    for k in ('TMP','TMX','REH','POP'):
        if k in p:p[k]=pd.to_numeric(p[k],errors='coerce')
    return p,base,(nx,ny)
