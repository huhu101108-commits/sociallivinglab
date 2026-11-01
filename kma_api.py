
import requests
import pandas as pd
from model import normalize

API_URL = (
    "https://apis.data.go.kr/1360000/"
    "AsosDalyInfoService/getWthrDataList"
)

def fetch_asos(
    service_key,
    start="2006-01-01",
    end="2025-12-31",
    progress=None
):
    if not service_key or not service_key.strip():
        raise ValueError("ASOS API 인증키가 없습니다.")

    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    records = []

    years = list(range(
        start_date.year,
        end_date.year + 1
    ))

    for i, year in enumerate(years):
        first = max(
            start_date,
            pd.Timestamp(year, 1, 1)
        )
        last = min(
            end_date,
            pd.Timestamp(year, 12, 31)
        )

        if first > last:
            continue

        params = {
            "ServiceKey": service_key,
            "pageNo": 1,
            "numOfRows": 400,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "DAY",
            "startDt": first.strftime("%Y%m%d"),
            "endDt": last.strftime("%Y%m%d"),
            "stnIds": "143"
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=40
        )
        response.raise_for_status()

        result = response.json().get("response", {})
        header = result.get("header", {})

        if header.get("resultCode") != "00":
            raise ValueError(
                "ASOS API 응답 오류: "
                + str(header.get("resultMsg", "알 수 없음"))
            )

        body = result.get("body", {})
        items = body.get("items", {}) or {}
        rows = items.get("item", [])

        if isinstance(rows, dict):
            rows = [rows]

        records.extend(rows)

        if progress:
            progress((i + 1) / len(years))

    if not records:
        raise ValueError("관측자료가 없습니다.")

    data = pd.DataFrame(records)

    return normalize([data])
