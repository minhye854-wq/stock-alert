# -*- coding: utf-8 -*-
"""
한국투자증권(KIS) Open API 기반 - 국내+해외 주식 매수/매도 신호 + 텔레그램 알림
(GitHub Actions 실행용 버전)

[로컬 버전과 다른 점]
1. API 키/토큰을 코드에 직접 넣지 않고 환경변수(GitHub Secrets)에서 읽습니다.
   -> 이 코드는 공개(Public) 저장소에 올려도 안전합니다 (키 값이 코드에 없음).
2. 무한 루프(while True)가 없습니다. GitHub Actions의 스케줄(cron)이
   5분마다 이 스크립트를 새로 실행시켜주는 구조라, 스크립트는 "한 번
   체크하고 끝"나면 됩니다.
3. KIS 접속토큰을 파일에 캐싱하지 않고, 실행할 때마다 새로 발급받습니다.
   (Actions 실행 환경이 매번 초기화되는 데다, 토큰을 저장소에 커밋하면
   민감한 실계좌 접속권한이 깃허브 기록에 남기 때문에 보안상 매번 새로
   발급받는 방식을 사용합니다. 5분 간격이라 재발급 제한에 걸리지 않습니다.)
4. 중복 알림 방지를 위한 signal_state.json 만 저장소에 다시 커밋해서
   다음 실행 때도 "이전에 이미 알림 보낸 신호"를 기억합니다.
   (이 파일에는 민감정보가 없어 커밋해도 안전합니다.)
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# ========================= CONFIG =========================
KIS_APP_KEY = os.environ.get("KIS_APP_KEY", "")
KIS_APP_SECRET = os.environ.get("KIS_APP_SECRET", "")
KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"   # 실전투자

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

WATCHLIST = [
    {"market": "KR", "code": "005930", "name": "삼성전자"},
    {"market": "KR", "code": "000660", "name": "SK하이닉스"},
    {"market": "KR", "code": "340570", "name": "티애엘"},
    {"market": "KR", "code": "047050", "name": "포스코인터내셔널"},
    {"market": "KR", "code": "087010", "name": "펩트론"},
    {"market": "US", "code": "AAPL", "exchange": "NAS", "name": "애플"},
    {"market": "US", "code": "TSLA", "exchange": "NAS", "name": "테슬라"},
]

REQUEST_GAP_SEC = 0.3     # 종목별 API 호출 사이 최소 대기 (초당 호출 제한 보호)
STATE_FILE = "signal_state.json"  # 신호 중복알림 방지용 상태 저장 파일 (저장소에 커밋됨)

SHORT_MA = 5
LONG_MA = 20
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
# ============================================================


def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[경고] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID 환경변수 미설정 - 콘솔에만 출력")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        res = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
        if res.status_code != 200:
            print(f"[텔레그램 전송 실패] status={res.status_code}, body={res.text[:300]}")
    except Exception as e:
        print(f"[텔레그램 전송 실패] {e}")


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception as e:
        print(f"[상태 저장 실패] {e}")


def get_access_token() -> str:
    """접속토큰 발급 (매 실행마다 새로 발급, 캐싱하지 않음 - 보안 목적)"""
    url = f"{KIS_BASE_URL}/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
    }
    res = requests.post(url, json=body, timeout=15)
    res.raise_for_status()
    return res.json()["access_token"]


def get_domestic_daily(token: str, code: str, days: int = 100) -> pd.DataFrame:
    url = f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "FHKST03010100",
        "custtype": "P",
    }
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": start,
        "FID_INPUT_DATE_2": end,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "1",
    }
    res = requests.get(url, headers=headers, params=params, timeout=15)
    res.raise_for_status()
    rows = res.json().get("output2", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[["stck_bsop_date", "stck_clpr"]].rename(
        columns={"stck_bsop_date": "date", "stck_clpr": "close"}
    )
    df["close"] = df["close"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


def get_overseas_daily(token: str, code: str, exchange: str, days: int = 100) -> pd.DataFrame:
    url = f"{KIS_BASE_URL}/uapi/overseas-price/v1/quotations/dailyprice"
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "HHDFS76240000",
        "custtype": "P",
    }
    params = {"AUTH": "", "EXCD": exchange, "SYMB": code, "GUBN": "0", "BYMD": "", "MODP": "1"}
    res = requests.get(url, headers=headers, params=params, timeout=15)
    res.raise_for_status()
    rows = res.json().get("output2", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[["xymd", "clos"]].rename(columns={"xymd": "date", "clos": "close"})
    df["close"] = df["close"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)
    return df.tail(days).reset_index(drop=True)


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


# ===================== 차트 패턴 감지 =====================
PATTERN_TOLERANCE = 0.025
EXTREMA_ORDER = 3


def is_similar(a: float, b: float, tol: float = PATTERN_TOLERANCE) -> bool:
    if a == 0 or b == 0:
        return False
    return abs(a - b) / ((a + b) / 2) < tol


def find_extrema(close: pd.Series, order: int = EXTREMA_ORDER):
    values = close.values
    n = len(values)
    peaks, troughs = [], []
    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        center = values[i]
        if center == window.max() and np.argmax(window) == order:
            peaks.append(i)
        if center == window.min() and np.argmin(window) == order:
            troughs.append(i)
    return peaks, troughs


def detect_chart_patterns(close: pd.Series) -> list:
    results = []
    peaks, troughs = find_extrema(close)
    last_price = close.iloc[-1]

    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if is_similar(close[p1], close[p2]):
            mid = [t for t in troughs if p1 < t < p2]
            if mid and last_price < close[mid[-1]]:
                results.append(("double_top", "🔴 쌍봉(Double Top) - 넥라인 이탈, 하락 매도 신호"))

    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if is_similar(close[t1], close[t2]):
            mid = [p for p in peaks if t1 < p < t2]
            if mid and last_price > close[mid[-1]]:
                results.append(("double_bottom", "🟢 쌍바닥(Double Bottom) - 넥라인 돌파, 상승 매수 신호"))

    if len(peaks) >= 3:
        p1, p2, p3 = peaks[-3], peaks[-2], peaks[-1]
        if is_similar(close[p1], close[p2]) and is_similar(close[p2], close[p3]):
            between = [t for t in troughs if p1 < t < p3]
            if between:
                support = min(close[t] for t in between)
                if last_price < support:
                    results.append(("triple_top", "🔴 삼중천정(Triple Top) - 지지선 이탈, 하락 매도 신호"))

    if len(troughs) >= 3:
        t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
        if is_similar(close[t1], close[t2]) and is_similar(close[t2], close[t3]):
            between = [p for p in peaks if t1 < p < t3]
            if between:
                resistance = max(close[p] for p in between)
                if last_price > resistance:
                    results.append(("triple_bottom", "🟢 트리플 바닥(Triple Bottom) - 저항선 돌파, 상승 매수 신호"))

    if len(troughs) >= 3:
        t1, t2, t3 = troughs[-3], troughs[-2], troughs[-1]
        if close[t2] < close[t1] and close[t2] < close[t3] and is_similar(close[t1], close[t3], tol=0.05):
            neck = [p for p in peaks if t1 < p < t3]
            if neck:
                neckline = max(close[p] for p in neck)
                if last_price > neckline:
                    results.append(("inverse_hs", "🟢 역 머리어깨형(Inverse H&S) - 넥라인 돌파, 강한 상승 매수 신호"))

    if len(close) >= 15:
        pole = close.iloc[-15:-5]
        flag = close.iloc[-5:]
        pole_change = (pole.iloc[-1] - pole.iloc[0]) / pole.iloc[0]
        flag_range = (flag.max() - flag.min()) / flag.mean()
        if pole_change > 0.07 and flag_range < 0.035 and last_price > flag.max():
            results.append(("bull_flag", "🟢 상승 깃발(Bull Flag) - 눌림목 후 돌파, 상승 지속 매수 신호"))
        if pole_change < -0.07 and flag_range < 0.035 and last_price < flag.min():
            results.append(("bear_flag", "🔴 하락 깃발(Bear Flag) - 반등 후 재하락, 매도 신호"))

    if len(peaks) >= 2 and len(troughs) >= 2:
        rp, rt = peaks[-2:], troughs[-2:]
        if close[rp[-1]] > close[rp[0]] and close[rt[-1]] > close[rt[0]]:
            peak_gap = close[rp[-1]] - close[rp[0]]
            trough_gap = close[rt[-1]] - close[rt[0]]
            if trough_gap > peak_gap > 0 and last_price < close[rt[-1]]:
                results.append(("rising_wedge", "🔴 상승 쐐기형(Rising Wedge) - 저점 이탈, 하락 전환 매도 신호"))

    if len(peaks) >= 2 and len(troughs) >= 2:
        rp, rt = peaks[-2:], troughs[-2:]
        if close[rp[-1]] < close[rp[0]] and close[rt[-1]] > close[rt[0]]:
            upper, lower = close[rp[-1]], close[rt[-1]]
            if last_price > upper:
                results.append(("triangle_up", "🟢 삼각수렴 상단 돌파 - 상승 매수 신호"))
            elif last_price < lower:
                results.append(("triangle_down", "🔴 삼각수렴 하단 이탈 - 하락 매도 신호"))

    lookback = close.iloc[-20:]
    box_high, box_low = lookback.max(), lookback.min()
    box_width = (box_high - box_low) / lookback.mean()
    if box_width < 0.08:
        if last_price >= box_high * 0.995:
            results.append(("box_resistance", f"🟢 박스권 저항선({box_high:,.0f}) 부근 - 돌파 시 매수 관찰"))
        elif last_price <= box_low * 1.005:
            results.append(("box_support", f"🔵 박스권 지지선({box_low:,.0f}) 부근 - 반등 매수 관찰"))

    return results


def analyze(token: str, item: dict, state: dict):
    name = item["name"]
    if item["market"] == "KR":
        df = get_domestic_daily(token, item["code"])
    else:
        df = get_overseas_daily(token, item["code"], item["exchange"])

    if df.empty or len(df) < LONG_MA + 2:
        print(f"[{name}] 데이터 부족, 스킵")
        return

    close = df["close"]
    df["MA_SHORT"] = close.rolling(SHORT_MA).mean()
    df["MA_LONG"] = close.rolling(LONG_MA).mean()
    df["RSI"] = calc_rsi(close, RSI_PERIOD)

    prev, last = df.iloc[-2], df.iloc[-1]
    signals = []

    if prev["MA_SHORT"] <= prev["MA_LONG"] and last["MA_SHORT"] > last["MA_LONG"]:
        signals.append(("golden_cross", "📈 골든크로스 발생 (매수 신호 후보)"))
    if prev["MA_SHORT"] >= prev["MA_LONG"] and last["MA_SHORT"] < last["MA_LONG"]:
        signals.append(("dead_cross", "📉 데드크로스 발생 (매도 신호 후보)"))
    if last["RSI"] < RSI_OVERSOLD:
        signals.append(("rsi_oversold", f"🔵 RSI 과매도 ({last['RSI']:.1f}) — 반등 매수 후보"))
    if last["RSI"] > RSI_OVERBOUGHT:
        signals.append(("rsi_overbought", f"🔴 RSI 과매수 ({last['RSI']:.1f}) — 차익실현 매도 후보"))

    signals.extend(detect_chart_patterns(close))

    current_keys = {k for k, _ in signals}
    prev_keys = set(state.get(name, []))
    new_keys = current_keys - prev_keys
    state[name] = list(current_keys)

    if signals:
        print(f"[{name}] 현재가 {last['close']:,.2f}\n" + "\n".join(msg for _, msg in signals))
    else:
        print(f"[{name}] 신호 없음")

    if new_keys:
        new_messages = [msg for k, msg in signals if k in new_keys]
        alert_msg = (
            f"[{name}] 현재가 {last['close']:,.2f}\n(새로 감지된 신호)\n" + "\n".join(new_messages)
        )
        send_telegram(alert_msg)


def main():
    if not KIS_APP_KEY or not KIS_APP_SECRET:
        print("[오류] KIS_APP_KEY / KIS_APP_SECRET 환경변수(Secrets)가 설정되지 않았습니다.")
        sys.exit(1)

    print(f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 체크 시작 ===")
    state = load_state()

    try:
        token = get_access_token()
    except Exception as e:
        print(f"[토큰 발급 실패] {e}")
        sys.exit(1)

    for item in WATCHLIST:
        try:
            analyze(token, item, state)
        except requests.HTTPError as e:
            print(f"[{item['name']}] API 오류: {e} / 응답: {e.response.text[:300]}")
        except Exception as e:
            print(f"[{item['name']}] 오류: {e}")
        time.sleep(REQUEST_GAP_SEC)

    save_state(state)


if __name__ == "__main__":
    main()
