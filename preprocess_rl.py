import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """1분봉 데이터를 다른 타임프레임으로 리샘플링"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 타임프레임 매핑
    timeframe_map = {
        '1m': '1T',
        '5m': '5T', 
        '15m': '15T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D',
        '1w': '1W'
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")
    
    # 리샘플링
    resampled = df.resample(timeframe_map[timeframe]).agg({
        'Open': 'first',
        'High': 'max', 
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return resampled

def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI 계산"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # 첫 번째 평균
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi = np.full(len(prices), np.nan)
    
    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    
    # 이후 값들 계산
    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX 계산"""
    tr = np.maximum(high[1:] - low[1:], 
                   np.maximum(np.abs(high[1:] - close[:-1]), 
                             np.abs(low[1:] - close[:-1])))
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                      np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                       np.maximum(low[:-1] - low[1:], 0), 0)
    
    # 평활화
    tr_smooth = np.full(len(tr), np.nan)
    plus_dm_smooth = np.full(len(plus_dm), np.nan)
    minus_dm_smooth = np.full(len(minus_dm), np.nan)
    
    if len(tr) >= period:
        tr_smooth[period-1] = np.mean(tr[:period])
        plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
        
        for i in range(period, len(tr)):
            tr_smooth[i] = (tr_smooth[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # DI 계산
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # ADX 계산
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    adx = np.full(len(close), np.nan)
    if len(dx[~np.isnan(dx)]) >= period:
        valid_dx = dx[~np.isnan(dx)]
        adx_start_idx = period + period - 1
        if adx_start_idx < len(adx):
            adx[adx_start_idx] = np.mean(valid_dx[:period])
            
            for i in range(adx_start_idx + 1, len(adx)):
                if adx_start_idx + (i - adx_start_idx) < len(dx) and not np.isnan(dx[adx_start_idx + (i - adx_start_idx)]):
                    adx[i] = (adx[i-1] * (period-1) + dx[adx_start_idx + (i - adx_start_idx)]) / period
    
    return adx

def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """볼린저 밴드 계산"""
    sma = pd.Series(prices).rolling(window=period).mean().values
    std = pd.Series(prices).rolling(window=period).std().values
    
    upper_band = sma + (std * std_mult)
    lower_band = sma - (std * std_mult)
    
    return upper_band, lower_band

def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD 계산"""
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR 계산"""
    tr = np.maximum(high[1:] - low[1:],
                   np.maximum(np.abs(high[1:] - close[:-1]),
                             np.abs(low[1:] - close[:-1])))
    
    atr = np.full(len(close), np.nan)
    if len(tr) >= period:
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(atr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i-1]) / period
    
    return atr

def calculate_volume_profile_windowed(close: np.ndarray, volume: np.ndarray, window_size: int = 52, bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """52스텝 윈도우 기준 볼륨 프로파일 계산 (데이터 누수 방지)"""
    n = len(close)
    current_bin_values = np.zeros(n)
    above_volume = np.zeros(n)
    below_volume = np.zeros(n)
    
    for i in range(n):
        # 현재 시점에서 과거 window_size만큼의 데이터만 사용
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        
        window_close = close[start_idx:end_idx]
        window_volume = volume[start_idx:end_idx]
        
        if len(window_close) < 2:
            # 데이터가 부족한 경우 0으로 설정
            current_bin_values[i] = 0
            above_volume[i] = 0
            below_volume[i] = 0
            continue
        
        # 윈도우 내 가격 범위
        price_min, price_max = np.min(window_close), np.max(window_close)
        
        if price_max <= price_min:
            # 가격 변동이 없는 경우
            current_bin_values[i] = 1.0
            above_volume[i] = 0
            below_volume[i] = 0
            continue
        
        # bin 생성
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        
        # 각 가격에 해당하는 bin 찾기
        bin_indices = np.digitize(window_close, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)
        
        # 각 bin별 볼륨 합계
        volume_profile = np.zeros(bins)
        for j in range(len(window_close)):
            volume_profile[bin_indices[j]] += window_volume[j]
        
        # 민맥스 정규화
        if volume_profile.max() > 0:
            volume_profile = volume_profile / volume_profile.max()
        
        # 현재 가격의 bin 인덱스
        current_price = close[i]
        current_bin_idx = np.digitize([current_price], bin_edges)[0] - 1
        current_bin_idx = np.clip(current_bin_idx, 0, bins - 1)
        
        # 특징 계산
        current_bin_values[i] = volume_profile[current_bin_idx]
        above_volume[i] = volume_profile[current_bin_idx+1:].sum() if current_bin_idx < bins-1 else 0
        below_volume[i] = volume_profile[:current_bin_idx].sum() if current_bin_idx > 0 else 0
    
    return current_bin_values, above_volume, below_volume

def process_single_asset_timeframe(args) -> None:
    """단일 자산의 특정 타임프레임 처리"""
    symbol, timeframe, raw_dir, output_dir = args
    
    print(f"📊 처리 중: {symbol}_{timeframe}")
    
    try:
        # 1분봉 데이터 로드
        csv_path = os.path.join(raw_dir, f'{symbol}_1m.csv')
        if not os.path.exists(csv_path):
            print(f"⚠️ 경고: {csv_path} 파일을 찾을 수 없습니다. 건너<binary data, 2 bytes><binary data, 2 bytes>니다.")
            return
            
        df_1m = pd.read_csv(csv_path)
        
        # 타임프레임 변환
        if timeframe == '1m':
            df = df_1m
        else:
            df = resample_to_timeframe(df_1m, timeframe)
        
        if len(df) < 52:
            print(f"⚠️ 경고: {symbol}_{timeframe} 데이터가 너무 적어 건너<binary data, 2 bytes><binary data, 2 bytes>니다. ({len(df)}개)")
            return

        # 기본 OHLCV
        open_prices = df['Open'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        close_prices = df['Close'].values
        volume = df['Volume'].values
        epsilon = 1e-10

        # 보조지표 계산
        rsi = calculate_rsi(close_prices)
        adx = calculate_adx(high_prices, low_prices, close_prices)
        bb_upper, bb_lower = calculate_bollinger_bands(close_prices)
        macd_line, macd_signal, macd_hist = calculate_macd(close_prices)
        atr = calculate_atr(high_prices, low_prices, close_prices)
        
        # 각 가격 관련 데이터의 첫 값을 기준으로 정규화 (로그 변동률)
        log_open = np.log((open_prices + epsilon) / (open_prices[0] + epsilon))
        log_high = np.log((high_prices + epsilon) / (high_prices[0] + epsilon))
        log_low = np.log((low_prices + epsilon) / (low_prices[0] + epsilon))
        log_close = np.log((close_prices + epsilon) / (close_prices[0] + epsilon))
        
        # 첫 값이 NaN일 수 있는 보조지표 처리
        # NaN이 아닌 첫 번째 유효한 값을 기준점으로 사용
        def get_first_valid_value(arr):
            for val in arr:
                if pd.notna(val) and val != 0:
                    return val
            return 1 # 모든 값이 유효하지 않으면 1로 fallback

        start_bb_upper = get_first_valid_value(bb_upper)
        start_bb_lower = get_first_valid_value(bb_lower)

        log_bb_upper = np.log((bb_upper + epsilon) / (start_bb_upper + epsilon))
        log_bb_lower = np.log((bb_lower + epsilon) / (start_bb_lower + epsilon))
        
        # 가격 기반 지표: 현재 종가로 나누어 정규화
        macd_line_norm = macd_line / (close_prices + epsilon)
        macd_signal_norm = macd_signal / (close_prices + epsilon)
        macd_hist_norm = macd_hist / (close_prices + epsilon)
        atr_norm = atr / (close_prices + epsilon)

        # 볼륨 민맥스 정규화
        volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + epsilon)

        # 볼륨 프로파일
        vp_current, vp_above, vp_below = calculate_volume_profile_windowed(close_prices, volume)
        
        # 모든 특징 결합
        features = np.column_stack([
            log_open,         # 0
            log_high,         # 1 -> 여기가 High
            log_low,          # 2 -> 여기가 Low
            log_close,        # 3
            volume_norm,      # 4
            rsi,              # 5
            adx,              # 6
            log_bb_upper,     # 7
            log_bb_lower,     # 8
            macd_line_norm,   # 9
            macd_signal_norm, # 10
            macd_hist_norm,   # 11
            atr_norm,         # 12
            vp_current,       # 13
            vp_above,         # 14
            vp_below          # 15
        ])
        
        # NaN, Inf 값 처리 (Pandas 사용)
        df_features = pd.DataFrame(features)
        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        for col in df_features.columns:
            first_valid_idx = df_features[col].first_valid_index()
            if first_valid_idx is not None:
                df_features[col].iloc[:first_valid_idx] = df_features[col].loc[first_valid_idx]
        
        df_features.fillna(method='ffill', inplace=True)
        df_features.fillna(0, inplace=True)
        
        features = df_features.values

        # 저장
        output_path = os.path.join(output_dir, f'{symbol}_{timeframe}.npy')
        np.save(output_path, features.astype(np.float32))
        
        print(f"✅ 완료: {symbol}_{timeframe} -> {features.shape}")
        
    except Exception as e:
        import traceback
        print(f"❌ 오류: {symbol}_{timeframe} - {str(e)}")
        print(traceback.format_exc())

def preprocess_crypto_data(raw_dir: str = 'raw', output_dir: str = 'preprocessed', n_processes: int = 4):
    """암호화폐 데이터 전처리 (병렬 처리)"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 자산과 타임프레임 정의
    symbols = ['BTC', 'ETH', 'ADA', 'XRP']
    timeframes = ['1h'] # 1시간봉 데이터만 처리
    
    # 처리할 작업 목록 생성
    tasks = []
    for symbol in symbols:
        for timeframe in timeframes:
            tasks.append((symbol, timeframe, raw_dir, output_dir))
    
    print(f"🚀 총 {len(tasks)}개 작업을 {n_processes}개 프로세스로 처리합니다.")
    print(f"📈 특징 구성: OHLC(relative log), Volume(norm), RSI, ADX, Bollinger(relative log), MACD(norm), ATR(norm), VolumeProfile")
    
    # 병렬 처리
    with Pool(processes=n_processes) as pool:
        pool.map(process_single_asset_timeframe, tasks)
    
    print("🎉 전처리 완료!")
    
    # 결과 요약
    print("\n📊 생성된 파일들:")
    for symbol in symbols:
        for timeframe in timeframes:
            file_path = os.path.join(output_dir, f'{symbol}_{timeframe}.npy')
            if os.path.exists(file_path):
                data = np.load(file_path)
                print(f"  {file_path}: {data.shape} (16 features)")

if __name__ == "__main__":
    preprocess_crypto_data()