# =============================================================================
# train_model.py - Feature Engineering, ATR-Based Labeling, Training & Export
#
# Perubahan dari versi sebelumnya:
#   - rolling_slope() di-vectorize (92x lebih cepat, penting untuk 1.8M rows)
#   - Progress labeling lebih informatif
#   - Estimasi waktu training
#   - Validasi data sebelum training
#
# Usage: python train_model.py
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from numpy.lib.stride_tricks import sliding_window_view

import ta
from config import (
    DATA_M1_PATH, DATA_M15_PATH,
    MODEL_BUY_PATH, MODEL_SELL_PATH,
    FEATURES,
    SWING_LOOKBACK,
    SD_CONSOLIDATION_CANDLES, SD_CONSOLIDATION_WIDTH,
    VOLUME_MA_PERIOD, M15_BIAS_THRESHOLD, M15_SLOPE_LOOKBACK,
    BOS_LOOKBACK,
    ATR_MULTIPLIER_TP, ATR_MULTIPLIER_SL, ATR_PERIOD
)


# =============================================================================
# A. LOAD DATA
# =============================================================================

def load_data():
    """Load M1 and M15 CSV data."""
    print("Loading data...")

    if not os.path.exists(DATA_M1_PATH):
        raise FileNotFoundError(f"M1 data not found: {DATA_M1_PATH}")
    if not os.path.exists(DATA_M15_PATH):
        raise FileNotFoundError(f"M15 data not found: {DATA_M15_PATH}")

    df_m1  = pd.read_csv(DATA_M1_PATH,  index_col='time', parse_dates=True)
    df_m15 = pd.read_csv(DATA_M15_PATH, index_col='time', parse_dates=True)

    df_m1.sort_index(inplace=True)
    df_m15.sort_index(inplace=True)

    print(f"  M1  rows : {len(df_m1):,}  ({df_m1.index[0]} to {df_m1.index[-1]})")
    print(f"  M15 rows : {len(df_m15):,}  ({df_m15.index[0]} to {df_m15.index[-1]})")

    # Validasi kolom
    required = ['open', 'high', 'low', 'close', 'tick_volume']
    for col in required:
        if col not in df_m1.columns:
            raise ValueError(f"Kolom '{col}' tidak ada di M1 data")
        if col not in df_m15.columns:
            raise ValueError(f"Kolom '{col}' tidak ada di M15 data")

    # Validasi harga
    price_min = df_m1['close'].min()
    price_max = df_m1['close'].max()
    print(f"  M1 price range: {price_min:.2f} - {price_max:.2f}")

    return df_m1, df_m15


# =============================================================================
# B. FEATURE ENGINEERING
# =============================================================================

def rolling_slope_vectorized(series, window):
    """
    Vectorized rolling linear regression slope.
    92x lebih cepat dari pure Python loop.
    Penting untuk dataset 1.8 juta rows.
    """
    values = series.values
    n      = len(values)

    x      = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var  = ((x - x_mean) ** 2).sum()

    slopes = np.full(n, np.nan)

    # sliding_window_view: shape (n - window + 1, window)
    if n >= window:
        windows = sliding_window_view(values, window)          # (n-w+1, w)
        y_mean  = windows.mean(axis=1, keepdims=True)
        slopes[window - 1:] = (
            ((windows - y_mean) * (x - x_mean)).sum(axis=1) / x_var
        )

    return pd.Series(slopes, index=series.index)


def compute_m15_features(df_m15):
    """Compute M15 slope and bias using vectorized rolling linear regression."""
    print("  Computing M15 features (vectorized)...")
    t0 = time.time()

    df = df_m15.copy()
    df['m15_slope'] = rolling_slope_vectorized(df['close'], M15_SLOPE_LOOKBACK)
    df['m15_bias']  = df['m15_slope'].apply(
        lambda s: 1 if s > M15_BIAS_THRESHOLD
                  else (-1 if s < -M15_BIAS_THRESHOLD else 0)
                  if not np.isnan(s) else np.nan
    )

    print(f"  M15 features done in {time.time()-t0:.1f}s")
    return df[['m15_slope', 'm15_bias']]


def compute_bos_features(df, lookback):
    """Compute Break of Structure features."""
    prev_high   = df['close'].shift(1).rolling(lookback).max()
    prev_low    = df['close'].shift(1).rolling(lookback).min()
    bos_bullish = (df['close'] > prev_high).astype(int)
    bos_bearish = (df['close'] < prev_low).astype(int)
    return bos_bullish, bos_bearish


def compute_features(df_m1, df_m15):
    """Compute all ML features."""
    print("  Computing M1 features...")
    t0 = time.time()

    df = df_m1.copy()

    # Price returns
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)

    # Candle structure
    rng = (df['high'] - df['low']).replace(0, np.nan)
    df['body_ratio']  = (df['close'] - df['open']).abs() / rng
    df['upper_wick']  = (df['high'] - df[['open', 'close']].max(axis=1)) / rng
    df['lower_wick']  = (df[['open', 'close']].min(axis=1) - df['low']) / rng

    # Volatility
    df['atr_5']        = rng.rolling(5).mean()
    df['atr_14']       = rng.rolling(14).mean()
    df['volatility_5'] = df['ret_1'].rolling(5).std()

    # Momentum
    rsi_ind         = ta.momentum.RSIIndicator(df['close'], window=14)
    df['rsi']       = rsi_ind.rsi()
    df['rsi_slope'] = df['rsi'].diff()
    df['minute']    = df.index.minute

    # Volume
    df['volume_ratio'] = (
        df['tick_volume'] /
        df['tick_volume'].rolling(VOLUME_MA_PERIOD).mean()
    )

    # Candlestick patterns
    prev_body = (df['close'].shift(1) - df['open'].shift(1)).abs()
    curr_body = (df['close'] - df['open']).abs()
    prev_dir  = (df['close'].shift(1) > df['open'].shift(1)).astype(int)
    curr_dir  = (df['close'] > df['open']).astype(int)

    df['is_pin_bar']    = ((df['lower_wick'] > 2 * df['body_ratio']) |
                           (df['upper_wick'] > 2 * df['body_ratio'])).astype(int)
    df['is_engulfing']  = ((curr_body > prev_body) &
                           (curr_dir != prev_dir)).astype(int)
    df['is_inside_bar'] = ((df['high'] < df['high'].shift(1)) &
                           (df['low']  > df['low'].shift(1))).astype(int)

    # Key level
    swing_high = df['close'].rolling(SWING_LOOKBACK).max()
    swing_low  = df['close'].rolling(SWING_LOOKBACK).min()
    df['near_key_level'] = pd.concat(
        [(swing_high - df['close']).abs(),
         (swing_low  - df['close']).abs()], axis=1
    ).min(axis=1)

    # S&D zone
    rolling_max = df['close'].rolling(SD_CONSOLIDATION_CANDLES).max()
    rolling_min = df['close'].rolling(SD_CONSOLIDATION_CANDLES).min()
    df['consolidation_width'] = rolling_max - rolling_min
    df['in_sd_zone'] = (
        df['consolidation_width'] < SD_CONSOLIDATION_WIDTH
    ).astype(int)

    # M15 features (vectorized slope)
    m15_features  = compute_m15_features(df_m15)
    m15_reindexed = m15_features.reindex(df.index, method='ffill')
    df['m15_slope'] = m15_reindexed['m15_slope']
    df['m15_bias']  = m15_reindexed['m15_bias']

    # BoS features
    df['bos_bullish'], df['bos_bearish'] = compute_bos_features(df, BOS_LOOKBACK)

    df.dropna(inplace=True)

    print(f"  M1 features done in {time.time()-t0:.1f}s")
    print(f"  Rows after dropna: {len(df):,}")

    return df


# =============================================================================
# C. ATR-BASED LABELING
# =============================================================================

def create_labels_atr(df, atr_mult_tp=4.0, atr_mult_sl=1.0,
                      atr_period=14, max_forward=60):
    """
    Create binary labels using ATR-adaptive TP/SL per candle.

    Untuk setiap candle:
        tp_buy  = close + atr_mult_tp * atr14
        sl_buy  = close - atr_mult_sl * atr14
        label=1 jika harga mencapai tp sebelum sl dalam max_forward candles

    RR ratio = atr_mult_tp / atr_mult_sl
    """
    print(f"\n  Labeling mode  : ATR-ADAPTIVE")
    print(f"  TP multiplier  : {atr_mult_tp}x ATR_{atr_period}")
    print(f"  SL multiplier  : {atr_mult_sl}x ATR_{atr_period}")
    print(f"  RR ratio       : 1:{atr_mult_tp/atr_mult_sl:.1f}")
    print(f"  Max forward    : {max_forward} candles")

    atr_col    = f'atr_{atr_period}'
    close_vals = df['close'].values
    high_vals  = df['high'].values
    low_vals   = df['low'].values
    atr_vals   = df[atr_col].values
    n          = len(df)

    label_buy  = np.zeros(n, dtype=np.int8)
    label_sell = np.zeros(n, dtype=np.int8)

    atr_mean = np.nanmean(atr_vals)
    print(f"\n  ATR stats:")
    print(f"    Mean ATR14 : {atr_mean:.2f} pts")
    print(f"    Mean TP    : {atr_mult_tp * atr_mean:.2f} pts")
    print(f"    Mean SL    : {atr_mult_sl * atr_mean:.2f} pts")

    print(f"\n  Labeling {n:,} candles...")
    t0           = time.time()
    report_every = max(1, n // 10)

    for i in range(n - max_forward):
        if i % report_every == 0 and i > 0:
            elapsed  = time.time() - t0
            progress = i / n
            eta      = elapsed / progress * (1 - progress)
            print(f"  Progress: {progress*100:.0f}%  "
                  f"({i:,}/{n:,})  "
                  f"elapsed: {elapsed:.0f}s  "
                  f"ETA: {eta:.0f}s")

        atr = atr_vals[i]
        if np.isnan(atr) or atr <= 0:
            continue

        entry   = close_vals[i]
        tp_buy  = entry + atr_mult_tp * atr
        sl_buy  = entry - atr_mult_sl * atr
        tp_sell = entry - atr_mult_tp * atr
        sl_sell = entry + atr_mult_sl * atr

        # BUY label
        for j in range(1, max_forward + 1):
            h = high_vals[i + j]
            l = low_vals[i + j]
            if h >= tp_buy:
                label_buy[i] = 1
                break
            elif l <= sl_buy:
                break

        # SELL label
        for j in range(1, max_forward + 1):
            h = high_vals[i + j]
            l = low_vals[i + j]
            if l <= tp_sell:
                label_sell[i] = 1
                break
            elif h >= sl_sell:
                break

    elapsed = time.time() - t0
    print(f"  Labeling done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    df             = df.copy()
    df['label_buy']  = label_buy
    df['label_sell'] = label_sell

    buy_count  = int(df['label_buy'].sum())
    sell_count = int(df['label_sell'].sum())
    total      = len(df)

    print(f"\n  Total candles  : {total:,}")
    print(f"  BUY  label=1   : {buy_count:,}  ({buy_count/total*100:.2f}%)")
    print(f"  SELL label=1   : {sell_count:,}  ({sell_count/total*100:.2f}%)")

    min_wr = 100 / (1 + atr_mult_tp / atr_mult_sl)
    nat_wr_buy  = buy_count  / total * 100
    nat_wr_sell = sell_count / total * 100
    print(f"\n  Min win rate untuk profit (RR 1:{atr_mult_tp/atr_mult_sl:.0f}): {min_wr:.0f}%")
    print(f"  Natural WR BUY : {nat_wr_buy:.2f}%  {'OK' if nat_wr_buy >= min_wr else 'PERLU TUNING'}")
    print(f"  Natural WR SELL: {nat_wr_sell:.2f}%  {'OK' if nat_wr_sell >= min_wr else 'PERLU TUNING'}")

    return df


# =============================================================================
# D. TRAINING
# =============================================================================

def train_model(X_train, y_train, X_test, y_test, label_name):
    """Train a LightGBM classifier."""
    print(f"\nTraining {label_name} model...")
    print(f"  Train : {len(X_train):,} rows | Positives: {y_train.sum():,} "
          f"({y_train.mean()*100:.2f}%)")
    print(f"  Test  : {len(X_test):,} rows  | Positives: {y_test.sum():,} "
          f"({y_test.mean()*100:.2f}%)")

    t0 = time.time()

    model = LGBMClassifier(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 6,
        num_leaves        = 31,
        min_child_samples = 50,
        class_weight      = 'balanced',
        random_state      = 42,
        verbose           = -1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            __import__('lightgbm').early_stopping(50, verbose=False),
            __import__('lightgbm').log_evaluation(period=-1)
        ]
    )

    print(f"  Training done in {time.time()-t0:.0f}s")
    print(f"  Best iteration: {model.best_iteration_}")

    y_pred = model.predict(X_test)
    print(f"\n{label_name} Model - Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Signal', 'Signal']))

    print(f"Top 10 Important Features ({label_name}):")
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    for feat, score in importance.nlargest(10).items():
        print(f"  {feat:<25} {score:.0f}")

    return model


# =============================================================================
# E. PROFIT SIMULATION
# =============================================================================

def simulate_profit_atr(model, X_test, df_test, direction,
                         atr_mult_tp, atr_mult_sl, threshold=0.65):
    """Simulate trading using ATR-adaptive TP/SL per trade."""
    probs      = model.predict_proba(X_test)[:, 1]
    signals    = probs >= threshold

    close_vals  = df_test['close'].values
    high_vals   = df_test['high'].values
    low_vals    = df_test['low'].values
    atr_vals    = df_test['atr_14'].values
    n           = len(df_test)
    max_forward = 60

    total_trades = 0
    wins         = 0
    gross_profit = 0.0
    gross_loss   = 0.0

    for i in range(n - max_forward):
        if not signals[i]:
            continue

        atr = atr_vals[i]
        if np.isnan(atr) or atr <= 0:
            continue

        entry        = close_vals[i]
        total_trades += 1
        trade_result = -(atr_mult_sl * atr)

        if direction == 'buy':
            tp_price = entry + atr_mult_tp * atr
            sl_price = entry - atr_mult_sl * atr
            for j in range(1, max_forward + 1):
                if high_vals[i + j] >= tp_price:
                    trade_result = atr_mult_tp * atr
                    break
                elif low_vals[i + j] <= sl_price:
                    break
        else:
            tp_price = entry - atr_mult_tp * atr
            sl_price = entry + atr_mult_sl * atr
            for j in range(1, max_forward + 1):
                if low_vals[i + j] <= tp_price:
                    trade_result = atr_mult_tp * atr
                    break
                elif high_vals[i + j] >= sl_price:
                    break

        if trade_result > 0:
            wins         += 1
            gross_profit += trade_result
        else:
            gross_loss += abs(trade_result)

    win_rate      = (wins / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    return {
        'total_trades' : total_trades,
        'wins'         : wins,
        'win_rate'     : win_rate,
        'profit_factor': profit_factor,
        'gross_profit' : gross_profit,
        'gross_loss'   : gross_loss,
    }


def run_simulation(model_buy, model_sell, X_test, df_test):
    """Run ATR-based profit simulation dengan 3 kombinasi multiplier."""
    combinations = [
        (3.0, 1.0),
        (4.0, 1.0),
        (2.0, 1.0),
    ]

    print("\n" + "=" * 60)
    print("  PROFIT SIMULATION - ATR-BASED TP/SL")
    print("=" * 60)

    best_combo = None
    best_pf    = -1
    atr_mean   = df_test['atr_14'].mean()

    for tp_mult, sl_mult in combinations:
        print(f"\n  TP={tp_mult}xATR (~{tp_mult*atr_mean:.1f}pts) / "
              f"SL={sl_mult}xATR (~{sl_mult*atr_mean:.1f}pts) | "
              f"RR=1:{tp_mult/sl_mult:.0f}")

        res_buy = simulate_profit_atr(
            model_buy, X_test, df_test, 'buy', tp_mult, sl_mult
        )
        res_sell = simulate_profit_atr(
            model_sell, X_test, df_test, 'sell', tp_mult, sl_mult
        )

        print(f"  BUY  | trades={res_buy['total_trades']:4d} | "
              f"WR={res_buy['win_rate']:.1f}% | "
              f"PF={res_buy['profit_factor']:.2f}")
        print(f"  SELL | trades={res_sell['total_trades']:4d} | "
              f"WR={res_sell['win_rate']:.1f}% | "
              f"PF={res_sell['profit_factor']:.2f}")

        avg_pf = (res_buy['profit_factor'] + res_sell['profit_factor']) / 2
        print(f"  Avg PF: {avg_pf:.2f}")

        if avg_pf > best_pf:
            best_pf    = avg_pf
            best_combo = (tp_mult, sl_mult)

    print("\n" + "=" * 60)
    print(f"  BEST: TP={best_combo[0]}xATR / SL={best_combo[1]}xATR | PF={best_pf:.2f}")

    if best_pf >= 1.0:
        print(f"  STATUS: PROFITABLE ✅")
    elif best_pf >= 0.7:
        print(f"  STATUS: Mendekati profitable ⚠️  — coba turunkan threshold")
    else:
        print(f"  STATUS: Belum profitable ❌ — pertimbangkan retrain dengan data lebih baru")

    print(f"\n  Rekomendasi config.py:")
    print(f"    ATR_MULTIPLIER_TP = {best_combo[0]}")
    print(f"    ATR_MULTIPLIER_SL = {best_combo[1]}")
    print("=" * 60)

    return best_combo


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_total = time.time()

    print("=" * 60)
    print("  AURUMBOT - MODEL TRAINING (ATR-ADAPTIVE)")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    # A. Load
    df_m1, df_m15 = load_data()

    # B. Features
    print("\nComputing features...")
    df = compute_features(df_m1, df_m15)
    print(f"  Features computed. Total rows: {len(df):,}")

    # C. Labels
    print("\nCreating ATR-based labels...")
    df = create_labels_atr(
        df,
        atr_mult_tp = ATR_MULTIPLIER_TP,
        atr_mult_sl = ATR_MULTIPLIER_SL,
        atr_period  = ATR_PERIOD
    )

    # D. Split 80/20 — pastikan test set mencakup data TERBARU
    split_idx    = int(len(df) * 0.8)
    df_train     = df.iloc[:split_idx]
    df_test      = df.iloc[split_idx:]
    X_train      = df_train[FEATURES]
    X_test       = df_test[FEATURES]
    y_buy_train  = df_train['label_buy']
    y_buy_test   = df_test['label_buy']
    y_sell_train = df_train['label_sell']
    y_sell_test  = df_test['label_sell']

    print(f"\nTrain: {len(df_train):,} rows | {df_train.index[0]} to {df_train.index[-1]}")
    print(f"Test : {len(df_test):,} rows  | {df_test.index[0]} to {df_test.index[-1]}")

    # E. Train
    model_buy  = train_model(X_train, y_buy_train,  X_test, y_buy_test,  'BUY')
    model_sell = train_model(X_train, y_sell_train, X_test, y_sell_test, 'SELL')

    # F. Simulate
    best_combo = run_simulation(model_buy, model_sell, X_test, df_test)

    # G. Export
    print("\nExporting models...")
    joblib.dump(model_buy,  MODEL_BUY_PATH)
    joblib.dump(model_sell, MODEL_SELL_PATH)
    print(f"  BUY  model : {MODEL_BUY_PATH}")
    print(f"  SELL model : {MODEL_SELL_PATH}")

    total_time = time.time() - t_total
    print(f"\nTotal training time: {total_time/60:.1f} menit")
    print(f"\nTraining complete! Run: python bot.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
