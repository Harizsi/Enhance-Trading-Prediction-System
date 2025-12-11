# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Enhanced Trading Prediction", layout="wide")

# -------------------------------
# Helper / cached functions
# -------------------------------
@st.cache_data(show_spinner=False)
def get_market_sentiment(symbol):
    """Fetch market sentiment score (best-effort)."""
    sentiment_score = 0.5
    fear_greed_index = None
    try:
        if any(x in symbol for x in ['BTC', 'ETH', '-USD']):
            resp = requests.get('https://api.alternative.me/fng/', timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                fear_greed_index = int(data['data'][0]['value'])
                sentiment_score = fear_greed_index / 100.0
        else:
            # Placeholder stock sentiment -- could be replaced with a news API integration
            sentiment_score = 0.55
    except Exception:
        pass
    return sentiment_score, fear_greed_index

@st.cache_data(ttl=60*5, show_spinner=False)
def fetch_market_data(symbol, interval, period):
    """Fetch market data via yfinance (cached)."""
    data = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True
    )
    if data is None or data.empty or len(data) < 20:
        raise ValueError(f"Insufficient data for {symbol} with interval={interval} period={period}.")
    df = data.copy()
    # Fill small gaps
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_pct > 0:
        df = df.fillna(method='ffill').fillna(method='bfill')
    return df

# -------------------------------
# Feature engineering
# -------------------------------
def create_advanced_features(df, interval):
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    features_df = pd.DataFrame(index=df.index)

    features_df['returns'] = close.pct_change()
    features_df['log_returns'] = np.log(close / close.shift(1))

    for window in [5, 10, 20]:
        features_df[f'sma_{window}'] = close.rolling(window=window).mean()

    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features_df['rsi'] = 100 - (100 / (1 + rs))

    features_df['volume_ratio'] = volume / volume.rolling(20).mean()
    features_df['high_low_range'] = (high - low) / close
    if ('m' in interval) or ('h' in interval):
        features_df['hour'] = df.index.hour
        features_df['day_of_week'] = df.index.dayofweek

    features_df['close_ratio_20'] = close / close.rolling(20).mean()
    features_df['volatility'] = features_df['returns'].rolling(20).std()

    features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return features_df

# -------------------------------
# Target engineering
# -------------------------------
def create_targets(df, horizon=1, threshold=0.001):
    close = df['Close']
    future_price = close.shift(-horizon)
    current_price = close
    direction = (future_price > current_price * (1 + threshold)).astype(int)
    targets = pd.DataFrame(index=df.index)
    targets['direction'] = direction
    # drop last rows where future is NaN
    targets = targets.dropna()
    return targets

# -------------------------------
# Feature selection & prepare data
# -------------------------------
def select_features(features_df, targets, n_features=20):
    y = targets['direction']
    # align indices
    aligned_idx = features_df.index.intersection(targets.index)
    X = features_df.loc[aligned_idx].fillna(0)
    y = targets.loc[aligned_idx, 'direction']
    if len(X) == 0:
        raise ValueError("No aligned rows between features and targets.")
    selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
    selector.fit(X.values, y.values.ravel())
    mask = selector.get_support()
    selected = X.columns[mask].tolist()
    return features_df[selected], selected

def prepare_data(features_df, targets, test_size=0.2):
    aligned_idx = features_df.index.intersection(targets.index)
    X = features_df.loc[aligned_idx]
    y = targets.loc[aligned_idx, 'direction']
    valid_mask = ~(y.isna() | X.isna().any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.index, X_test.index

# -------------------------------
# Model training + evaluation
# -------------------------------
def train_ensemble_model(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    rf_model.fit(X_train, y_train.values.ravel())
    gb_model.fit(X_train, y_train.values.ravel())
    return {'Random Forest': rf_model, 'Gradient Boosting': gb_model}

def evaluate_models(models, X_test, y_test):
    results = {}
    best_name = None
    best_f1 = -1
    for name, model in models.items():
        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results[name] = {
            'accuracy': acc,
            'f1': f1,
            'confusion_matrix': cm,
            'report': report,
            'model': model,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
    return results, best_name

# -------------------------------
# Generate signals + risk sizing
# -------------------------------
def generate_trading_signals(results, features_df, scaler, horizon=3):
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    latest_features = features_df.tail(horizon).fillna(0)
    latest_scaled = scaler.transform(latest_features)
    predictions = best_model.predict(latest_scaled)
    probabilities = []
    confidence_scores = []
    for i, pred in enumerate(predictions):
        try:
            prob = best_model.predict_proba(latest_scaled[i:i+1])[0]
            confidence = prob[pred] * 100
            probabilities.append(prob)
        except Exception:
            confidence = 50
            probabilities.append([0.5, 0.5])
        confidence_scores.append(confidence)

    signals = []
    for i in range(horizon):
        signals.append({
            'period': i+1,
            'direction': 'UP' if predictions[i] == 1 else 'DOWN',
            'confidence': confidence_scores[i],
            'probability_up': probabilities[i][1] * 100,
            'probability_down': probabilities[i][0] * 100
        })
    return signals, best_model_name

def calculate_position_size(account_balance, risk_per_trade=0.02, stop_loss_pct=0.02):
    risk_amount = account_balance * risk_per_trade
    position_size = risk_amount / stop_loss_pct
    return {
        'account_balance': account_balance,
        'risk_per_trade': risk_per_trade,
        'risk_amount': risk_amount,
        'stop_loss_pct': stop_loss_pct,
        'position_size': position_size,
        'max_position_pct': position_size / account_balance
    }

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📈 Enhanced Trading Prediction System (Streamlit)")

with st.sidebar:
    st.header("Inputs")
    symbol = st.text_input("Symbol", value="AAPL")
    interval_choice = st.selectbox("Timeframe / Interval",
                                   options=['1m', '5m', '15m', '30m', '60m', '1d'],
                                   index=2)
    period_choice = st.selectbox("History Period",
                                 options=['7d', '30d', '90d', '6mo', '1y', '2y'],
                                 index=1)
    horizon = st.slider("Prediction horizon (candles)", min_value=1, max_value=10, value=3)
    test_size = st.slider("Test set fraction", 0.05, 0.5, 0.2, 0.05)
    n_features = st.slider("Max features to select", 3, 50, 20)
    account_balance = st.number_input("Account balance (USD)", value=10000.0, step=100.0)
    run_button = st.button("Run Prediction")

st.markdown("Small note: Intraday intervals (like `1m`) may be restricted by data provider limits. If yfinance returns little/empty data, try a coarser interval or shorter period.")

if run_button:
    try:
        with st.spinner("Fetching market sentiment..."):
            sentiment_score, fng = get_market_sentiment(symbol)
        st.metric("Market Sentiment", f"{sentiment_score:.2f}")
        if fng:
            st.write(f"Fear & Greed Index: {fng}")

        with st.spinner("Downloading market data..."):
            df = fetch_market_data(symbol, interval_choice, period_choice)
        st.success(f"Data fetched: {df.shape[0]} rows")

        # show price chart
        st.subheader("Price chart")
        st.line_chart(df['Close'])

        with st.spinner("Generating features..."):
            features_df = create_advanced_features(df, interval_choice)
        st.write(f"Features: {features_df.shape[1]} columns, {features_df.shape[0]} rows")
        st.dataframe(features_df.tail(5))

        with st.spinner("Creating targets..."):
            targets = create_targets(df, horizon=1)
        st.write("Target distribution:")
        st.bar_chart(targets['direction'].value_counts())

        with st.spinner("Selecting features..."):
            selected_df, selected_features = select_features(features_df, targets, n_features=n_features)
        st.write(f"Selected features ({len(selected_features)}): {selected_features}")
        st.dataframe(selected_df.tail(3))

        with st.spinner("Preparing dataset..."):
            X_train, X_test, y_train, y_test, scaler, idx_train, idx_test = prepare_data(selected_df, targets, test_size=test_size)
        st.write(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

        with st.spinner("Training models..."):
            models = train_ensemble_model(X_train, y_train)

        with st.spinner("Evaluating..."):
            results, best_name = evaluate_models(models, X_test, y_test)

        # Show metrics
        st.subheader("Model evaluation (summary)")
        for name, info in results.items():
            st.markdown(f"**{name}** — Accuracy: {info['accuracy']:.3f}, F1: {info['f1']:.3f}")
            cm = info['confusion_matrix']
            st.write("Confusion matrix:")
            st.write(pd.DataFrame(cm, index=['True DOWN','True UP'], columns=['Pred DOWN','Pred UP']))
            # small classification metrics
            report = info['report']
            st.write(pd.DataFrame(report).transpose().round(3))

        st.success(f"Best model: {best_name}")

        with st.spinner("Generating trading signals..."):
            signals, best_model_name = generate_trading_signals(results, selected_df, scaler, horizon=horizon)

        st.subheader("Trading signals")
        for s in signals:
            st.write(f"Candle {s['period']}: **{s['direction']}** — Confidence: {s['confidence']:.1f}% (P_UP: {s['probability_up']:.1f}%)")

        # Market overview numbers
        last_close = float(df['Close'].iloc[-1])
        last_volume = float(df['Volume'].iloc[-1])
        avg_volume = float(df['Volume'].rolling(20).mean().iloc[-1])

        st.subheader("Market overview")
        st.write({
            "Symbol": symbol,
            "Timeframe": interval_choice,
            "Last Close": f"${last_close:.4f}",
            "Volume (last)": f"{last_volume:,.0f}",
            "Avg Volume (20)": f"{avg_volume:,.0f}",
            "Sentiment": f"{sentiment_score:.2f}"
        })

        # Risk management & recommendation (based on first signal)
        position_info = calculate_position_size(account_balance)
        st.subheader("Risk management")
        st.write({
            "Account balance": f"${position_info['account_balance']:,.2f}",
            "Risk per trade": f"{position_info['risk_per_trade']:.1%}",
            "Risk amount": f"${position_info['risk_amount']:,.2f}",
            "Recommended position size (notional)": f"${position_info['position_size']:,.2f}",
            "Max position % of account": f"{position_info['max_position_pct']:.1%}"
        })

        st.subheader("Trading recommendation (simple rule)")
        if signals:
            first = signals[0]
            entry = last_close
            if first['direction'] == 'UP' and first['confidence'] > 60:
                stop_loss = entry * 0.98
                take_profit = entry * 1.03
                st.success("BUY recommended (rule-based)")
                st.write({
                    "Entry": f"${entry:.4f}",
                    "Stop Loss": f"${stop_loss:.4f} (-2%)",
                    "Take Profit": f"${take_profit:.4f} (+3%)",
                    "Risk/Reward (approx)": "1 : 1.5"
                })
            elif first['direction'] == 'DOWN' and first['confidence'] > 60:
                stop_loss = entry * 1.02
                take_profit = entry * 0.97
                st.warning("SELL recommended (rule-based)")
                st.write({
                    "Entry": f"${entry:.4f}",
                    "Stop Loss": f"${stop_loss:.4f} (+2%)",
                    "Take Profit": f"${take_profit:.4f} (-3%)",
                    "Risk/Reward (approx)": "1 : 1.5"
                })
            else:
                st.info(f"No clear high-confidence signal (top confidence {first['confidence']:.1f}%).")

        # Build a simple report and offer download
        st.subheader("Download report")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines = [
            f"Trading Analysis Report",
            f"Symbol: {symbol}",
            f"Interval: {interval_choice}",
            f"Period: {period_choice}",
            f"Date: {now}",
            f"Best model: {best_model_name if 'best_model_name' in locals() else best_name}",
            ""
        ]
        report_lines.append("Signals:")
        for s in signals:
            report_lines.append(f"Candle {s['period']}: {s['direction']} (Confidence: {s['confidence']:.1f}%)")
        report_text = "\n".join(report_lines)
        st.download_button("Download text report", data=report_text, file_name=f"trading_report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

st.markdown("---")
st.markdown("Disclaimer: This app is for educational/demo purposes only. Not financial advice. Always do your own research.")
