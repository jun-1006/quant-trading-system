import streamlit as st
import streamlit_authenticator as stauth
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 1. 網頁基本設定 (必須在最頂部) ===
st.set_page_config(page_title="量化技術分析決策系統", page_icon="📈", layout="centered")

# === 2. 帳號與安全設定 ===
config = {
    'credentials': {
        'usernames': {
            'eric': {
                'name': 'Eric',
                'password': stauth.Hasher.hash('123') 
            }
        }
    },
    'cookie': {
        'name': 'some_cookie_name',
        'key': 'some_signature_key',
        'expiry_days': 30
    }
}

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login()

if st.session_state.get("authentication_status") is False:
    st.error('帳號或密碼錯誤')
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning('請輸入帳號與密碼')
    st.stop()

# 登入成功後
st.sidebar.write(f'歡迎回來, *{st.session_state["name"]}*')
authenticator.logout('登出', 'sidebar')
st.sidebar.divider()

# === 3. 初始化 Session State ===
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'summary' not in st.session_state:
    st.session_state.summary = {}

# === 4. 左側邊欄 (風險參數設定區) ===
with st.sidebar:
    st.header("⚙️ 風險管理參數設定")
    total_capital = st.number_input("💵 總可用資金:", min_value=500, max_value=10000000, value=100000, step=100)
    risk_percent = st.slider("⚠️ 單筆最大容許虧損 (%):", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    reward_risk_ratio = st.slider("⚖️ 預期風報比 (停利/停損):", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    st.info(f"💡 單筆最大容許虧損：**${total_capital * (risk_percent/100):,.0f}**")

# === 5. 主畫面 ===
st.title("📈 量化技術分析決策系統")
st.markdown("本系統整合多因子綜合評分，提供客觀的交易決策與白盒化的邏輯解析。")
st.divider()

ticker_symbol = st.text_input("請輸入股票或 ETF 代碼 (台股加 .TW，如 0050.TW；美股如 VOO):", "0050.TW")

if st.button("🚀 執行量化分析", type="primary"):
    with st.spinner(f"正在從交易所獲取 {ticker_symbol} 最新數據與繪製圖表..."):
        stock_data = yf.download(ticker_symbol, period="1y", progress=False)
        
        if stock_data.empty:
            st.error("無法獲取數據，請檢查股票代碼是否正確。")
            st.session_state.analyzed = False
        else:
            # 清洗欄位索引
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)
            else:
                stock_data.columns = stock_data.columns.astype(str)
            stock_data.columns = stock_data.columns.str.strip()
            
            df = stock_data[['Open', 'High', 'Low', 'Close']].copy()
            
            # --- 核心指標計算區 ---
            df['20MA'] = df['Close'].rolling(window=20).mean()
            
            # RSI
            delta = df['Close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            df['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))
            
            # MACD
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
            
            # 布林通道
            std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['20MA'] + (2 * std)
            df['BB_Lower'] = df['20MA'] - (2 * std)
            
            # ATR
            df['High-Low'] = df['High'] - df['Low']
            df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
            df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
            df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            # KD 指標 (包含防錯機制)
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            denom = (high_14 - low_14).replace(0, 1e-9)
            rsv = (df['Close'] - low_14) / denom * 100
            df['K'] = rsv.ewm(com=2, adjust=False).mean()
            df['D'] = df['K'].ewm(com=2, adjust=False).mean()
            
            # --- 終極防護：取代 dropna，改用向下/向上填補，確保長度與欄位完整 ---
            # 新的寫法（支援 Pandas 2.x+ 新版本）
            df = df.bfill().fillna(0)
            
            # --- 獨立計分系統 ---
            df['Score_MA'] = np.where(df['Close'] > df['20MA'], 1, np.where(df['Close'] < df['20MA'], -1, 0))
            df['Score_MACD'] = np.where(df['MACD_Hist'] > 0, 1, np.where(df['MACD_Hist'] < 0, -1, 0))
            df['Score_RSI'] = np.where(df['RSI'] < 40, 1, np.where(df['RSI'] > 75, -1, 0))
            df['Score_BB'] = np.where(df['Close'] <= df['BB_Lower'] * 1.01, 1, np.where(df['Close'] >= df['BB_Upper'] * 0.99, -1, 0))
            df['Score_KD'] = np.where(df['K'] > df['D'], 1, np.where(df['K'] < df['D'], -1, 0))
            
            # 總分系統擴充為五位專家（總分範圍變為 -5 到 +5）
            df['Score'] = df['Score_MA'] + df['Score_MACD'] + df['Score_RSI'] + df['Score_BB'] + df['Score_KD']
            
            
            df['Action'] = 'Hold'
            df.loc[df['Score'] >= 2, 'Action'] = 'Buy'
            df.loc[df['Score'] <= -2, 'Action'] = 'Sell'

            # 存入 Session State
            st.session_state.df = df
            st.session_state.summary = {
                'latest_date': df.index[-1].strftime('%Y-%m-%d'),
                'latest_close': float(df['Close'].iloc[-1]),
                'latest_score': int(df['Score'].iloc[-1]),
                'score_ma': int(df['Score_MA'].iloc[-1]),
                'score_macd': int(df['Score_MACD'].iloc[-1]),
                'score_rsi': int(df['Score_RSI'].iloc[-1]),
                'score_bb': int(df['Score_BB'].iloc[-1]),
                'score_kd': int(df['Score_KD'].iloc[-1]),
                'latest_action': df['Action'].iloc[-1],
                'latest_atr': float(df['ATR'].iloc[-1]),
                'val_ma': float(df['20MA'].iloc[-1]),
                'val_rsi': float(df['RSI'].iloc[-1]),
                'val_macd': float(df['MACD_Hist'].iloc[-1]),
                'val_bbu': float(df['BB_Upper'].iloc[-1]),
                'val_bbl': float(df['BB_Lower'].iloc[-1]),
                'val_k': float(df['K'].iloc[-1]),
                'val_d': float(df['D'].iloc[-1])
            }
            st.session_state.analyzed = True

# === 6. 數據渲染介面 ===
if st.session_state.analyzed and st.session_state.df is not None:
    df = st.session_state.df
    summary = st.session_state.summary
    score = summary['latest_score']
    
    st.success(f"系統分析完成 | 數據更新日期：{summary['latest_date']}")
    score_color = "🟢 強勢偏多" if score >= 2 else "🔴 弱勢偏空" if score <= -2 else "🟡 震盪觀望"
        
    sc1, sc2 = st.columns([1, 2])
    sc1.metric("綜合評分 (-5 到 +5)", f"{score} 分")
    sc2.info(f"**市場狀態判定：** {score_color}")
    
    tab1, tab2 = st.tabs(["📊 決策與動態線圖", "🔍 綜合評分明細解析"])
    
    with tab1:
        st.markdown("### 🤖 系統策略操作建議")
        if summary['latest_action'] == 'Buy':
            st.success(f"🟢 **建議：買入 (Buy)** —— 多項技術指標產生共振，具備進場優勢。")
            st.markdown("#### 🛡️ 風險與資金配置計畫")
            stop_loss_price = summary['latest_close'] - (1.5 * summary['latest_atr'])
            take_profit_price = summary['latest_close'] + (1.5 * summary['latest_atr'] * reward_risk_ratio)
            risk_per_share = summary['latest_close'] - stop_loss_price
            
            # 防呆：避免風險為 0 或負數
            if risk_per_share <= 0: risk_per_share = 0.01 
            
            max_loss_amount = total_capital * (risk_percent / 100)
            suggested_shares = max_loss_amount / risk_per_share
            total_investment = suggested_shares * summary['latest_close']
            
            rc1, rc2, rc3 = st.columns(3)
            rc1.warning(f"**🛑 建議停損價:**\n\n {stop_loss_price:.2f}")
            rc2.info(f"**🎯 預期停利價:**\n\n {take_profit_price:.2f}")
            rc3.success(f"**🛒 建議買進股數:**\n\n 約 {int(suggested_shares):,} 股")
            st.caption(f"*資金估算：此部位將佔用資金約 ${total_investment:,.0f}，預期最大虧損控制在 ${max_loss_amount:,.0f} 以內。*")
        elif summary['latest_action'] == 'Sell':
            st.error(f"🔴 **建議：賣出 (Sell)** —— 技術面出現多重破綻，建議減碼或出場。")
        else:
            st.info(f"🟡 **建議：繼續持有 / 空手觀望 (Hold)** —— 目前價格處於常態波動區間，未達觸發閾值。")
            
        st.divider()

        # --- Plotly 互動式技術線圖 ---
        st.markdown("#### 📈 互動式技術線圖 (近半年走勢)")
        
        plot_df = df.tail(120)
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            subplot_titles=('價格與通道', 'RSI', 'KD 指標'),
                            row_width=[0.3, 0.2, 0.5])

        # 第一層 (價格與通道)
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Upper'], line=dict(color='gray', dash='dash'), name='上軌'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Lower'], line=dict(color='gray', dash='dash'), name='下軌'), row=1, col=1)

        # 第二層 (RSI)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=75, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=40, line_dash="dot", line_color="green", row=2, col=1)

        # 第三層 (KD)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['K'], name='K值', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['D'], name='D值', line=dict(color='orange')), row=3, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 交易指南 ---
        with st.expander("📖 不知道怎麼看這張圖？點擊查看 RSI 圖表閱讀指南"):
            st.markdown("""
            **RSI (相對強弱指標)** 是一個用來衡量近期價格動能的工具，數值介於 0 到 100 之間：
            
            * 🟣 **紫線 (RSI 數值)**：代表當下市場的買賣動能。
            * 🔴 **向上突破紅線 (RSI > 75)**：稱為「**超買**」。代表短線漲幅過大、市場情緒過熱，就像拉緊的橡皮筋，隨時有回檔下跌的風險。
            * 🟢 **向下跌破綠線 (RSI < 40)**：稱為「**超賣**」。代表短線跌幅過深、市場過度恐慌，賣壓可能已宣洩完畢，隨時有醞釀反彈的機會。
            
            *💡 本系統策略：在多頭趨勢中尋找「跌破綠線」的超賣時機買入；在「突破紅線」過熱時提示賣出風險。*
            """)
            
        with st.expander("🛡️ 專業交易指南：如何解讀趨勢與通道訊號"):
            st.markdown("""
            #### 1. 20MA 月線 (趨勢生命線)
            * **突破月線 (收盤價 > 20MA)**：代表多方取得市場控制權。
                * **策略**：趨勢偏多，短線可考慮**分批進場**，或將空單停損。
            * **跌破月線 (收盤價 < 20MA)**：代表賣壓開始湧現，中長期趨勢可能轉弱。
                * **策略**：系統傾向轉為**保守/觀望**，降低槓桿，嚴格執行停損。

            #### 2. 布林通道邊界 (波動極限)
            * **股價突破上軌 (價格 > BB_Upper)**：價格處於異常偏高區，通常是強弩之末。
                * **策略**：不宜追價，應準備**獲利了結**，或將移動停利設緊。
            * **股價跌破下軌 (價格 < BB_Lower)**：價格處於異常偏低區，市場陷入恐懼。
                * **策略**：若趨勢仍看好，此處為**反彈買點**；若跌勢強勁，則為**搶反彈止跌訊號**。
            
            #### 3. KD 指標 (短線動能節奏)
            * **黃金交叉 (K 值 > D 值)**：代表多方動能增強。
                * **策略**：短線進場訊號，若同時配合價格站上 20MA，勝率更高。
            * **死亡交叉 (K 值 < D 值)**：代表空方動能增強。
                * **策略**：短線出場訊號，建議減碼或觀察是否有支撐。
            """)
            st.info("💡 **貼心提醒**：量化指標僅供決策輔助，短線波動建議搭配當下的總體市場消息進行綜合判斷。")

    with tab2:
        st.markdown("### 🧩 多因子專家給分狀態")
        st.markdown(f"**當前收盤價：** `{summary['latest_close']:.2f}`")
        
        def render_score_card(title, score, pos_msg, neg_msg, neutral_msg, value_str):
            color = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
            msg = pos_msg if score > 0 else neg_msg if score < 0 else neutral_msg
            st.markdown(f"""
            **{color} {title} (得分: {score})**
            * 狀態：{msg}
            * 當前數值：`{value_str}`
            ---
            """)
            
        render_score_card(
            "趨勢專家 (20MA 月線)", summary['score_ma'],
            "收盤價高於月線，趨勢偏多 (+1)", "收盤價低於月線，趨勢偏空 (-1)", "價格剛好落在月線上 (0)",
            f"MA: {summary['val_ma']:.2f}"
        )
        render_score_card(
            "動能專家 (MACD 柱狀體)", summary['score_macd'],
            "柱狀體大於零，多方蓄力 (+1)", "柱狀體小於零，空方蓄力 (-1)", "動能不明確 (0)",
            f"MACD Hist: {summary['val_macd']:.2f}"
        )
        render_score_card(
            "反轉專家 (RSI 14日)", summary['score_rsi'],
            "數值低於 40，具備超賣反彈契機 (+1)", "數值高於 75，短線過熱風險高 (-1)", "數值介於 40~75，動能中性 (0)",
            f"RSI: {summary['val_rsi']:.2f}"
        )
        render_score_card(
            "邊界專家 (布林通道)", summary['score_bb'],
            "價格跌破或接近下軌，過度悲觀 (+1)", "價格突破或接近上軌，過度樂觀 (-1)", "價格於通道內正常波動 (0)",
            f"上軌: {summary['val_bbu']:.2f} | 下軌: {summary['val_bbl']:.2f}"
        )
        render_score_card(
            "動能節奏專家 (KD 指標)", summary['score_kd'],
            "K 值大於 D 值，短線黃金交叉，多方佔優 (+1)", "K 值小於 D 值，短線死亡交叉，空方修正 (-1)", "K 值與 D 值糾結，方向不明 (0)",
            f"K 值: {summary['val_k']:.2f} | D 值: {summary['val_d']:.2f}"
        )

        st.markdown("#### 📈 近期運算明細表格")
        display_days = st.radio("選擇顯示天數：", options=[5, 10, 15, 20], index=0, horizontal=True)
        display_cols = ['Close', 'Score', 'Action', '20MA', 'MACD_Hist', 'RSI', 'BB_Lower', 'K', 'D']
        st.dataframe(df[display_cols].tail(display_days).round(2), use_container_width=True)