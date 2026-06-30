import streamlit as st
import streamlit_authenticator as stauth
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 1. 網頁基本設定 (必須在最頂部) ===
st.set_page_config(page_title="資金控管決策交易系統", page_icon="📈", layout="centered")

# ─── 核心功能：具備快取與例外處理的資料下載引擎 ───
@st.cache_data(ttl=300, show_spinner=False)  # 暫存 5 分鐘，避免重複打 API
def get_stock_data(ticker, period="1y"):     # 👈 這裡的參數改為 period，完美對接下方的呼叫
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        return None
    
import json

# ─── 安全風控：將唯讀的 secrets 轉換為可修改的標準字典 (Deep Copy) ───
# 這樣 authenticator 就能正常寫入登入紀錄，不會再觸發唯讀錯誤
credentials_dict = json.loads(json.dumps(dict(st.secrets["credentials"])))

authenticator = stauth.Authenticate(
    credentials_dict,
    st.secrets["cookie"]["name"],
    st.secrets["cookie"]["key"],
    st.secrets["cookie"]["expiry_days"]
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
st.title("📈 資金控管決策交易系統")
st.markdown("本系統整合多因子綜合評分，提供客觀的交易決策與白盒化的邏輯解析。")
st.divider()

ticker_symbol = st.text_input("請輸入股票或 ETF 代碼 (台股加 .TW，如 0050.TW；美股如 VOO):", "0050.TW")

if st.button("🚀 執行量化分析", type="primary"):
    with st.spinner(f"正在從交易所獲取 {ticker_symbol} 最新數據與繪製圖表..."):
            # 使用我們剛在最上方建立好的快取下載引擎
            stock_data = get_stock_data(ticker_symbol, period="1y")
            
            # 防呆：如果找不到資料或網路斷線，優雅地擋下並提示使用者
            if stock_data is None:
                st.error("❌ 無法獲取數據，請檢查股票代碼是否正確（台股請加 .TW，如 0050.TW），或稍後再試。")
                st.session_state.analyzed = False
                st.stop()  # 👈 加上這行非常重要！這會停止執行下方所有程式碼，避免畫面紅字崩潰
            # 清洗欄位索引
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)
            else:
                stock_data.columns = stock_data.columns.astype(str)
            stock_data.columns = stock_data.columns.str.strip()
            
            df = stock_data[['Open', 'High', 'Low', 'Close']].copy()
            
            # --- 核心指標計算區 ---
            df['5MA'] = df['Close'].rolling(window=5).mean()
            df['20MA'] = df['Close'].rolling(window=20).mean()
            
            # RSI
            delta = df['Close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            df['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))
            # 處理市場完全無波動 (gain=0, loss=0) 導致 rs 為 NaN 的極端情況
            df['RSI'] = df['RSI'].fillna(50)
            
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
            
            # 移除指標暖機期的空值，確保訊號絕對乾淨
            df.dropna(inplace=True)
            
            # --- 權重型獨立計分系統 (總分滿分 \pm 10) ---
            # 1. 20MA 權重 3 分
            df['Score_MA'] = np.where(df['Close'] > df['20MA'], 3, np.where(df['Close'] < df['20MA'], -3, 0))
            
            # 2. 5MA 權重 2 分
            df['Score_5MA'] = np.where(df['Close'] > df['5MA'], 2, np.where(df['Close'] < df['5MA'], -2, 0))
            
            # 3. MACD 柱狀體 權重 2 分
            df['Score_MACD'] = np.where(df['MACD_Hist'] > 0, 2, np.where(df['MACD_Hist'] < 0, -2, 0))
            
            # 4. KD 指標 權重 1 分
            df['Score_KD'] = np.where(df['K'] > df['D'], 1, np.where(df['K'] < df['D'], -1, 0))
            
            # 5. RSI 指標 權重 1 分
            df['Score_RSI'] = np.where(df['RSI'] < 40, 1, np.where(df['RSI'] > 75, -1, 0))
            
            # 6. 布林通道邊界 權重 1 分
            df['Score_BB'] = np.where(df['Close'] <= df['BB_Lower'] * 1.01, 1, np.where(df['Close'] >= df['BB_Upper'] * 0.99, -1, 0))
            
            # 總分加總：3 + 2 + 2 + 1 + 1 + 1 = 滿分 10 分
            df['Score'] = df['Score_MA'] + df['Score_5MA'] + df['Score_MACD'] + df['Score_KD'] + df['Score_RSI'] + df['Score_BB']
            
            
           # --- 訊號觸發門檻修改 ---
            df['Action'] = 'Hold'
            df.loc[df['Score'] >= 4, 'Action'] = 'Buy'   # 👈 這裡將 2 改成 4（多方強烈共振才進場）
            df.loc[df['Score'] <= -2, 'Action'] = 'Sell' # 👈 這裡維持 -2（讓階梯式減碼提早發動）

            # 存入 Session State
            st.session_state.df = df
            st.session_state.summary = {
                'latest_date': df.index[-1].strftime('%Y-%m-%d'),
                'latest_close': float(df['Close'].iloc[-1]),
                'latest_score': int(df['Score'].iloc[-1]),
                'score_5ma': int(df['Score_5MA'].iloc[-1]),
                'score_ma': int(df['Score_MA'].iloc[-1]),
                'score_macd': int(df['Score_MACD'].iloc[-1]),
                'score_rsi': int(df['Score_RSI'].iloc[-1]),
                'score_bb': int(df['Score_BB'].iloc[-1]),
                'score_kd': int(df['Score_KD'].iloc[-1]),
                'latest_action': df['Action'].iloc[-1],
                'latest_atr': float(df['ATR'].iloc[-1]),
                'val_5ma': float(df['5MA'].iloc[-1]),
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
    # ─── 儀表板狀態文字對齊 10 分制 ───
    score_color = "🟢 強勢偏多 (觸發買進)" if score >= 4 else "🔴 弱勢偏空 (觸發減碼)" if score <= -2 else "🟡 震盪觀望"
        
    sc1, sc2 = st.columns([1, 2])
    sc1.metric("綜合評分 (-10 到 +10)", f"{score} 分")
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
            
            # 將原本的浮點數股數加上 int() 轉換為整數型態，更符合實際下單狀況
            suggested_shares = int(max_loss_amount / risk_per_share) 
            
            # ─── ⚖️ 資金天花板防呆機制（新嵌入位置） ───
            # 計算目前總可用資金在當前股價下，滿倉最多能買進的股數天花板
            max_affordable_shares = int(total_capital / summary['latest_close'])
            
            # 如果風險模型算出來的股數大於現金允許購買的最大值，進行強制攔截
            if suggested_shares > max_affordable_shares:
                suggested_shares = max_affordable_shares
                st.warning(f"⚠️ **資金控管啟動**：因近期股價波動極小，依風險公式計算的建議股數已超出總可用資金。系統已強制將建議股數下修至滿水位（{max_affordable_shares:,} 股）。")
            
            # 依據防呆修正後的股數，精確計算實際總投入金額
            total_investment = suggested_shares * summary['latest_close']
            
            rc1, rc2, rc3 = st.columns(3)
            rc1.warning(f"**🛑 建議停損價:**\n\n {stop_loss_price:.2f}")
            rc2.info(f"**🎯 預期停利價:**\n\n {take_profit_price:.2f}")
            rc3.success(f"**🛒 建議買進股數:**\n\n 約 {int(suggested_shares):,} 股")
            st.caption(f"*資金估算：此部位將佔用資金約 ${total_investment:,.0f}，預期最大虧損控制在 ${max_loss_amount:,.0f} 以內。*")
        elif summary['latest_action'] == 'Sell':
            st.error(f"🔴 **建議：賣出 (Sell)** —— 技術面出現多重破綻，系統判定風險升溫。")
            st.markdown("#### 🛡️ 出場與做空防守計畫")
            
            # 1. 根據分數嚴重程度，判定動態減碼比例
            if score == -2:
                sell_advice = "減碼 1/3 (觀望)"
            elif score == -3:
                sell_advice = "減碼 50% (降風險)"
            else: # -4 或 -5 分
                sell_advice = "100% 清倉 (全面防守)"
                
            # 2. 計算反向做空 (Short) 的停利與停損 (邏輯與做多相反)
            # 做空的停損是往上加 (價格漲過頭要停損)，停利是往下減 (跌到目標價回補)
            short_stop_loss = summary['latest_close'] + (1.5 * summary['latest_atr'])
            short_take_profit = summary['latest_close'] - (1.5 * summary['latest_atr'] * reward_risk_ratio)
            
            rc1, rc2, rc3 = st.columns(3)
            rc1.warning(f"**📉 持有多單減碼建議:**\n\n {sell_advice}")
            rc2.info(f"**🎯 融券做空預期停利:**\n\n {short_take_profit:.2f}")
            rc3.error(f"**🛑 融券做空防守停損:**\n\n {short_stop_loss:.2f}")
            
            st.caption(f"*策略解析：現股多單請參考減碼比例分批出場；若欲反向融券做空，請嚴格將停損點設定於 ${short_stop_loss:.2f} 以對抗軋空風險。*")
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

        # 第一層 (價格、雙均線與布林通道)
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K線'), row=1, col=1)
        
        # 🟢 新增：5日均線 (使用亮藍色實線，代表短線攻擊)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['5MA'], line=dict(color='#00D2FF', width=1.5), name='5MA (週線)'), row=1, col=1)
        
        # 🟡 新增：20日均線 (使用亮橘色實線，代表中期生命線)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['20MA'], line=dict(color='#FF9F00', width=2), name='20MA (月線)'), row=1, col=1)
        
        # 布林通道維持灰色虛線
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Upper'], line=dict(color='rgba(128,128,128,0.5)', dash='dash'), name='布林上軌'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Lower'], line=dict(color='rgba(128,128,128,0.5)', dash='dash'), name='布林下軌'), row=1, col=1)

        # 第二層 (RSI)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=75, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=40, line_dash="dot", line_color="green", row=2, col=1)

        # 第三層 (KD)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['K'], name='K值', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['D'], name='D值', line=dict(color='orange')), row=3, col=1)

        # 圖表外觀與佈局設定
        fig.update_layout(
            title_text=f"{ticker_symbol} 綜合技術指標圖表",
            height=800,
            xaxis_rangeslider_visible=False, # 隱藏 K 線下方的滑桿以利保持整潔
            showlegend=True
        )
        
        # ─── 移除無交易日的空白區塊，並優化 X 軸標籤顯示 ───
        tick_vals = []
        tick_text = []
        
        # 利用 Pandas 按年份與月份將資料分組
        for (year, month), group in plot_df.groupby([plot_df.index.year, plot_df.index.month]):
            # 抓取該月第 1 個交易日 (索引為 0)
            if len(group) > 0:
                day_1 = group.index[0]
                tick_vals.append(day_1)
                tick_text.append(day_1.strftime('%Y-%m-%d')) # 格式化為乾淨的 YYYY-MM-DD
                
            # 抓取該月第 8 個交易日 (索引為 7)
            if len(group) >= 8:
                day_8 = group.index[7]
                tick_vals.append(day_8)
                tick_text.append(day_8.strftime('%Y-%m-%d'))

        # 更新 X 軸設定
        fig.update_xaxes(
            type='category',
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=-45 # 讓日期稍微向左傾斜，視覺上更整齊
        )

        # 渲染圖表到 Streamlit 網頁上
        st.plotly_chart(fig, use_container_width=True)

        # ─── 整合版：系統指標與交易指南下拉式說明 ───
        with st.expander("📖 系統指標與交易指南綜合說明 (點擊展開)"):
            st.markdown("""
            ### 📊 核心技術指標解析
            本系統採用六大指標加權共振 (滿分 $\pm 10$ 分)，以客觀數據輔助主觀判斷：
            * **雙均線系統 (5MA/20MA)**：20日線 (3分) 為中期結構防守線，5日線 (2分) 為短線動能攻擊線。兩者皆上揚為強烈多頭。
            * **MACD 柱狀體**：測量波段加速度，紅柱代表多方動能，綠柱代表空方動能。
            * **RSI (相對強弱指標)**：測量市場情緒溫度。突破紫線代表市場過熱，股價隨時可能向下回檔；跌破綠線代表賣壓宣洩完，極可能迎來向上反彈。
            * **KD 指標**：尋找短線精確轉折節奏，K > D 偏多，反之偏空。
            * **布林通道 (BB)**：捕捉極端邊界，股價觸及下軌容易反彈，觸及上軌容易回檔。

            ---
            ### ⚔️ 資金控管與決策指南
            * **🟢 買進 (Buy)｜總分 $\ge 4$ 分**：代表多方強烈共振。請依據系統算出的「建議買進股數」嚴格控管資金，並將防守點設定於「ATR 動態停損價」。
            * **🔴 賣出 (Sell)｜總分 $\le -2$ 分**：代表短線動能或主結構轉弱。系統會依據分數嚴重程度給予「階梯式減碼 (1/3、50%、全面清倉)」建議，避免一次性重倉套牢。
            * **🟡 觀望 (Hold)｜總分 -1 到 3 分**：市場處於盤整震盪或多空力道抵銷，建議空手觀望或維持原有部位，不宜盲目擴大曝險。
            """)

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
            "攻擊線專家 (5位均線/週線)", summary['score_5ma'],
            "收盤價高於 5MA，短線多頭攻擊強勁 (+1)", "收盤價低於 5MA，短線動能熄火偏空 (-1)", "價格剛好落在 5MA 上 (0)",
            f"5MA: {summary['val_5ma']:.2f}"
        )    
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
        display_cols = ['Close', 'Score', 'Action', '5MA', '20MA', 'MACD_Hist', 'RSI', 'BB_Lower', 'K', 'D']
        st.dataframe(df[display_cols].tail(display_days).round(2), use_container_width=True)