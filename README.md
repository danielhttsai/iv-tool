# 工具變數（IV）線上工具

一個可在瀏覽器使用的 IV 教學 / 分析工具:把 IV 的分析流程,以及 A1–A4b 的假設檢驗框架,
做成白話、可互動的網頁。**Python(FastAPI)後端 + 原生網頁前端**。

## 功能（五個分頁）

1. **IV 是什麼** — 用「隨機接種提醒」故事白話講解工具變數、干擾因子、complier / defier、LATE。
2. **互動講解** — 拉滑桿改變 complier 比例與樣本量，即時看 IV 估計與信賴區間如何變動（弱工具的陷阱）。
3. **資料分析** — 載入內建接種提醒範例或上傳自己的 CSV，選欄位後一鍵跑：naive 迴歸、第一階段、簡化式、Wald、2SLS（含/不含共變項），輸出係數比較圖與結果卡。
4. **假設檢驗** — 對目前資料自動檢查 A1–A4b，以紅／黃／綠燈號 + 白話結論呈現；並加入「反證法」精神：用**工具不等式（instrumental inequalities, Balke–Pearl）**試著從資料直接推翻 A2／A3（違反即出局，推不翻才算過關），A3 另列出 Homayra 等人提醒的**偏誤放大（bias amplification）**陷阱。
5. **用 AI 強化** — 沿用接種提醒情境，白話示範機器學習能幫 IV 做的兩件事，以及一條安全規則：
   - **藥方一・合成工具**：把很多個各自很弱的候選工具，用 ML 合成一個夠強的工具（拉滑桿即時看 F 強度與信賴區間收窄）。
   - **藥方二・可彎的第一階段**：當工具的影響是曲線（非直線）時，直線第一階段抓不到、彈性 ML 第一階段才抓得到。
   - **藥方三・安全帶（交叉擬合）**：現場用同一個隨機森林模型示範「偷看版（in-sample，會把干擾因子洩漏回來、黏在 naive）」vs「交叉擬合版（out-of-fold，拉回真值）」，說明為什麼一定要用 out-of-fold 預測避免「禁止迴歸 / 過度配適」偏誤。
   全部即時模擬合成資料，真值固定為 LATE = 1.80。

## 安裝與啟動

需求：Python 3.10+（已在 3.14 測試）。

```powershell
cd "D:\Drive\IV detection\webtool\backend"
python -m pip install -r requirements.txt
python gen_data.py          # 產生內建範例 data\demo_vaccine.csv（首次必跑）
python -m uvicorn app:app --port 8000
```

開啟瀏覽器：<http://127.0.0.1:8000>

## 內建範例資料（純屬虛構的合成資料）

`data/demo_vaccine.csv` 是一個**完全虛構、為本工具自行設計**的合成情境，
欄位、數字、變數結構都是自訂的，請勿當成真實資料使用：

> 情境：衛生單位在「隨機抽中的社區」主動寄出免費接種提醒、並安排到府接種。
> 工具 Z = `vaccine_reminder`、處置 A = `vaccinated`、結果 Y = `health_score_change`，
> 看不見的干擾因子 = 個人健康意識。

資料設計（真值）：

| 指標 | 設計值 | 本工具 |
|---|---|---|
| naive 處置係數（受干擾高估） | ~2.60 | 2.61 |
| 第一階段係數（compliers） | ~0.090（9.0%） | 0.089 |
| 真正 LATE（Wald / 2SLS） | 1.80 | 1.78 |

欄位：`health_score_change`(Y)、`vaccinated`(A)、`vaccine_reminder`(Z)、
`age, female, bmi, chronic_conditions, income_band`(共變項)。

## 自己的資料

CSV 須為數值欄位。上傳後在「資料分析」分頁選擇 Y / A / Z 與共變項即可。
工具/處置若為二元（0/1），假設檢驗中的 McFadden R²、單調性等檢查才會啟用。

## 程式結構

```
webtool/
├─ backend/
│  ├─ iv_core.py        # IV 計算核心（OLS / 第一階段 / 簡化式 / Wald / 2SLS）
│  ├─ assumptions.py    # A1–A4b 假設檢驗
│  ├─ ml_iv.py          # ML + IV 教學示範（合成工具 / 可彎第一階段 / 禁止迴歸·交叉擬合）
│  ├─ app.py            # FastAPI：API + 靜態前端
│  ├─ gen_data.py       # 產生內建合成範例資料
│  ├─ test_iv_core.py   # pytest：鎖定教材數字
│  └─ data/demo_vaccine.csv
└─ frontend/
   ├─ index.html  ├─ app.js  └─ styles.css
```

統計（OLS、2SLS、F 統計量、McFadden R²）以 numpy/scipy 自行實作，
不依賴 statsmodels / linearmodels，避免在新版 Python 上的編譯問題。
分頁⑤的「藥方三・禁止迴歸現場示範」另用 scikit-learn 的隨機森林，
對比偷看版與交叉擬合版的第一階段。

## 測試

```powershell
cd "D:\Drive\IV detection\webtool\backend"
python -m pytest -q
```

## API 摘要

| 端點 | 用途 |
|---|---|
| `GET /api/example` | 內建資料預覽與預設欄位對應 |
| `POST /api/upload` | 上傳 CSV，回傳欄位清單與 token |
| `POST /api/analyze` | 回傳所有 IV 估計（naive / 第一階段 / 簡化式 / Wald / 2SLS） |
| `POST /api/assumptions` | 回傳 A1–A4b 檢驗結果與燈號 |
| `GET /api/interactive` | 互動分頁：依 complier 比例 / 強度即時模擬 Wald |
| `GET /api/ml_synthesis` | 藥方一：多個弱工具合成一個強工具（交叉擬合） |
| `GET /api/ml_nonlinear` | 藥方二：直線 vs 可彎的第一階段 |
| `GET /api/ml_forbidden` | 藥方三：隨機森林偷看版 vs 交叉擬合版（禁止迴歸偏誤） |
| `GET /api/ml_compare` | 各種做法的綜合比較 |

## 參考文獻

方法與假設框架整理自下列文獻：

- **Homayra, F., et al. (2024).** Comparative Analysis of Instrumental Variables on the
  Assignment of Buprenorphine/Naloxone or Methadone for the Treatment of Opioid Use
  Disorder. *Epidemiology, 35*(2), 218–231. https://doi.org/10.1097/EDE.0000000000001697
  〔A1–A4b 假設檢驗框架〕
- **Leung, M. (2024).** *Instrumental Variables*〔講義〕. NCKU — Bias in Epidemiology.
- **Singh, A., Hosanagar, K., & Gandhi, A.** Machine Learning Instrument Variables for
  Causal Inference. University of Pennsylvania.〔合成工具〕
- **Bakhitov, E. (2025).** On machine learning instrumental variable estimators.
  *Economics Letters.* 〔Elsevier ScienceDirect: S0165176525004380〕
- **Bruns-Smith, D. (2025).** *Two-Stage Machine Learning for Nonparametric Instrumental
  Variable Regression.* Stanford University.〔可彎的第一階段〕
- **Peng, J. (2024).** *Machine Learning for Instrumental Variable Regression: From Bias to
  Resilience.* University of Connecticut. SSRN 5008641.〔禁止迴歸偏誤〕
- **Wu, A., Kuang, K., Xiong, R., & Wu, F. (2022).** Instrumental Variables in Causal
  Inference and Machine Learning: A Survey. *arXiv:2212.05778.*
- **Kaufman, E. J., Keele, L. J., et al.** Operative and Nonoperative Outcomes of Emergency
  General Surgery Conditions: An Observational Study Using a Novel Instrumental Variable.〔應用範例〕
