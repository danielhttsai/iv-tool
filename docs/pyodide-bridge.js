/*
 * Pyodide 橋接：讓這個工具的 Python 後端完全在瀏覽器裡跑,不需要伺服器。
 *
 * 作法:載入 Pyodide 與數學套件,把 backend 的 .py 寫進 Pyodide 檔案系統,
 * 然後攔截所有對 /api/* 的 fetch,改交給 py/api.py 的 route() 處理。
 * 這樣原本的 app.js 一行都不用改 —— 它以為自己還在呼叫真的後端。
 *
 * 此檔由 build_docs.py 複製到 docs/。Python 來源仍是 backend/*.py(單一來源)。
 */
(function () {
  "use strict";

  // Pyodide 套件版本需與 index.html 載入的 pyodide.js 相符
  var PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  // Cache-bust for the Python sources fetched below. Bump whenever any backend
  // .py changes so returning browsers don't run a stale module from HTTP cache.
  var PY_VER = "87";
  var PY_MODULES = ["i18n", "iv_core", "assumptions", "ml_iv", "gen_data", "rdd_core", "rdd_survival", "rdd_assumptions", "rdd_gen", "rdd_ml", "did_core", "did_gen", "did_assumptions", "did_ml", "tit_core", "tit_gen", "tit_assumptions", "tit_realmle", "its_core", "its_gen", "its_assumptions", "its_ml", "perr_core", "perr_gen", "perr_assumptions", "ccw_core", "ccw_gen", "ccw_assumptions", "cctc_core", "cctc_gen", "cctc_assumptions", "seq_core", "seq_gen", "seq_assumptions", "cc_core", "cc_gen", "cc_assumptions", "cc_ml", "sccs_core", "sccs_gen", "sccs_assumptions", "sccs_ml", "acnu_core", "acnu_gen", "acnu_assumptions", "acnu_ml", "pnu_core", "pnu_gen", "pnu_assumptions", "pnu_ml", "nc_core", "nc_gen", "nc_assumptions", "nc_ml", "med_core", "med_gen", "med_assumptions", "med_ml", "ps_core", "ps_gen", "ps_assumptions", "ps_ml", "tmle_core", "tmle_gen", "tmle_assumptions", "tmle_ml", "gm_core", "gm_gen", "gm_assumptions", "gm_ml", "tnd_core", "tnd_gen", "tnd_assumptions", "tnd_ml", "pssa_core", "pssa_gen", "pssa_assumptions", "tscan_core", "tscan_gen", "tscan_assumptions", "wce_core", "wce_gen", "wce_assumptions", "api"];

  var pyodide = null;
  var routeFn = null;
  var readyPromise = null;
  var sklearnLoaded = false;
  var pendingApiCalls = 0;   // >0 時背景預熱會讓路給使用者主動觸發的分析

  // ---- 載入進度遮罩 -------------------------------------------------------
  function buildOverlay() {
    var o = document.createElement("div");
    o.id = "pyloader";
    o.innerHTML =
      '<div class="pyloader-box">' +
      '<div class="pyloader-logo"><img src="assets/phdc-logo.png" alt="PHDc · Population Health Data Center" /></div>' +
      '<div class="pyloader-title">真實世界證據與準實驗<span>工具箱</span></div>' +
      '<div class="pyloader-msg" id="pyloader-msg">啟動中…</div>' +
      '<div class="pyloader-track"><div class="pyloader-bar" id="pyloader-bar"></div></div>' +
      '<div class="pyloader-note">第一次開啟需下載運算核心,請稍候;之後會被瀏覽器快取。</div>' +
      "</div>";
    var css = document.createElement("style");
    css.textContent =
      "#pyloader{position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;" +
      "background:radial-gradient(120% 120% at 50% 0%,#ffffff 0%,#f1f7f6 60%,#e6f1ef 100%);color:#14283c;" +
      "transition:opacity .5s ease;font-family:system-ui,'Noto Sans TC',sans-serif}" +
      ".pyloader-box{max-width:460px;width:86%;text-align:center;padding:2rem}" +
      ".pyloader-logo{margin:0 auto 1.5rem;display:block}" +
      ".pyloader-logo img{height:96px;width:auto;display:block;margin:0 auto}" +
      ".pyloader-title{font-size:1.5rem;font-weight:800;margin-bottom:1.2rem;letter-spacing:.5px;color:#14283c}" +
      ".pyloader-title span{color:#3f8268}" +
      ".pyloader-msg{font-size:1rem;margin-bottom:1rem;min-height:1.4em;color:#334155}" +
      ".pyloader-track{height:8px;border-radius:99px;background:#dbe7e4;overflow:hidden}" +
      ".pyloader-bar{height:100%;width:5%;border-radius:99px;background:linear-gradient(90deg,#3f8268,#79c2a2);transition:width .4s ease}" +
      ".pyloader-note{margin-top:1.2rem;font-size:.8rem;color:#64748b;line-height:1.6}";
    document.head.appendChild(css);
    (document.body || document.documentElement).appendChild(o);
  }
  function setStatus(msg, pct) {
    var m = document.getElementById("pyloader-msg");
    if (m && msg != null) m.textContent = msg;
    var b = document.getElementById("pyloader-bar");
    if (b && pct != null) b.style.width = pct + "%";
  }
  function hideOverlay() {
    var o = document.getElementById("pyloader");
    if (!o) return;
    o.style.opacity = "0";
    setTimeout(function () { if (o.parentNode) o.parentNode.removeChild(o); }, 600);
  }

  // ---- 背景載入小提示(角落膠囊)------------------------------------------
  // 主遮罩在「運算核心」就緒後就移除,讓使用者馬上閱讀教學;數學套件改在背景載入,
  // 這顆膠囊只是告知「運算還在熱機」,不擋住畫面。
  function buildPill() {
    if (document.getElementById("pycompute")) return;
    var en = (document.documentElement.lang || "").indexOf("en") === 0;
    var p = document.createElement("div");
    p.id = "pycompute";
    p.innerHTML = '<span class="pyc-dot"></span><span id="pyc-msg">' +
      (en ? "Warming up the compute core… (you can read the teaching tabs now)"
          : "運算核心熱機中…（教學分頁已可閱讀）") + "</span>";
    var css = document.createElement("style");
    css.textContent =
      "#pycompute{position:fixed;right:14px;bottom:14px;z-index:9998;display:flex;align-items:center;gap:.5rem;" +
      "background:#14283c;color:#fff;padding:.5rem .9rem;border-radius:999px;max-width:80vw;" +
      "font:600 .82rem system-ui,'Noto Sans TC',sans-serif;box-shadow:0 4px 14px rgba(0,0,0,.18);" +
      "opacity:.96;transition:opacity .4s}" +
      ".pyc-dot{width:9px;height:9px;border-radius:50%;background:#79c2a2;flex:0 0 auto;animation:pycpulse 1s infinite}" +
      "@keyframes pycpulse{0%,100%{opacity:.35}50%{opacity:1}}";
    document.head.appendChild(css);
    (document.body || document.documentElement).appendChild(p);
  }
  function setPill(msg) { var m = document.getElementById("pyc-msg"); if (m && msg) m.textContent = msg; }
  function removePill() {
    var p = document.getElementById("pycompute"); if (!p) return;
    p.style.opacity = "0";
    setTimeout(function () { if (p.parentNode) p.parentNode.removeChild(p); }, 400);
  }

  // ---- 初始化 Pyodide -----------------------------------------------------
  async function init() {
    // 先同步蓋上遮罩(此時還在 <head> 解析、<body> 尚未繪出),
    // 避免第一頁(IV)先閃一下才被 loading 介面蓋住。
    buildOverlay();
    if (document.readyState === "loading") {
      await new Promise(function (r) { document.addEventListener("DOMContentLoaded", r); });
    }
    try {
      setStatus("正在載入運算核心(Pyodide)…", 25);
      pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });

      // 核心就緒就立刻撤遮罩:①是什麼、⑥如果……、⑦SAS、怎麼選 等教學內容都是純前端、
      // 不需 Python,使用者可以馬上開始讀。較大的數學套件(numpy/scipy/pandas,含 openblas)
      // 與分析程式改在背景載入;真正按下②③④的分析時,ready() 會自動等到它就緒。
      setStatus("運算核心就緒 ✓", 100);
      hideOverlay();
      buildPill();

      // 套件下載(CDN)與分析程式抓取(同源)並行跑,縮短總等待。
      var fetchP = Promise.all(PY_MODULES.map(function (name) {
        return fetch("py/" + name + ".py?v=" + PY_VER).then(function (resp) {
          if (!resp.ok) throw new Error("載入 " + name + ".py 失敗(" + resp.status + ")");
          return resp.text().then(function (src) { return [name, src]; });
        });
      }));
      await pyodide.loadPackage(["numpy", "scipy", "pandas"]);
      var sources = await fetchP;
      sources.forEach(function (ns) { pyodide.FS.writeFile(ns[0] + ".py", ns[1]); });
      await pyodide.runPythonAsync("import api");
      routeFn = pyodide.runPython("api.route");

      removePill();

      // 預熱在背景「慢慢跑」:逐一觸發各方法分析以編譯 numpy/scipy 熱路徑,
      // 每跑一個就讓出主執行緒,避免凍住 UI。使用者不必等預熱就能開始用。
      backgroundWarmup();
    } catch (err) {
      setStatus("載入失敗：" + (err && err.message ? err.message : err), null);
      setPill("載入失敗：" + (err && err.message ? err.message : err));
      console.error("[pyodide-bridge] init failed", err);
      throw err;
    }
  }
  function ready() {
    if (!readyPromise) readyPromise = init();
    return readyPromise;
  }

  // 預熱:只跑「預設可見的 IV」一個方法(example + analyze)。
  // 共用的 numpy/scipy/pandas 熱路徑被第一個 analyze 編譯後就熱了,其餘 21 個
  // 方法各自的程式碼很小,等使用者第一次切到時即時編譯即可(單一呼叫、可接受)。
  // 不再把 22 個方法全跑一遍——那會讓重運算(ccw/seq/tmle 等)在啟動時一個接一個
  // 凍住主執行緒,造成「網頁有點當」。
  // 另外:在瀏覽器空檔(requestIdleCallback)才跑、跑前讓頁面先穩定,進一步降低卡頓。
  var warmupDone = false;
  async function backgroundWarmup() {
    if (warmupDone) return;
    var calls = [
      ["GET", "/api/example", "{}", "{}"],
      ["POST", "/api/analyze", "{}", JSON.stringify({ source: "example", lang: "zh" })],
    ];
    await sleep(400);            // 讓首屏先繪好、互動先就緒
    for (var i = 0; i < calls.length; i++) {
      // 若使用者正在主動操作(有待處理的 /api 呼叫),先讓路給他,避免和預熱搶主執行緒。
      while (pendingApiCalls > 0) { await sleep(80); }
      await idle();              // 等到瀏覽器空檔再跑這一個同步呼叫
      try { routeFn(calls[i][0], calls[i][1], calls[i][2], calls[i][3]); }
      catch (e) { /* 預熱失敗不影響功能,忽略 */ }
      await sleep(50);           // 讓出主執行緒,UI 保持回應
    }
    warmupDone = true;
  }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  // 等到瀏覽器主執行緒空檔(無 requestIdleCallback 時退回 setTimeout)。
  function idle() {
    return new Promise(function (r) {
      if (typeof requestIdleCallback === "function") {
        requestIdleCallback(function () { r(); }, { timeout: 1000 });
      } else { setTimeout(r, 50); }
    });
  }

  async function ensureSklearn() {
    if (sklearnLoaded) return;
    await pyodide.loadPackage("scikit-learn");
    sklearnLoaded = true;
  }

  // ---- 處理一個 /api/* 請求 ----------------------------------------------
  async function handleApi(method, url, init) {
    pendingApiCalls++;
    try {
      return await _handleApi(method, url, init);
    } finally {
      pendingApiCalls--;
    }
  }
  async function _handleApi(method, url, init) {
    await ready();
    var u = new URL(url, location.href);
    var path = u.pathname;
    var query = {};
    u.searchParams.forEach(function (v, k) { query[k] = v; });

    var bodyObj = {};
    if (init && init.body) {
      if (typeof FormData !== "undefined" && init.body instanceof FormData) {
        var file = init.body.get("file");
        bodyObj = { _csv_text: file ? await file.text() : "" };
      } else if (typeof init.body === "string") {
        try { bodyObj = JSON.parse(init.body); } catch (e) { bodyObj = {}; }
      }
    }

    if (path === "/api/ml_forbidden" || path === "/api/did_dml" || path === "/api/its_mlcf" || path === "/api/cc_forest" || path === "/api/sccs_selfmatch" || path === "/api/acnu_psml" || path === "/api/pnu_psml" || path === "/api/med_natural_ml" || path === "/api/ps_ml" || path === "/api/tmle_ml" || path === "/api/gm_ml" || path === "/api/tnd_ml") {
      setStatus && null; // sklearn 首次載入(無遮罩,由按鈕顯示「計算中」)
      await ensureSklearn();
    }

    var out = routeFn(method, path, JSON.stringify(query), JSON.stringify(bodyObj));
    var env = JSON.parse(out);
    return new Response(JSON.stringify(env.body), {
      status: env.status,
      headers: { "Content-Type": "application/json" },
    });
  }

  // ---- 攔截 fetch ---------------------------------------------------------
  var realFetch = window.fetch ? window.fetch.bind(window) : null;
  window.fetch = function (input, init) {
    var url = typeof input === "string" ? input : (input && input.url) || "";
    if (url && url.indexOf("/api/") !== -1) {
      var method = (init && init.method) ||
        (typeof input !== "string" && input && input.method) || "GET";
      return handleApi(method, url, init || {});
    }
    return realFetch(input, init);
  };

  // 立刻開始載入(背景),這樣 app.js 載入時遮罩已在跑
  ready();
})();
