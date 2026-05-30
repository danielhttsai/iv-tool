"""把這個工具打包成可放上 GitHub Pages 的純靜態網站(輸出到 docs/)。

它不需要伺服器:所有計算改用 Pyodide 在瀏覽器裡跑(見 web/pyodide-bridge.js)。
單一來源原則 —— Python 仍然只有 backend/*.py + web/api.py;前端只有 frontend/* +
web/pyodide-bridge.js。docs/ 是「產生出來」的成品,改了來源後重跑這支程式即可。

用法:
    python build_docs.py
然後把 GitHub Pages 設成 main 分支的 /docs 目錄。
"""
from __future__ import annotations

import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(HERE, "frontend")
BACKEND = os.path.join(HERE, "backend")
WEB = os.path.join(HERE, "web")
DOCS = os.path.join(HERE, "docs")

PYODIDE_SCRIPT = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js"
BACKEND_PY = ["i18n.py", "iv_core.py", "assumptions.py", "ml_iv.py", "gen_data.py"]


def _clean_docs():
    if os.path.isdir(DOCS):
        shutil.rmtree(DOCS)
    os.makedirs(os.path.join(DOCS, "py"), exist_ok=True)


def _build_index():
    with open(os.path.join(FRONTEND, "index.html"), encoding="utf-8") as f:
        html = f.read()

    # 1) 絕對路徑改成相對路徑(GitHub Pages 會掛在 /<repo>/ 子路徑底下)
    html = html.replace('href="/styles.css"', 'href="styles.css"')
    html = html.replace('src="/i18n.js"', 'src="i18n.js"')
    html = html.replace('src="/app.js"', 'src="app.js"')

    # 2) 在 </head> 前注入 Pyodide 與橋接(必須在 app.js 之前先攔截 fetch)
    inject = (
        f'<script src="{PYODIDE_SCRIPT}"></script>\n'
        '<script src="pyodide-bridge.js"></script>\n'
        "<!-- 本檔由 build_docs.py 產生,請勿手改;來源為 frontend/index.html -->\n"
    )
    html = re.sub(r"</head>", inject + "</head>", html, count=1)

    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(html)


def _copy_assets():
    shutil.copyfile(os.path.join(FRONTEND, "app.js"), os.path.join(DOCS, "app.js"))
    shutil.copyfile(os.path.join(FRONTEND, "i18n.js"), os.path.join(DOCS, "i18n.js"))
    shutil.copyfile(os.path.join(FRONTEND, "styles.css"), os.path.join(DOCS, "styles.css"))
    shutil.copyfile(os.path.join(WEB, "pyodide-bridge.js"), os.path.join(DOCS, "pyodide-bridge.js"))

    for name in BACKEND_PY:
        shutil.copyfile(os.path.join(BACKEND, name), os.path.join(DOCS, "py", name))
    shutil.copyfile(os.path.join(WEB, "api.py"), os.path.join(DOCS, "py", "api.py"))

    # .nojekyll 讓 GitHub Pages 不要用 Jekyll 處理(更快、且不過濾任何檔案)
    open(os.path.join(DOCS, ".nojekyll"), "w").close()


def main():
    _clean_docs()
    _build_index()
    _copy_assets()
    files = []
    for root, _dirs, names in os.walk(DOCS):
        for n in names:
            files.append(os.path.relpath(os.path.join(root, n), DOCS))
    print("已產生 docs/ :")
    for f in sorted(files):
        print("  docs/" + f.replace("\\", "/"))


if __name__ == "__main__":
    main()
