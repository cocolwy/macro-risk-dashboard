"""
Frontend: Flask 仪表盘 — 实时展示量化团队工作进展
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, send_file, abort

app = Flask(__name__)

# 工作区路径（由 run.py 设置）
WORKSPACE = Path(os.environ.get("QUAN_WORKSPACE", "workspace"))
STATE_FILE = WORKSPACE / "state.json"
OUTPUT_DIR = WORKSPACE / "output"


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    """返回完整状态快照"""
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
    else:
        data = {"agents": {}, "logs": [], "messages": []}

    # 附加图表列表
    charts = []
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.glob("*.png")):
            name = f.stem.replace("_", " ").title()
            charts.append({"name": name, "file": f.name})
    data["charts"] = charts

    return jsonify(data)


@app.route("/chart/<filename>")
def chart(filename):
    """提供图表文件"""
    path = OUTPUT_DIR / filename
    if path.exists() and path.suffix == ".png":
        return send_file(str(path), mimetype="image/png")
    abort(404)


def start_server(host="0.0.0.0", port=8765, debug=False):
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    start_server(debug=True)
