from __future__ import annotations
import os
import io
import re
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client

SYSTEM_PROMPT = (
    "You are an expert server log analyst. "
    "Given a server error log, always respond in the following format:\n\n"
    "Inference\n"
    "One or two sentences identifying the most likely root cause of the error.\n\n"
    "Troubleshooting Steps\n"
    "- At least 3 clear, specific, actionable steps to resolve the issue.\n\n"
    "Suggested Linux Commands\n"
    "- At least 3 Linux commands relevant to diagnosing this error. "
    "For each command, write the command followed by a dash and a one-sentence explanation "
    "of what it does and why it is useful here.\n\n"
    "Use exactly these section headers. Always provide all three sections even if the log is ambiguous."
)

FALLBACK_COMMANDS = [
    ("journalctl -xe", "Shows recent systemd journal entries including errors and warnings."),
    ("dmesg | tail -30", "Displays the last 30 kernel log messages, useful for hardware and driver issues."),
    ("systemctl status", "Lists the status of all systemd services to identify any that have failed."),
    ("df -h", "Shows disk usage in human-readable format — useful if a full disk is causing errors."),
    ("free -m", "Displays memory usage in megabytes to check for memory exhaustion."),
    ("cat /var/log/syslog | tail -50", "Shows the last 50 lines of the system log for recent error messages."),
]


def parse_ai_response(text: str) -> dict:
    inference = ""
    steps: list[str] = []
    commands: list[dict] = []

    in_inference = False
    in_steps = False
    in_cmds = False

    for line in text.split("\n"):
        l = line.strip()
        if not l:
            continue
        if re.match(r"^inference", l, re.I):
            in_inference, in_steps, in_cmds = True, False, False
            continue
        if re.match(r"^troubleshooting steps", l, re.I):
            in_inference, in_steps, in_cmds = False, True, False
            continue
        if re.match(r"^suggested linux commands", l, re.I):
            in_inference, in_steps, in_cmds = False, False, True
            continue

        if in_inference and not inference:
            inference = l
        elif in_steps and re.match(r"^[-*•]", l):
            steps.append(l.lstrip("-*• ").strip())
        elif in_cmds and re.match(r"^[-*•]", l):
            raw = l.lstrip("-*• ").strip()
            # Split on ' - ' or ' — ' or ': '
            for sep in [" — ", " - ", " – ", ": "]:
                if sep in raw:
                    cmd, expl = raw.split(sep, 1)
                    commands.append({"cmd": cmd.strip(), "explanation": expl.strip()})
                    break
            else:
                commands.append({"cmd": raw, "explanation": ""})

    # Pad commands with fallbacks if needed
    existing_cmds = {c["cmd"] for c in commands}
    for cmd, expl in FALLBACK_COMMANDS:
        if len(commands) >= 3:
            break
        if cmd not in existing_cmds:
            commands.append({"cmd": cmd, "explanation": expl})

    return {
        "inference": inference or "No clear root cause identified. Review the troubleshooting steps below.",
        "steps": steps[:5] if steps else ["Review the full log for repeated error patterns.",
                                           "Check service status with systemctl status.",
                                           "Inspect recent system logs with journalctl -xe."],
        "commands": commands[:5],
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    filename = f.filename.lower()

    try:
        if filename.endswith(".docx"):
            if not DOCX_AVAILABLE:
                return jsonify({"error": "DOCX support not available. Upload a .txt or .log file."}), 400
            doc = DocxDocument(io.BytesIO(f.read()))
            log_content = "\n".join(p.text for p in doc.paragraphs)
        else:
            log_content = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if not log_content.strip():
        return jsonify({"error": "The uploaded file appears to be empty."}), 400

    try:
        client = get_client()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": log_content[:4000]},
            ],
            temperature=0.4,
            max_tokens=1200,
        )
        ai_text = response.choices[0].message.content
        parsed = parse_ai_response(ai_text)
        parsed["preview"] = log_content[:3000]
        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": f"OpenAI API error: {e}"}), 500


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
