"""`rag.sh vscode <project>`: write Copilot MCP config + instructions into a project.

Only touches the project you point it at; nothing is installed globally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import config

MARKER = "## Project docs: use the `docs-rag` tools"
EXAMPLES = config.REPO_DIR / "examples" / "vscode"


def install(project: Path, docs: list[str] | None = None) -> None:
    if not project.is_dir():
        sys.exit(f"Not a folder: {project}")
    sources = [f"${{workspaceFolder}}/{d.strip('/')}" for d in docs] if docs else ["${workspaceFolder}"]
    args = ["--index-dir", "${workspaceFolder}/.rag_index", "mcp"]
    for s in sources:
        args += ["--source", s]

    # .vscode/mcp.json: add or replace just our server entry.
    mcp_path = project / ".vscode" / "mcp.json"
    mcp_path.parent.mkdir(exist_ok=True)
    data = json.loads(mcp_path.read_text()) if mcp_path.exists() else {}
    data.setdefault("servers", {})["docs-rag"] = {
        "type": "stdio", "command": str(config.REPO_DIR / "rag.sh"), "args": args,
    }
    mcp_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[rag] wrote {mcp_path}")

    # .github/copilot-instructions.md: append our section once.
    instr = project / ".github" / "copilot-instructions.md"
    instr.parent.mkdir(exist_ok=True)
    existing = instr.read_text() if instr.exists() else ""
    if MARKER not in existing:
        section = (EXAMPLES / "copilot-instructions.md").read_text()
        instr.write_text((existing.rstrip() + "\n\n" if existing.strip() else "") + section)
        print(f"[rag] updated {instr}")

    gi = project / ".gitignore"
    if gi.exists() and ".rag_index" not in gi.read_text():
        gi.write_text(gi.read_text().rstrip() + "\n.rag_index/\n")
        print(f"[rag] added .rag_index/ to {gi}")

    print("[rag] done. In VS Code: open Copilot Chat in Agent mode, then start 'docs-rag' from the tools menu.")
