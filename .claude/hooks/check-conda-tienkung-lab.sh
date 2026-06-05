#!/usr/bin/env bash
# .claude/hooks/check-conda-tienkung-lab.sh
#
# PreToolUse hook for the Bash tool. Intercepts python / python3 / pip / pip3
# invocations and refuses them unless the command also activates the
# `tienkung_lab` conda env. Ensures every Python execution in this project
# uses the right interpreter + dependencies instead of the base environment.
#
# Triggered by ./settings.local.json under "hooks.PreToolUse".
# stdin: JSON like {"tool_name":"Bash","tool_input":{"command":"..."}}
# exit 0 (always) — decision is conveyed via the JSON written to stdout.
#
# Uses python3 (not jq) for JSON I/O because jq is not installed on this
# machine. python3 in PATH is conda base's python — fine here because the hook
# subprocess does NOT go through the Bash tool, so it doesn't recurse.

set -eu  # NOT -o pipefail; we want the script to continue past grep -q non-matches.

# Read the command verbatim. Python's json module is strict; bad JSON → empty cmd.
cmd=$(python3 -c "import json,sys
try:
    d=json.load(sys.stdin)
    print(d.get('tool_input',{}).get('command','') or '', end='')
except Exception:
    pass" 2>/dev/null || true)

# Detect python / python3 / pip / pip3 invoked AS A COMMAND, not as a substring
# in an argument. "At command position" means: start-of-string, OR immediately
# after a shell statement separator (`;`, `&&`, `||`, `|`, `&`).
#
# Matches:    python ...                       cd /x && python ...
#             pip install                      foo; pip3 list
# Doesn't:    find -name "*python*"            grep python file.txt
#             which python                     ./mypython.sh
python_at_command_position='(^|[;&|])[[:space:]]*(python3?|pip3?)([[:space:]]|$)'

if ! printf '%s' "$cmd" | grep -qE "$python_at_command_position"; then
    # No python/pip invocation at command position — allow without comment.
    exit 0
fi

# Python/pip IS being invoked. Require the activation substring to be present
# somewhere in the same shell line (covers `source ... && conda activate
# tienkung_lab && python ...`).
if printf '%s' "$cmd" | grep -qF 'conda activate tienkung_lab'; then
    exit 0
fi

# Block with a clear, actionable message.
python3 -c "
import json, sys
fix = ('source /home/szxx/Downloads/New_miniconda3/etc/profile.d/conda.sh && '
       'conda activate tienkung_lab && <your-command>')
msg = ('⛔ Python/pip 调用前请先激活 conda env \`tienkung_lab\`,'
       '否则会用到 base 环境的 python(版本/依赖不对)。\n\n'
       '请改用:\n  ' + fix + '\n\n'
       '(此规则由项目 .claude/settings.local.json 的 PreToolUse 钩子强制。'
       '如需临时绕过,可在 .claude/settings.local.json 注释 hooks 段。)')
out = {
    'hookSpecificOutput': {
        'hookEventName': 'PreToolUse',
        'permissionDecision': 'deny',
        'permissionDecisionReason': msg,
    }
}
json.dump(out, sys.stdout, ensure_ascii=False)
"
