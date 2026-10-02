#!/usr/bin/env python3
"""Rule enforcement hooks: PostToolUse on Edit|Write|MultiEdit|NotebookEdit,
PreToolUse and PostToolUse on Bash to record what a shell command changed,
and Stop for turn-level rules.

Rules come straight from the instruction files a person already keeps:
CLAUDE.md, AGENTS.md (nested ones scoped to their directory), .claude/rules/*,
.cursor/rules/* and ~/.claude/CLAUDE.md. There is
nothing to compile and nothing extra to commit. Each bullet or paragraph is
classified once by Jev in a single batched request — is this an instruction
about the code a coding agent writes, or a fact, description, pointer or
process rule? and, if it is, can one edit hunk break it, or does judging it
need every change from the task? — and the verdicts are cached by file hash
under ~/.claude, so the per-edit request carries only real instructions and
a changed instruction file is reclassified on its next use. Scope globs
(`paths:` front matter, `(scope: glob)` tails, a nested file's directory)
keep out-of-scope rules out of the request entirely.

Per-edit rules judge each edit hunk; whole-turn rules (scope creep, an
abstraction with a single caller, total size) judge the session's changes
together at Stop. Verdicts are banded: at or above ACT the hook blocks and
the agent sees the cited rule; between FLAG and ACT the uncertainty goes to
the user as a notice; below stays silent. One rule may block the same file
at most twice per session — past that it only flags, because a repair that
can't land is a loop, not enforcement.

Always exits 0 and prints nothing on any failure — enforcement must never
corrupt a session.

Env:
  TYPESAFE_API_KEY or OPENROUTER_API_KEY   required (else silently disabled)

Off when `rules` ("Rule checks" in /claude-jev or /config) is off.
"""

import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comparators
import jev

ACT = 0.80
FLAG = 0.50
MAX_RULES = 40
MAX_STATE_CHARS = 8000
MAX_TASK_CHARS = 600
MAX_ANSWER_CHARS = 1500
MAX_BLOCKS = 2
MAX_STOP_BLOCKS = 2
HOOK_BUDGET = 9.0
ESCALATE_MIN = 3.0
MAX_HUNK_CHARS = 2000
MAX_TURN_CHARS = 16000
MAX_PROMPT_TAIL = 400
MAX_NESTED_DEPTH = 4

CONTEXT_LINES = 4
MAX_CONTEXT_CHARS = 1200

SIBLING_CAP = 40

ESCALATE = True
MAX_BLOCK_CHARS = 3000
RULE_CONTEXT_CHARS = 400

CALIB_FILE = os.path.join(jev.config_dir(), "jev-rules-calib.json")
ACT_DECISIVE = 0.70
ACT_NOISY = 0.85
CALIB_MIN_CHECKS = 20
CALIB_DECISIVE = 0.10
CALIB_NOISY = 0.25
RELEVANCE_GATE = True

COMPARATORS = True
DEFAULT_LOG = os.path.join(jev.config_dir(), "jev-router-log.jsonl")
BLOCK_DIR = os.path.join(jev.config_dir(), "jev-rule-blocks")

_DEADLINE: float | None = None


def budget() -> float | None:
    """Seconds left in this hook's budget, or None outside a hook run (evals
    call these functions directly). hooks.json gives each hook 10s; the
    margin covers rule loading, comparators, and output."""
    if _DEADLINE is None:
        return None
    return max(0.1, _DEADLINE - time.monotonic())


RULE_FILES = ("CLAUDE.md", "AGENTS.md")
RULE_DIRS = (".claude/rules", ".cursor/rules")
GLOBAL_RULE_FILES = (os.path.join(jev.config_dir(), "CLAUDE.md"),)
MAX_ITEM_CHARS = 600

EXCLUDED = re.compile(
    r"(^|/)(node_modules|\.git|dist|build|\.next|coverage|"
    r"\.claude|vendor|target)(/|$)|\.lock$|"
    r"package-lock\.json$|pnpm-lock\.yaml$|yarn\.lock$"
)
SKIP_DIRS = {
    "node_modules",
    ".git",
    "dist",
    "build",
    "out",
    ".next",
    "vendor",
    "coverage",
    ".turbo",
    ".cache",
    "target",
    ".claude",
    "Library",
}


def same_filesystem_dir(parent: str, path: str) -> bool:
    """Whether `path` is a real directory, not a symlink, on the same
    filesystem as `parent`. A mount
    point is a different project on a different volume, so a nested
    instruction file there is not this project's, and lstat on the mountpoint
    itself never reaches what is mounted behind it: a volume that is slow or
    unreachable cannot wedge the walk."""
    try:
        entry = os.lstat(path)
        return stat.S_ISDIR(entry.st_mode) and entry.st_dev == os.lstat(parent).st_dev
    except OSError:
        return False


BULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(.*)")
HEADING = re.compile(r"^\s*#{1,6}\s+")
NAMED = re.compile(r"^\*\*([\w-]+)\*\*:?\s*(.*)")
SCOPE_TAIL = re.compile(r"\(scope:\s*([^)]+)\)\s*$")
FRONT_MATTER = re.compile(r"^---\s*$")


def slug(text: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", text.lower())[:5]) or "rule"


def glob_match(path: str, globs: list[str]) -> bool:
    """Repo-relative posix path vs `**`/`*`/`?` globs (stdlib only).
    `**/` matches zero or more directories, like picomatch."""

    def rx(g: str) -> str:
        out = []
        parts = re.split(r"(\*\*)", g)
        for i, part in enumerate(parts):
            if part == "**":
                nxt = parts[i + 1] if i + 1 < len(parts) else ""
                if nxt.startswith("/"):
                    parts[i + 1] = nxt[1:]
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
            else:
                out.append(re.escape(part).replace(r"\*", "[^/]*").replace(r"\?", "[^/]"))
        return "^" + "".join(out) + "$"

    return any(re.match(rx(g.strip()), path) for g in globs if g.strip())


def frontmatter_paths(lines: list[str]) -> list[str]:
    """`paths:` list from YAML front matter at the top of a rules file —
    the same convention .claude/rules/*.md uses."""
    paths = []
    if not lines or not FRONT_MATTER.match(lines[0]):
        return paths
    in_paths = False
    for raw in lines[1:]:
        line = raw.rstrip()
        if FRONT_MATTER.match(line):
            break
        if re.match(r"^paths:\s*$", line):
            in_paths = True
            continue
        if in_paths:
            m = re.match(r'^\s*-\s*["\']?(.*?)["\']?\s*$', line)
            if m:
                paths.append(m.group(1))
                continue
            in_paths = False
        else:
            m = re.match(r"^paths:\s*\[(.*)\]", line)
            if m:
                paths += [p.strip().strip("\"'") for p in m.group(1).split(",")]
    return paths


SENTENCE = re.compile(r"(?<=\.)\s+(?=[A-Z])")
MIN_ITEM_CHARS = 20


def split_sentences(text: str) -> list[str]:
    parts = [t.strip() for t in SENTENCE.split(text)]
    keep = [t for t in parts if len(t) >= MIN_ITEM_CHARS]
    return keep or [text]


def markdown_items(lines: list[str]) -> list[tuple[int, str]]:
    """(line, text) per bullet or paragraph, with wrapped lines joined.
    Instruction files wrap at ~80 columns, so a rule is rarely one line;
    judging fragments produced mid-sentence "rules" and lost the rest.
    Code fences, headings and front matter are skipped — examples and
    titles are not rules."""
    items: list[tuple[int, str]] = []
    cur_line, cur = 0, []
    bullet = False
    in_fence = False
    in_front = bool(lines) and bool(FRONT_MATTER.match(lines[0]))

    def flush():
        nonlocal cur
        if cur:
            text = " ".join(x.strip() for x in cur)
            if not bullet and len(text) > MAX_ITEM_CHARS:
                items.extend((cur_line, t) for t in split_sentences(text))
            else:
                items.append((cur_line, text))
        cur = []

    for i, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        if in_front:
            if i > 1 and FRONT_MATTER.match(line):
                in_front = False
            continue
        if line.strip().startswith("```"):
            in_fence = not in_fence
            flush()
            continue
        if in_fence:
            continue
        if not line.strip() or HEADING.match(line) or line.lstrip().startswith("|"):
            flush()
            continue
        m = BULLET.match(line)
        if m:
            flush()
            cur_line, cur, bullet = i, [m.group(1)], True
        elif cur and (line.startswith((" ", "\t")) or not BULLET.match(cur[0])):
            cur.append(line)
        else:
            flush()
            cur_line, cur, bullet = i, [line], False
    flush()
    return items


INSTRUCTION_Q = (
    "Is item [{i}] a rule about the code or files a coding agent writes, "
    "such that a reviewer looking at one diff could tell whether it was "
    "followed? Facts, descriptions and pointers are not. Neither are "
    "process rules about how to work — what to read first, which "
    "commands or tools to run, how to communicate, when to delegate "
    "— because no single diff can show compliance. The section "
    "heading before each item says what the section is about; an "
    "item under a heading about workflow, sessions, tools or "
    "delegation is a process rule even when it mentions code size."
)

TURN_Q = (
    "Does judging item [{i}] need every change the agent made for the task, "
    "not just one edit hunk — because it is about the change as a whole: "
    "its total size, scope creep, edits outside what was asked, an "
    "abstraction with a single caller, or the same code repeated across "
    "files? Answer no for a rule a single hunk can break on its own."
)

TURN_CRITERIA = {
    "true": "A rule about the change as a whole: how much was changed, whether "
    "it stayed within the task, whether new code has callers, whether "
    "the same code now appears in several places.",
    "false": "A rule one hunk can break by itself: a forbidden construct, call, "
    "pattern, comment style, naming, error handling, or a required "
    "element in the code being added.",
}

POLARITY_Q = "Does item [{i}] forbid something, or require something?"
POLARITY_CRITERIA = {
    "forbid": "the rule says not to do or add something",
    "require": "the rule says something must be present or done a certain way",
}
SUBJECT_Q = "What kind of thing in a code diff does item [{i}] govern?"
SUBJECT_CRITERIA = {
    "imports_deps": "imports, requires, dependencies, third-party packages",
    "comments": "comments, docstrings, explanatory text inside code",
    "naming": "what things are called: identifiers, files, exports, casing",
    "types": "type annotations, interfaces, type safety, schemas",
    "tests": "tests, test files, assertions, fixtures",
    "errors": "error handling, exceptions, failure paths, fallbacks",
    "literals_constants": "literal values, magic numbers, hardcoded strings, constants",
    "files_structure": "which files exist, where code lives, file layout",
    "commands_process": "commands to run, workflow, process, how to work",
    "other": "anything else, or the rule governs the change as a whole",
}

CHOICE_MIN = 0.5
DEFAULT_SUBJECT = "other"
DEFAULT_POLARITY = "forbid"
CLASSIFY_CACHE = os.path.join(jev.config_dir(), "jev-rules-cache.json")
INSTRUCTION_MIN = 0.5
TURN_MIN = 0.5
ITEMS_PER_REQUEST = 15


def section_headings(lines: list[str], items: list[tuple[int, str]]) -> dict[int, str]:
    """Item line -> text of the nearest heading above it."""
    out: dict[int, str] = {}
    current = "no heading"
    pos = 0
    ordered = sorted(items)
    for i, raw in enumerate(lines, 1):
        if HEADING.match(raw):
            current = HEADING.sub("", raw).strip()
        while pos < len(ordered) and ordered[pos][0] == i:
            out[i] = current
            pos += 1
    return out


def choice_of(answer: dict | None, criteria: dict, default: str) -> str:
    """A choice answer, or the default when the model is not confident enough
    or names something outside the set."""
    a = answer or {}
    pick, conf = a.get("choice"), a.get("confidence", 0.0)
    if pick in criteria and isinstance(conf, (int, float)) and conf >= CHOICE_MIN:
        return pick
    return default


def classify_items(lines: list[str], items: list[tuple[int, str]]) -> dict[int, dict]:
    """Item index -> {"when", "polarity", "subject"} for the items Jev judges
    to be instructions; items that are not instructions are left out.
    Keyed by index, not line: one prose paragraph can yield several items.
    One batched request per chunk of ITEMS_PER_REQUEST items, cached by the
    chunk's own content hash — a file edit re-judges only the chunks that
    changed, and a hook killed mid-classification keeps the chunks that
    landed. This is the whole of what a compiled rubric used to hold that
    markdown parsing couldn't recover: it happens on first use, in
    ~/.claude, with nothing for the user to run or commit."""
    try:
        with open(CLASSIFY_CACHE) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    headings = section_headings(lines, items)

    def chunk_request(start: int) -> tuple[str, str, dict]:
        chunk = items[start : start + ITEMS_PER_REQUEST]
        state = "\n\n".join(f"[{i}] ({headings[ln]}) {text}" for i, (ln, text) in enumerate(chunk))
        questions = {}
        for i in range(len(chunk)):
            questions[f"q{i}"] = {"type": "noul", "instructions": INSTRUCTION_Q.format(i=i)}
            questions[f"t{i}"] = {
                "type": "noul",
                "instructions": TURN_Q.format(i=i),
                "criteria": TURN_CRITERIA,
            }
            questions[f"p{i}"] = {
                "type": "choice",
                "instructions": POLARITY_Q.format(i=i),
                "criteria": POLARITY_CRITERIA,
            }
            questions[f"s{i}"] = {
                "type": "choice",
                "instructions": SUBJECT_Q.format(i=i),
                "criteria": SUBJECT_CRITERIA,
            }
        key = hashlib.sha256(json.dumps([state, questions], sort_keys=True).encode()).hexdigest()
        return key, state, questions

    def classify_chunk(start: int) -> dict[int, dict]:
        key, state, questions = chunk_request(start)
        hit = cache.get(key)
        if isinstance(hit, dict):
            return {
                start + int(k): v
                for k, v in hit.items()
                if k.isdigit() and isinstance(v, dict) and v.get("when") in ("edit", "turn")
            }
        chunk = items[start : start + ITEMS_PER_REQUEST]
        answers = jev.ask(state, questions, timeout=budget())
        out = {}
        for i in range(len(chunk)):
            p = (answers.get(f"q{i}") or {}).get("noul")
            if not (isinstance(p, (int, float)) and p >= INSTRUCTION_MIN):
                continue
            t = (answers.get(f"t{i}") or {}).get("noul")
            turn = isinstance(t, (int, float)) and t >= TURN_MIN
            out[i] = {
                "when": "turn" if turn else "edit",
                "polarity": choice_of(answers.get(f"p{i}"), POLARITY_CRITERIA, DEFAULT_POLARITY),
                "subject": choice_of(answers.get(f"s{i}"), SUBJECT_CRITERIA, DEFAULT_SUBJECT),
            }
        cache[key] = {str(k): v for k, v in out.items()}
        tmp = CLASSIFY_CACHE + f".{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(cache, f)
            os.replace(tmp, CLASSIFY_CACHE)
        except OSError:
            pass
        return {start + k: v for k, v in out.items()}

    meta: dict[int, dict] = {}
    starts = list(range(0, len(items), ITEMS_PER_REQUEST))
    if len(starts) <= 1:
        for start in starts:
            meta.update(classify_chunk(start))
        return meta
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(starts), 8)) as ex:
        futs = {ex.submit(classify_chunk, s): s for s in starts}
        for f in concurrent.futures.as_completed(futs):
            try:
                meta.update(f.result())
            except Exception as e:
                errors.append(e)
    if errors and not meta:
        raise errors[0]
    return meta


def parse_rules(
    path: str, base_label: str | None = None, file_scope: list[str] | None = None
) -> list[dict]:
    """Instructions from a markdown file, each with its line for citation.
    Only items that tell the agent to do or not do something qualify."""
    rules = []
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return rules
    base = base_label or os.path.basename(path)

    file_hash = hashlib.sha256("".join(lines).encode()).hexdigest()[:12]
    scope0 = list(file_scope or []) + frontmatter_paths(lines)
    items = [(ln, t.strip()) for ln, t in markdown_items(lines) if len(t.strip()) >= MIN_ITEM_CHARS]
    meta = classify_items(lines, items)
    for idx, (line_no, text) in enumerate(items):
        if idx not in meta:
            continue
        around = " ".join(t for _ln, t in items[max(0, idx - 1) : idx + 2] if t != text)[
            :RULE_CONTEXT_CHARS
        ]
        if len(text) > MAX_ITEM_CHARS:
            cut = text.rfind(". ", 0, MAX_ITEM_CHARS)
            text = text[: cut + 1] if cut > 100 else text[:MAX_ITEM_CHARS]
        scope = list(scope0)
        sm = SCOPE_TAIL.search(text)
        if sm:
            scope += [g.strip() for g in sm.group(1).split(",")]
            text = text[: sm.start()].strip()
        name = None
        nm = NAMED.match(text)
        if nm:
            name, text = nm.group(1), nm.group(2).strip() or text
        rules.append(
            {
                "id": name or slug(text),
                "text": text,
                "file": base,
                "line": line_no,
                "scope": scope,
                "when": meta[idx]["when"],
                "file_hash": file_hash,
                "polarity": meta[idx]["polarity"],
                "subject": meta[idx]["subject"],
                "context": around,
            }
        )
    return rules


def nested_files(cwd: str) -> list[tuple[str, str]]:
    """(path, scope-glob) for AGENTS.md/CLAUDE.md below the root — abide's
    convention: a nested instruction file governs its own tree. Root files
    are the caller's job."""
    out = []

    def walk(d: str, depth: int):
        if depth > MAX_NESTED_DEPTH:
            return
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            return
        if depth > 0:
            rel = os.path.relpath(d, cwd)
            for name in RULE_FILES:
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    out.append((p, f"{rel}/**"))
        for e in entries:
            sub = os.path.join(d, e)
            if e in SKIP_DIRS or e.startswith("."):
                continue
            if same_filesystem_dir(d, sub):
                if not os.path.exists(os.path.join(sub, ".git")):
                    walk(sub, depth + 1)

    walk(cwd, 0)
    return out


def dedupe(rules: list[dict]) -> list[dict]:
    """AGENTS.md is often a byte-for-byte copy of CLAUDE.md; one judgment
    per distinct instruction, first source wins the citation."""
    seen, out = set(), []
    for r in rules:
        key = " ".join(r["text"].lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_rules(cwd: str) -> list[dict]:
    """Every instruction in the project's and the user's instruction files,
    classified and cached on the way in."""
    rules = []
    for name in RULE_FILES:
        rules += parse_rules(os.path.join(cwd, name))
    for d in RULE_DIRS:
        path = os.path.join(cwd, d)
        if os.path.isdir(path):
            for fn in sorted(os.listdir(path)):
                if fn.endswith((".md", ".mdc")):
                    rules += parse_rules(os.path.join(path, fn), base_label=f"{d}/{fn}")
    for path, scope in nested_files(cwd):
        rules += parse_rules(path, base_label=os.path.relpath(path, cwd), file_scope=[scope])
    for path in GLOBAL_RULE_FILES:
        rules += parse_rules(path, base_label="~/" + os.path.relpath(path, os.path.expanduser("~")))
    return dedupe(rules)


def relative(path: str, cwd: str) -> str:
    try:
        return os.path.relpath(path, cwd)
    except ValueError:
        return path


def outside(rel: str) -> bool:
    """A path that climbs out of cwd. The project's rules describe the project,
    so a scratch file in /tmp is not theirs to judge."""
    return rel.startswith(os.pardir + os.sep) or rel == os.pardir


def request_with_answers(task: str, answers: list[str]) -> str:
    """The request the hook sends: the typed prompt plus, when the user
    answered the agent's questions since it, each pick with its option's
    description. Takes answers oldest first; caps the whole section."""
    if not answers:
        return task
    return (
        task + "\nThe user's answers to the agent's questions since "
        "that request:\n" + "\n".join(answers)[:MAX_ANSWER_CHARS]
    )


def last_user_prompt(transcript_path: str | None) -> tuple[str, str, int]:
    """The uuid of the user's last request, its text, and how many of the
    user's answers ride in it — rules like "don't touch generated files"
    only mean something against the task, and the uuid marks which turn a
    hunk belongs to."""
    if not transcript_path:
        return "", "", 0
    try:
        with open(transcript_path, errors="replace") as f:
            lines = f.readlines()[-MAX_PROMPT_TAIL:]
    except OSError:
        return "", "", 0
    answers: list[str] = []
    for line in reversed(lines):
        if len(line) > 500_000:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        text = user_prompt(d)
        if text:
            request = request_with_answers(text[:MAX_TASK_CHARS], list(reversed(answers)))
            return d.get("uuid") or "", request, len(answers)
        answers.extend(reversed(question_answers(d)))
    return "", "", 0


def question_answers(d: dict) -> list[str]:
    """The user's picks from an AskUserQuestion result, each with the chosen
    option's description, so an approval given there counts like one typed.
    Only the main thread's results: a subagent cannot answer for the user."""
    result = d.get("toolUseResult")
    if d.get("type") != "user" or d.get("isSidechain") or not isinstance(result, dict):
        return []
    answers = result.get("answers")
    if not isinstance(answers, dict):
        return []
    options = {
        q.get("question"): q.get("options") or []
        for q in result.get("questions") or []
        if isinstance(q, dict)
    }
    out = []
    for question, answer in answers.items():
        if not isinstance(answer, str):
            continue
        picked = set(answer.split(", "))
        notes = [
            o.get("description", "")
            for o in options.get(question, [])
            if isinstance(o, dict) and o.get("label") in picked | {answer}
        ]
        described = f" ({'; '.join(n for n in notes if n)})" if any(notes) else ""
        out.append(f"Q: {question} A: {answer}{described}")
    return out


def user_prompt(d: dict) -> str:
    """The text of a transcript entry the user typed as a request, or ""
    for tool results, harness wrappers, slash commands, and subagent turns."""
    if d.get("type") != "user" or d.get("isSidechain"):
        return ""
    c = (d.get("message") or {}).get("content")
    text = c if isinstance(c, str) else ""
    if isinstance(c, list):
        text = "\n".join(
            b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
        )
    text = (text or "").strip()
    return "" if text.startswith(("<", "/", "#")) else text


def write_hunk(cwd: str, rel: str, content: str) -> str:
    """A Write replaces the file, so PostToolUse holds only the new bytes.
    When the file is tracked and was clean, `git diff` recovers the actual
    change; a whole-file payload made the judge read every existing line as
    the agent's doing (a 4x false-block rate on the stress corpus). A new
    or untracked file is judged whole, marked as such."""
    try:
        r = subprocess.run(
            ["git", "diff", "--no-color", "--no-ext-diff", "-U3", "--", rel],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
        tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel], cwd=cwd, capture_output=True, timeout=5
            ).returncode
            == 0
        )
        if tracked:
            return ""
    except (OSError, subprocess.SubprocessError):
        pass
    return f"NEW FILE (whole content):\n{content}"


def edit_hunks(inp: dict, cwd: str | None = None) -> str:
    """Old→new per edit: rules about *removing* something need both sides."""
    if isinstance(inp.get("edits"), list):
        parts = []
        for e in inp["edits"]:
            if isinstance(e, dict):
                hunk = ""
                if e.get("old_string"):
                    hunk += f"REMOVED:\n{e['old_string']}\n"
                if e.get("new_string"):
                    hunk += f"ADDED:\n{e['new_string']}"
                parts.append(hunk)
        return "\n\n".join(p for p in parts if p)
    old, new = inp.get("old_string"), inp.get("new_string")
    if old is not None or new is not None:
        hunk = ""
        if old:
            hunk += f"REMOVED:\n{old}\n"
        if new:
            hunk += f"ADDED:\n{new}"
        return hunk
    content = inp.get("content") or inp.get("new_source") or ""
    if content and cwd and inp.get("file_path"):
        return write_hunk(cwd, relative(inp["file_path"], cwd), content)
    return content


COMMENT = re.compile(r"(^|\s)(#|//|/\*|\*/|<!--)|\"\"\"|\'\'\'")
IMPORTISH = re.compile(
    r"(?m)^\s*[-+]?\s*(import\b|from\s+\S+\s+import\b|export\s+\*|"
    r"const\s+\w+\s*=\s*require\(|require\(|use\s+\w|#include\b|using\b)"
)
MANIFEST = re.compile(
    r"(^|/)(package\.json|requirements[^/]*\.txt|pyproject\.toml|"
    r"go\.mod|Cargo\.toml|Gemfile|setup\.py)$"
)
TESTISH = re.compile(r"(?i)\btest|\bspec\b|describe\(|\bit\(|assert|expect\(")
NUMBER = re.compile(r"(?<![\w.])-?\d[\d_]*(\.\d+)?\b")
STRINGY = re.compile(r"\"[^\"\n]{4,}\"|\'[^\'\n]{4,}\'|`[^`\n]{4,}`")
DEFINES = re.compile(
    r"(?m)^\s*[-+]?\s*(def\b|class\b|function\b|const\b|let\b|var\b|"
    r"type\b|interface\b|enum\b|struct\b|fn\b|export\b)"
)
TYPEISH = re.compile(
    r"(?m)(:\s*[A-Z][\w\[\]<>.]*|\bany\b|\bas\b|\binterface\b|\btype\b|"
    r"->\s*[\w\[\]]+|\bSchema\b)"
)
ERRORISH = re.compile(
    r"(?i)\b(try|catch|except|finally|throw|raise|Result|Error|Exception|" r"panic|rescue)\b"
)

SUBJECT_TESTS = {
    "imports_deps": lambda h, rel: bool(IMPORTISH.search(h) or MANIFEST.search(rel)),
    "comments": lambda h, rel: bool(COMMENT.search(h)),
    "tests": lambda h, rel: bool(TESTISH.search(h) or TESTISH.search(rel)),
    "literals_constants": lambda h, rel: bool(
        STRINGY.search(h) or any(m.group(0) not in ("0", "1", "-1") for m in NUMBER.finditer(h))
    ),
    "naming": lambda h, rel: bool(DEFINES.search(h)),
    "types": lambda h, rel: bool(TYPEISH.search(h)),
    "errors": lambda h, rel: bool(ERRORISH.search(h)),
}


def touches(hunk: str, subject: str, rel: str = "") -> bool:
    """Could this hunk possibly break a rule about `subject`? Subjects with
    no cheap test (files_structure, commands_process, other) always can."""
    test = SUBJECT_TESTS.get(subject)
    return True if test is None else bool(test(hunk, rel))


def split_relevant(in_scope: list[dict], hunk: str, rel: str = "") -> tuple[list[dict], list[dict]]:
    """(rules worth a question, rules this hunk cannot break)."""
    if not RELEVANCE_GATE:
        return list(in_scope), []
    keep, drop = [], []
    for r in in_scope:
        (keep if touches(hunk, r.get("subject") or DEFAULT_SUBJECT, rel) else drop).append(r)
    return keep, drop


EDIT_FORBID_CRITERIA = {
    "true": "The new code visibly does the forbidden thing.",
    "false": "The edit does not do it, or only removes or leaves untouched code that did.",
}

EDIT_REQUIRE_CRITERIA = {
    "true": "A case the rule clearly governs was added, and the required element is absent.",
    "false": "The rule does not govern what changed, the requirement is "
    "present, or it is a matter of degree or taste.",
}


def needle_of(inp: dict) -> str:
    """The first line the edit put in the file — the anchor for its context."""
    new = inp.get("new_string")
    if new is None and isinstance(inp.get("edits"), list):
        for e in inp["edits"]:
            if isinstance(e, dict) and e.get("new_string"):
                new = e["new_string"]
                break
    if new is None:
        new = inp.get("content") or inp.get("new_source") or ""
    for line in (new or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


MODULE_EXT = (".py", ".ts", ".tsx", ".js", ".jsx", ".mjs")


def sibling_modules(path: str) -> str:
    """Names importable from the edited file's own directory. A rule banning
    third-party imports read `import observed` as third-party and blocked a
    sibling module at 0.86; the judge cannot tell the two apart without the
    list."""
    try:
        entries = sorted(os.listdir(os.path.dirname(path) or "."))
    except OSError:
        return ""
    names = []
    for e in entries:
        full = os.path.join(os.path.dirname(path) or ".", e)
        stem, ext = os.path.splitext(e)
        if os.path.isdir(full):
            if any(
                os.path.exists(os.path.join(full, f"index{x}")) for x in MODULE_EXT
            ) or os.path.exists(os.path.join(full, "__init__.py")):
                names.append(e)
        elif ext in MODULE_EXT and not stem.startswith("."):
            names.append(stem)
    return ", ".join(dict.fromkeys(names))


def file_context(path: str, needle: str) -> str:
    """CONTEXT_LINES either side of where the edit landed, read from disk
    after the write. Empty when the file, or the line, cannot be found —
    context is a help, never a requirement."""
    if CONTEXT_LINES <= 0 or not path or not needle:
        return ""
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return ""
    for i, line in enumerate(lines):
        if needle in line:
            lo = max(0, i - CONTEXT_LINES)
            return "\n".join(lines[lo : i + CONTEXT_LINES + 1])[:MAX_CONTEXT_CHARS]
    return ""


STRICT_PREAMBLE = "This edit was already judged possibly in breach of this rule. Decide it. "


def enclosing_block(path: str, needle: str) -> str:
    """The function or class the anchor line sits in, by indentation. Read
    after the edit, so it shows the code as the agent left it."""
    if not path or not needle:
        return ""
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return ""
    anchor = next((i for i, line in enumerate(lines) if needle in line), None)
    if anchor is None:
        return ""

    def indent(line: str) -> int:
        return len(line) - len(line.lstrip())

    depth = indent(lines[anchor])
    start = anchor
    for i in range(anchor - 1, -1, -1):
        if lines[i].strip() and indent(lines[i]) < depth:
            start = i
            depth = indent(lines[i])
            if depth == 0:
                break
    end = len(lines)
    for i in range(anchor + 1, len(lines)):
        if lines[i].strip() and indent(lines[i]) <= depth and i > start:
            end = i
            break
    return "\n".join(lines[start:end])[:MAX_BLOCK_CHARS]


def rule_question(rule: dict, strict: bool = False) -> dict:
    """One yes/no question per rule, in the user's own words. The answer is
    the probability the rule is broken. Asked as one concrete check —
    does the new code do the forbidden thing, or lack the required one —
    because "does this edit violate X" invites a judgment on pre-existing
    code the agent did not write."""
    pre = STRICT_PREAMBLE if strict else ""
    edit = rule.get("when") == "edit"
    scope_note = (
        "Judge only what the edit itself introduces, not pre-existing code."
        if edit
        else "Judge only what these changes introduce, not pre-existing code."
    )
    if (rule.get("polarity") or DEFAULT_POLARITY) == "require":
        what = "this edit" if edit else "these changes"
        return {
            "type": "noul",
            "instructions": pre + f"Does {what} add or change code that this rule "
            f"clearly covers, and do it WITHOUT what the rule "
            f'requires: "{rule["text"]}"? Answer yes only '
            f"when both hold and the requirement is plainly "
            f"missing from the new code and the surrounding "
            f"lines shown. If the rule does not apply to what "
            f"changed, or the requirement is met even "
            f"imperfectly, answer no.",
            "criteria": EDIT_REQUIRE_CRITERIA,
        }
    added = (
        "the ADDED or CHANGED code in this edit"
        if edit
        else "the ADDED or CHANGED code in these changes"
    )
    return {
        "type": "noul",
        "instructions": pre + f"Does {added} do what this rule forbids: "
        f'"{rule["text"]}"? {scope_note}',
        "criteria": EDIT_FORBID_CRITERIA,
    }


def verdict(answer: dict | None) -> float:
    p = (answer or {}).get("noul")
    return min(1.0, max(0.0, p)) if isinstance(p, (int, float)) else 0.0


def session_path(session_id: str) -> str:
    safe = re.sub(r"[^\w-]", "_", session_id or "unknown")
    return os.path.join(BLOCK_DIR, f"{safe}.json")


def session_state(session_id: str) -> dict:
    try:
        with open(session_path(session_id)) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"blocks": {}, "hunks": [], "files": [], "stop_blocks": 0}
    if "blocks" not in data:
        return {"blocks": data, "hunks": [], "files": [], "stop_blocks": 0}
    return data


def save_state(session_id: str, state: dict) -> None:
    try:
        os.makedirs(BLOCK_DIR, exist_ok=True)
        with open(session_path(session_id), "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def update_state(session_id: str, change):
    """Read, change, and write the session state under an exclusive lock, so
    parallel PostToolUse hooks cannot drop each other's hunks. Returns what
    `change` returns."""
    os.makedirs(BLOCK_DIR, exist_ok=True)
    lock_file = open(session_path(session_id) + ".lock", "w")
    if fcntl:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
    elif msvcrt:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
    try:
        state = session_state(session_id)
        result = change(state)
        save_state(session_id, state)
        return result
    finally:
        if fcntl:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        elif msvcrt:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLOCK, 1)
        lock_file.close()


def enter_turn(state: dict, turn: str) -> None:
    if state.get("turn") != turn:
        state["turn"], state["hunks"], state["files"] = turn, [], []
        state["partial"], state["snapshots"] = False, {}


def record_hunk(state: dict, turn: str, rel: str, hunk: str) -> None:
    """What the agent changed this turn, kept for the Stop-time turn check."""
    enter_turn(state, turn)
    total = sum(len(h) for h in state["hunks"])
    room = MAX_TURN_CHARS - total
    if room <= 0:
        return
    state["hunks"].append(f"--- {rel}\n{hunk[: min(MAX_HUNK_CHARS, room)]}")
    if rel not in state["files"]:
        state["files"].append(rel)


ADDED_HEAD_CHARS = 200


def added_body(hunk: str) -> str:
    """Just the lines the edit puts in the file, whichever shape the hunk
    came in: an Edit's ADDED section, a whole new file, or the + side of a
    git diff."""
    marker = "ADDED:\n"
    if marker in hunk:
        return hunk.split(marker, 1)[1]
    if hunk.startswith("NEW FILE"):
        return hunk.split("\n", 1)[1] if "\n" in hunk else ""
    if hunk.lstrip().startswith(("diff --git", "---", "@@", "index ")):
        return "\n".join(
            line[1:]
            for line in hunk.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
    return hunk


def added_head(hunk: str) -> str:
    """The start of what the edit put in the file. A later pass matches it
    against the file's next state: an unrelated edit to the same file is not
    a repair."""
    return added_body(hunk).strip()[:ADDED_HEAD_CHARS]


def input_digest(event: dict) -> str | None:
    """Fingerprint of the tool input, so a later pass can tell a retry of the
    same edit (the rule fought the agent) from a repair that landed."""
    try:
        payload = json.dumps(event.get("tool_input"), sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def rule_hashes(in_scope: list[dict]) -> dict:
    """Rule-file label -> content hash, for the files this call drew on."""
    return {r["file"]: r["file_hash"] for r in in_scope if r.get("file_hash")}


def log_decision(
    event: dict,
    answers: dict,
    probs: dict,
    violations: list,
    n_rules: int,
    n_scoped_out: int,
    phase: str,
    input_hash: str | None = None,
    blocked: list | None = None,
    hashes: dict | None = None,
    ms: int | None = None,
    n_irrelevant: int = 0,
    head: str | None = None,
    escalated: list | None = None,
    cmp_chars: dict | None = None,
    sg: str | None = None,
    user_answers: int = 0,
) -> None:
    try:
        with open(DEFAULT_LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "kind": "rules",
                        "phase": phase,
                        "session_id": event.get("session_id"),
                        "cwd": event.get("cwd"),
                        "file": (event.get("tool_input") or {}).get("file_path"),
                        "input_hash": input_hash,
                        "added_head": head,
                        "n_rules": n_rules,
                        "n_scoped_out": n_scoped_out,
                        "n_irrelevant": n_irrelevant,
                        "escalated": escalated or [],
                        "comparators": cmp_chars or {},
                        "sg": sg,
                        "ms": ms,
                        "probs": probs,
                        "violations": violations,
                        "blocked": blocked or [],
                        "rule_hashes": hashes or {},
                        "user_answers": user_answers,
                    }
                )
                + "\n"
            )
    except OSError:
        pass


def cite(v: dict) -> str:
    text = " ".join(v["text"].split())
    if len(text) > 220:
        text = text[:217] + "..."
    where = f"{v['file']} line {v['line']}" if v.get("line") else v["file"]
    kind = f"[{v.get('polarity') or DEFAULT_POLARITY}/{v.get('subject') or DEFAULT_SUBJECT}] "
    return f'- {kind}Rule "{v["rule"]}" from {where}: "{text}" ({v["prob"]:.2f})'


def ask_rules(state_text: str, rules: list[dict], strict: bool = False) -> dict:
    """One batched request, questions keyed by rule id."""
    if not rules:
        return {}
    questions, seen = {}, set()
    for r in rules:
        key = r["id"]
        n = 2
        while key in seen:
            key = f"{r['id']}-{n}"
            n += 1
        seen.add(key)
        r["_qkey"] = key
        questions[key] = rule_question(r, strict)
    return jev.ask(state_text, questions, timeout=budget())


def load_calib() -> dict:
    """Per-rule {median, n} over past real edits, written by
    `rules_eval.py report --write-calib`. Absent, every rule acts at ACT."""
    try:
        with open(CALIB_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


CALIB = load_calib()


def act_for(rule: dict, act: float = ACT, calib: dict | None = None) -> float:
    """The threshold this rule blocks at."""
    c = (CALIB if calib is None else calib).get(rule["id"])
    if not isinstance(c, dict) or act != ACT:
        return act
    median, n = c.get("median"), c.get("n", 0)
    if not isinstance(median, (int, float)):
        return act
    if n >= CALIB_MIN_CHECKS and median <= CALIB_DECISIVE:
        return ACT_DECISIVE
    if median >= CALIB_NOISY:
        return ACT_NOISY
    return act


def hits_from(rules: list[dict], probs: dict, act: float, flag: float) -> list[dict]:
    """Every rule at or above flag, banded against its own act threshold."""
    hits = []
    for r in rules:
        p = probs.get(r["id"], 0.0)
        if p >= flag:
            hits.append(
                {
                    "rule": r["id"],
                    "text": r["text"],
                    "file": r["file"],
                    "line": r.get("line", 0),
                    "polarity": r.get("polarity"),
                    "subject": r.get("subject"),
                    "prob": round(p, 2),
                    "band": "act" if p >= act_for(r, act) else "flag",
                }
            )
    return hits


def collect_verdicts(
    rules: list[dict], answers: dict, act: float, flag: float
) -> tuple[list[dict], dict]:
    """(hits at or above flag, every rule's probability for calibration)."""
    probs = {r["id"]: round(verdict(answers.get(r.get("_qkey") or r["id"])), 3) for r in rules}
    return hits_from(rules, probs, act, flag), probs


def scoped_rules(rules: list[dict], phase: str, files: list[str]) -> list[dict]:
    """The questions one request carries: rules for the phase whose scope
    covers a changed file, capped after scoping so a scoped rule is never
    crowded out by unscoped ones loaded before it."""
    hit = [
        r
        for r in rules
        if r.get("when") == phase
        and (not r["scope"] or any(glob_match(f, r["scope"]) for f in files))
    ]

    tiers: dict[tuple, dict[str, list]] = {}
    for r in hit:
        tier = (not r["scope"], r["file"].startswith("~/"))
        tiers.setdefault(tier, {}).setdefault(r["file"], []).append(r)
    out: list[dict] = []
    for tier in sorted(tiers):
        queues = list(tiers[tier].values())
        while queues and len(out) < MAX_RULES:
            for q in list(queues):
                if q:
                    out.append(q.pop(0))
                if not q:
                    queues.remove(q)
    return out[:MAX_RULES]


def judge_edit(
    rel: str,
    hunk: str,
    task: str,
    in_scope: list[dict],
    context: str = "",
    siblings: str = "",
    block: str = "",
    cwd: str = "",
    act: float = ACT,
    flag: float = FLAG,
) -> tuple[list[dict], dict, dict, list[dict], list[str], dict]:
    """One judged edit, no session side effects: (hits, probs, answers, rules
    skipped as irrelevant, rules escalated, comparator sizes by subject).
    The hook and the eval harness share this so they measure the same thing."""
    parts = [f"File: {rel}"]
    if task:
        parts.append(f"The user's current request: {task}")
    parts.append(f"The edit:\n{hunk[:MAX_STATE_CHARS]}")
    if context:
        parts.append(f"Surrounding lines after the edit:\n{context}")
    asked, skipped = split_relevant(in_scope, hunk, rel)
    if siblings and any(r.get("subject") == "imports_deps" for r in asked):
        names = ", ".join(siblings.split(", ")[:SIBLING_CAP])
        parts.append(f"Local modules importable from this file's directory: {names}")
    cmp_chars: dict = {}
    if COMPARATORS and cwd:
        for subject in dict.fromkeys(r.get("subject") for r in asked):
            found = comparators.comparator(subject or DEFAULT_SUBJECT, hunk, rel, cwd)
            if found:
                cmp_chars[subject] = len(found)
                parts.append(found)
    answers = ask_rules("\n\n".join(parts), asked)
    hits, probs = collect_verdicts(in_scope, answers, act, flag)

    escalated: list[str] = []
    undecided = [r for r in asked if flag <= probs.get(r["id"], 0.0) < act_for(r, act)]
    if ESCALATE and undecided and (budget() is None or budget() >= ESCALATE_MIN):
        extra = list(parts)
        if block:
            extra.append(f"The function or block this edit landed in, after the edit:\n{block}")
        around = "\n".join(f"[{r['id']}] {r['context']}" for r in undecided if r.get("context"))
        if around:
            extra.append(f"The instruction file says, around this rule:\n{around}")
        try:
            second = ask_rules("\n\n".join(extra), undecided, strict=True)
        except jev.JevError:
            second = {}
        if second:
            for r in undecided:
                probs[r["id"]] = round(verdict(second.get(r.get("_qkey") or r["id"])), 3)
                escalated.append(r["id"])
            hits = hits_from(in_scope, probs, act, flag)
    return hits, probs, answers, skipped, escalated, cmp_chars


def handle_edit(event: dict) -> dict:
    inp = event.get("tool_input") or {}
    cwd = event.get("cwd") or os.getcwd()
    file_path = inp.get("file_path") or ""
    rel = relative(file_path, cwd)
    if EXCLUDED.search(rel) or outside(rel):
        return {}
    rules = load_rules(cwd)
    if not rules:
        return {}
    in_scope = scoped_rules(rules, "edit", [rel])
    state = edit_hunks(inp, cwd).strip()
    if not state:
        return {}
    sid = event.get("session_id") or "unknown"
    turn, task, n_answers = last_user_prompt(event.get("transcript_path"))
    update_state(sid, lambda st: record_hunk(st, turn, rel, state))
    if not in_scope:
        return {}

    context = file_context(file_path, needle_of(inp))
    siblings = sibling_modules(file_path)
    block = enclosing_block(file_path, needle_of(inp)) if ESCALATE else ""

    sg = comparators.which()[1]
    t0 = time.monotonic()
    hits, probs, answers, skipped, escalated, cmp_chars = judge_edit(
        rel, state, task, in_scope, context, siblings, block, cwd
    )
    ms = int((time.monotonic() - t0) * 1000)

    def spend_blocks(st: dict) -> list:
        acting = []
        for v in hits:
            key = f"{v['rule']}|{rel}"
            if v["band"] == "act" and st["blocks"].get(key, 0) < MAX_BLOCKS:
                st["blocks"][key] = st["blocks"].get(key, 0) + 1
                acting.append(v)
        return acting

    acting = update_state(sid, spend_blocks) if hits else []
    flagged = [v for v in hits if v not in acting]
    log_decision(
        event,
        answers,
        probs,
        [{k: v[k] for k in ("rule", "file", "line", "prob", "band")} for v in hits],
        len(rules),
        len(rules) - len(in_scope),
        "edit",
        input_hash=input_digest(event),
        blocked=[v["rule"] for v in acting],
        hashes=rule_hashes(in_scope),
        ms=ms,
        n_irrelevant=len(skipped),
        head=added_head(state),
        escalated=escalated,
        cmp_chars=cmp_chars,
        sg=sg,
        user_answers=n_answers,
    )

    out = {}
    if flagged:
        listed = ", ".join(f"{v['rule']} {v['prob']:.2f}" for v in flagged)
        out["systemMessage"] = (
            f"[jev rules] uncertain about {listed} on {rel} — not sent to the agent"
        )
    if acting:
        lines = ["This edit appears to break a rule from this repository's instructions."]
        lines += [cite(v) for v in acting]
        lines.append(f"Repair {rel} now, then continue with the task.")
        out["decision"] = "block"
        out["reason"] = "\n".join(lines)
    return out


def git(cwd: str, *args: str, env: dict | None = None) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=4, env=env)
    if r.returncode != 0:
        raise subprocess.SubprocessError(r.stderr.strip())
    return r.stdout


def worktree_tree(cwd: str) -> tuple[str, str, str]:
    """The repository root, a tree object of its working tree as it is now,
    untracked files included, and a hash of the ignored paths git leaves out
    of that tree. The tree is built in a scratch copy of the index so the
    user's staging area is never touched."""
    root = git(cwd, "rev-parse", "--show-toplevel").strip()
    index = os.path.join(root, git(root, "rev-parse", "--git-path", "index").strip())
    scratch = os.path.join(BLOCK_DIR, f"index-{os.getpid()}")
    os.makedirs(BLOCK_DIR, exist_ok=True)
    try:
        if os.path.exists(index):
            shutil.copy2(index, scratch)
        env = {**os.environ, "GIT_INDEX_FILE": scratch}
        git(root, "add", "-A", env=env)
        ignored = git(
            root, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory"
        )
        return (
            root,
            git(root, "write-tree", env=env).strip(),
            hashlib.sha256(ignored.encode()).hexdigest(),
        )
    finally:
        if os.path.exists(scratch):
            os.remove(scratch)


def snapshot_key(event: dict) -> str:
    return event.get("tool_use_id") or json.dumps(event.get("tool_input"), sort_keys=True)


def handle_bash_before(event: dict) -> dict:
    """Snapshot the working tree before a shell command, so the matching
    PostToolUse can record what it changed. A failed snapshot marks the turn's
    diff as partial."""
    sid = event.get("session_id") or "unknown"
    cwd = event.get("cwd") or os.getcwd()
    turn, _, _ = last_user_prompt(event.get("transcript_path"))
    if not turn:
        return {}
    try:
        tree = worktree_tree(cwd)
    except (OSError, subprocess.SubprocessError):
        tree = None

    def keep(st: dict) -> None:
        enter_turn(st, turn)
        if tree:
            st["snapshots"][snapshot_key(event)] = tree
        else:
            st["partial"] = True

    update_state(sid, keep)
    return {}


def handle_bash_after(event: dict) -> dict:
    """Record each file a shell command changed as a hunk, like an edit."""
    sid = event.get("session_id") or "unknown"
    cwd = event.get("cwd") or os.getcwd()
    turn, _, _ = last_user_prompt(event.get("transcript_path"))
    key = snapshot_key(event)
    before = update_state(
        sid, lambda st: st.get("snapshots", {}).pop(key, None) if st.get("turn") == turn else None
    )
    if not turn or not before:
        return {}
    root, old, old_ignored = before
    try:
        _, new, new_ignored = worktree_tree(root)
        if new_ignored != old_ignored:
            update_state(sid, lambda st: st.update(partial=True))
        names = git(root, "diff", "--name-only", "-z", old, new).split("\0")
        hunks = []
        for name in filter(None, names):
            rel = relative(os.path.join(root, name), os.path.realpath(cwd))
            if EXCLUDED.search(rel) or outside(rel):
                continue
            hunk = git(root, "diff", "--no-color", "--no-ext-diff", "-U3", old, new, "--", name)
            hunks.append((rel, hunk))
    except (OSError, subprocess.SubprocessError):
        update_state(sid, lambda st: st.update(partial=True))
        return {}

    def keep(st: dict) -> None:
        for rel, hunk in hunks:
            record_hunk(st, turn, rel, hunk)

    if hunks:
        update_state(sid, keep)
    return {}


def handle_stop(event: dict) -> dict:
    """Turn rules judge the turn's changes as a whole — the questions a
    per-edit hunk can't answer (scope creep, an abstraction with one caller,
    a file that grew past its cap)."""
    sid = event.get("session_id") or "unknown"
    sstate = session_state(sid)
    turn, task, n_answers = last_user_prompt(event.get("transcript_path"))
    if not turn or sstate.get("turn") != turn or not (sstate["hunks"] or sstate.get("partial")):
        return {}
    cwd = event.get("cwd") or os.getcwd()
    rules = load_rules(cwd)
    turn_rules = scoped_rules(rules, "turn", sstate["files"])
    if not turn_rules:
        return {}

    diff = "\n\n".join(sstate["hunks"])
    parts = [f"Files changed this turn: {', '.join(sstate['files']) or 'none recorded'}"]
    if sstate.get("partial"):
        parts.append(
            "This turn also ran shell commands. Changes they made "
            "are not in the diff below, so it is partial."
        )
    if task:
        parts.append(f"The user's current request: {task}")
    parts.append(f"The changes:\n{diff[:MAX_TURN_CHARS]}")
    asked, skipped = split_relevant(turn_rules, diff, ", ".join(sstate["files"]))
    t0 = time.monotonic()
    answers = ask_rules("\n\n".join(parts), asked)
    ms = int((time.monotonic() - t0) * 1000)

    hits, probs = collect_verdicts(turn_rules, answers, ACT, FLAG)
    already = bool(event.get("stop_hook_active"))

    def spend_blocks(st: dict) -> list:
        acting = []
        for v in hits:
            if v["band"] == "act" and not already and st["stop_blocks"] < MAX_STOP_BLOCKS:
                st["stop_blocks"] += 1
                acting.append(v)
        return acting

    acting = update_state(sid, spend_blocks) if hits else []
    flagged = [v for v in hits if v not in acting]
    log_decision(
        event,
        answers,
        probs,
        [{k: v[k] for k in ("rule", "file", "line", "prob", "band")} for v in hits],
        len(rules),
        len(rules) - len(turn_rules),
        "turn",
        input_hash=None,
        blocked=[v["rule"] for v in acting],
        hashes=rule_hashes(turn_rules),
        ms=ms,
        n_irrelevant=len(skipped),
        user_answers=n_answers,
    )

    out = {}
    if flagged:
        listed = ", ".join(f"{v['rule']} {v['prob']:.2f}" for v in flagged)
        out["systemMessage"] = (
            f"[jev rules] uncertain about {listed} at end of turn — not sent to the agent"
        )
    if acting:
        files = ", ".join(sstate["files"])
        lines = [
            "The changes this turn appear to break a rule from this repository's instructions."
        ]
        lines += [cite(v) for v in acting]
        lines.append(f"Repair {files} before you finish. Keep the fix to what the rule asks.")
        out["decision"] = "block"
        out["reason"] = "\n".join(lines)
    return out


def main() -> None:
    global _DEADLINE
    event = {}
    try:
        _DEADLINE = time.monotonic() + HOOK_BUDGET
        if not jev.enabled("rules"):
            return
        event = json.load(sys.stdin)
        name = event.get("hook_event_name") or "PostToolUse"
        bash = event.get("tool_name") == "Bash"
        if name == "Stop":
            out = handle_stop(event)
        elif name == "PreToolUse":
            out = handle_bash_before(event) if bash else {}
        else:
            out = handle_bash_after(event) if bash else handle_edit(event)
        if out:
            json.dump(out, sys.stdout)
            sys.stdout.write("\n")
    except Exception as e:
        log_error(e, event)
        return


def log_error(e: Exception, event: dict) -> None:
    """A swallowed failure, logged so a missing decision row can be
    explained. A separate kind, so stats never scores it as a decision."""
    try:
        with open(DEFAULT_LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "kind": "rules-error",
                        "session_id": event.get("session_id"),
                        "event": event.get("hook_event_name"),
                        "path": (event.get("tool_input") or {}).get("file_path"),
                        "error": f"{type(e).__name__}: {e}"[:300],
                    }
                )
                + "\n"
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)
