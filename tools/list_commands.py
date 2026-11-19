"""Write docs/COMMANDS.md: every command the add-on shows, panel by panel.

    py tools/list_commands.py           # write the file
    py tools/list_commands.py --check   # exit 1 if the file is out of date

Read straight from the source (no Blender needed): the sidebar panels in the
order ui.classes registers them, the buttons each panel's draw() can show
(conditional ones included, helpers it calls followed), the menus the add-on
adds to, and for each operator its label, tooltip, modes (from its poll) and
whether it runs in the viewport (has a modal). Operators no panel or menu
shows are listed at the end, so a button cannot go missing unnoticed.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "spline_surface")
OUT = os.path.join(ROOT, "docs", "COMMANDS.md")
PREFIX = "spline_surface."

# Menus the add-on appends to: (module, function, where it shows)
MENUS = [("__init__", "_add_menu", "3D ビュー → Add → Surface（オブジェクトモード）"),
         ("curveediting", "_context_menu", "カーブの編集モード → 右クリックメニュー")]

# The conditions in the panels' draw code, in words (unknown ones are shown as code).
CONDITIONS = {
    "meta is not None": "Extend・Bridge Edges・Corner Fill で作った面がアクティブのとき",
    "entries": "アクティブな面に Match Edge の追従があるとき（追従ごとに1行）",
    "relinks": "シーンを Full Copy して、元のシーンのカーブを読んでいる面があるとき",
    "on": "Panel Gap がオンのとき",
    "s.diagnose_hidden": "Diagnose で「意図した形」の印を付けた所があるとき",
}

PANEL_NOTES = {
    "SS_PT_curves": "カーブ編集（v0.83〜）。編集モードでは Slide・Dissolve・Constrained Move が加わる",
}


def _modules():
    out = {}
    for name in os.listdir(PKG):
        if name.endswith(".py"):
            path = os.path.join(PKG, name)
            src = open(path, encoding="utf-8").read()
            out[name[:-3]] = (src, ast.parse(src))
    return out


def _const(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _operators(mods):
    """{idname: info} for every operator class, base classes in the same module followed."""
    found = {}
    for mod, (src, tree) in mods.items():
        classes = {c.name: c for c in tree.body if isinstance(c, ast.ClassDef)}

        def attr(cls, key, seen=()):
            for node in cls.body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == key for t in node.targets):
                    return _const(node.value)
                if isinstance(node, ast.FunctionDef) and node.name == key:
                    return node
            for b in cls.bases:
                name = getattr(b, "id", None)
                if name in classes and name not in seen:
                    v = attr(classes[name], key, seen + (name,))
                    if v is not None:
                        return v
            return None

        for cls in classes.values():
            idname = attr(cls, "bl_idname")
            if not (isinstance(idname, str) and idname.startswith(PREFIX)):
                continue
            if not any(getattr(b, "attr", None) == "Operator" for b in cls.bases):
                continue
            poll = attr(cls, "poll")
            poll_src = ast.get_source_segment(src, poll) if isinstance(poll, ast.FunctionDef) else ""
            if "EDIT_CURVE" in poll_src and "OBJECT" in poll_src:
                mode = "オブジェクト／編集"
            elif "EDIT_CURVE" in poll_src:
                mode = "カーブの編集モード"
            elif "OBJECT" in poll_src:
                mode = "オブジェクト"
            else:
                mode = "どのモードでも"
            desc = attr(cls, "bl_description") or ""
            found[idname] = {"label": attr(cls, "bl_label") or idname, "desc": " ".join(desc.split()),
                             "mode": mode, "modal": isinstance(attr(cls, "modal"), ast.FunctionDef),
                             "module": mod}
    return found


def _module_const(mods, mod, node):
    """The value of a literal, or of a module-level constant (NAME or module.NAME)."""
    v = _const(node)
    if v is not None:
        return v
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in mods:
        mod, name = node.value.id, node.attr
    elif isinstance(node, ast.Name):
        name = node.id
    else:
        return None
    for n in mods[mod][1].body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return _const(n.value)
    return None


def _bind(target, value):
    if isinstance(target, ast.Name):
        return {target.id: value}
    if isinstance(target, ast.Tuple) and isinstance(value, (tuple, list)):
        out = {}
        for t, v in zip(target.elts, value):
            out.update(_bind(t, v))
        return out
    return {}


def _eval(node, env):
    """A string label from constants, loop variables and +; None if it depends on more."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _eval(node.left, env), _eval(node.right, env)
        return a + b if isinstance(a, str) and isinstance(b, str) else None
    return None


def _label_text(mods, mod, src, node, parents):
    """How the button's text reads: a plain string, 'A / B / C' for a loop, or
    'X (「Y」 while on)' for an on/off label; None: the operator's own label."""
    if isinstance(node, ast.IfExp):
        on, off = _const(node.body), _const(node.orelse)
        if isinstance(on, str) and isinstance(off, str):
            return "%s（オンの間は「%s」）" % (off, on)
    v = _const(node)
    if isinstance(v, str):
        return v
    p = parents.get(node)
    while p is not None and not isinstance(p, ast.For):
        p = parents.get(p)
    if p is not None:
        values = _module_const(mods, mod, p.iter)
        if isinstance(values, (list, tuple)):
            labels = [_eval(node, _bind(p.target, v)) for v in values]
            if all(isinstance(x, str) for x in labels):
                return " / ".join(labels)
    return "(" + ast.get_source_segment(src, node) + ")"


def _condition(src, node, parents, stop, mods=None, mod=None):
    """When a button is shown: from the if blocks around it (inside the draw function),
    and 'one per row' when it sits in a loop over results."""
    conds, per_row = [], False
    child, p = node, parents.get(node)
    while p is not None and p is not stop:
        if isinstance(p, ast.For) and child in p.body and (
                mods is None or not isinstance(_module_const(mods, mod, p.iter), (list, tuple))):
            per_row = True
        if isinstance(p, ast.If) and child in p.body:
            conds.append(ast.get_source_segment(src, p.test))
        elif isinstance(p, ast.If) and child in p.orelse:
            conds.append("not (" + ast.get_source_segment(src, p.test) + ")")
        child, p = p, parents.get(p)
    row = "一覧の行ごと（結果があるとき）" if per_row else ""
    if not conds:
        return row
    text = " and ".join(reversed(conds))
    if "EDIT_CURVE" in text and "not" not in text:
        words = "カーブの編集モードのとき"
    elif text in CONDITIONS:
        words = CONDITIONS[text]
    else:
        words = "条件付き：`%s`" % text.replace("|", "\\|")
    return words if not row or "1行" in words else words + "、" + row


def _calls(mods, mod, fn_node, seen=(), cond=""):
    """[(line, kind, data)] in source order: ('op', (idname, text, when)) and ('label', text),
    following the add-on's own draw helpers it calls."""
    src = mods[mod][0]
    funcs = {f.name: f for f in mods[mod][1].body if isinstance(f, ast.FunctionDef)}
    parents = {}
    for n in ast.walk(fn_node):
        for c in ast.iter_child_nodes(n):
            parents[c] = n
    out = []
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        where = (node.lineno, node.col_offset)
        if isinstance(f, ast.Attribute) and f.attr == "operator" and node.args:
            idname = _const(node.args[0])
            if isinstance(idname, str) and idname.startswith(PREFIX):
                text = None
                for kw in node.keywords:
                    if kw.arg == "text":
                        text = _label_text(mods, mod, src, kw.value, parents)
                when = _condition(src, node, parents, fn_node, mods, mod) or cond
                if text is None:            # a loop around a button without its own text
                    q = parents.get(node)
                    while q is not None and not isinstance(q, ast.For):
                        q = parents.get(q)
                out.append((where, "op", (idname, text, when)))
        elif isinstance(f, ast.Attribute) and f.attr == "label":
            for kw in node.keywords:
                v = _const(kw.value) if kw.arg == "text" else None
                if isinstance(v, str) and v.endswith(":"):
                    out.append((where, "label", v))
        elif isinstance(f, ast.Name) and f.id in funcs and f.id not in seen:
            when = _condition(src, node, parents, fn_node, mods, mod) or cond
            out += [(where, k, d) for _, k, d in _calls(mods, mod, funcs[f.id], seen + (f.id,), when)]
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in mods \
                and f.value.id != mod:
            other = {g.name: g for g in mods[f.value.id][1].body if isinstance(g, ast.FunctionDef)}
            if f.attr in other and f.attr.startswith("draw"):
                when = _condition(src, node, parents, fn_node, mods, mod) or cond
                out += [(where, k, d) for _, k, d in _calls(mods, f.value.id, other[f.attr], seen, when)]
    out.sort(key=lambda x: x[0])
    return out


def _panels(mods):
    src, tree = mods["ui"]
    classes = {c.name: c for c in tree.body if isinstance(c, ast.ClassDef)}
    order = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "classes" for t in node.targets):
            order = [e.id for e in node.value.elts]
    out = []
    for name in order:
        cls = classes[name]
        if not name.startswith("SS_PT_"):
            continue
        label = next((_const(n.value) for n in cls.body if isinstance(n, ast.Assign)
                      and any(getattr(t, "id", None) == "bl_label" for t in n.targets)), name)
        parent = next((_const(n.value) for n in cls.body if isinstance(n, ast.Assign)
                       and any(getattr(t, "id", None) == "bl_parent_id" for t in n.targets)), None)
        draw = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "draw"), None)
        out.append((name, label, parent, _calls(mods, "ui", draw) if draw else []))
    return out


def _esc(s):
    return s.replace("|", "\\|")


def build():
    mods = _modules()
    ops = _operators(mods)
    shown = set()
    lines = ["# コマンド一覧（UI の棚卸し）", "",
             "サイドバー（N キー → **Spline Surface** タブ）とメニューに出る操作を、画面に出る順に全部並べた一覧です。"
             "実機で確かめるときのチェック表として使えます（確認の列に ✔ を付けていく）。",
             "",
             "- **ボタン**：画面に出る名前。（ ）の中は、状況で変わる名前",
             "- **モード**：押せるモード。オブジェクト／編集＝オブジェクトモードとカーブの編集モードの両方。どのモードでも＝モードの制限なし（表示の切り替えなど）",
             "- **種類**：**ビュー** は押すと 3D ビューでクリック・ドラッグする操作（Esc でやめる）。**即時** は押すとすぐ実行（多くは左下の「最後の操作を調整」で値を変えられる）",
             "- **出るとき**：いつも出ているか、ある状態のときだけ出るか（条件はソースの式のまま）",
             "- **説明**：ボタンにマウスを乗せると出る説明（英語）と同じ文。使い方の詳しい説明は [説明書](MANUAL.md)",
             "",
             "この一覧はソースから作っています（`py tools/list_commands.py`）。ボタンを足したのに作り直していないと、テストが失敗します。",
             ""]
    for name, label, parent, calls in _panels(mods):
        if name == "SS_PT_main":
            continue
        head = "## %s" % label if parent in (None, "SS_PT_main") else "### %s（%s の中）" % (label, parent.replace("SS_PT_", ""))
        rows = []
        section = ""
        seen_here = set()
        for _, kind, data in calls:
            if kind == "label":
                section = data.rstrip(":")
                continue
            idname, text, when = data
            key = (idname, text)
            if key in seen_here:
                continue
            seen_here.add(key)
            shown.add(idname)
            info = ops.get(idname, {"label": idname, "desc": "", "mode": "?", "modal": False})
            button = text if (text and not text.startswith("(")) else info["label"]
            if text and text.startswith("("):
                button = "%s（名前は状況で変わる）" % info["label"]
            if text == "":
                button = "（アイコンだけ）%s" % info["label"]
            rows.append("| %s | %s | `%s` | %s | %s | %s | %s | |" % (
                _esc(section), _esc(button), idname[len(PREFIX):], info["mode"],
                "ビュー" if info["modal"] else "即時", when or "いつも", _esc(info["desc"])))
        if not rows:
            continue
        lines += [head, ""]
        if name in PANEL_NOTES:
            lines += [PANEL_NOTES[name], ""]
        lines += ["| まとまり | ボタン | 操作 | モード | 種類 | 出るとき | 説明 | 確認 |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += rows + [""]
    lines += ["## メニュー", "", "| 場所 | ボタン | 操作 | モード | 種類 | 確認 |", "|---|---|---|---|---|---|"]
    for mod, fn, where in MENUS:
        node = next(f for f in mods[mod][1].body if isinstance(f, ast.FunctionDef) and f.name == fn)
        for _, kind, data in _calls(mods, mod, node):
            if kind != "op":
                continue
            idname = data[0]
            shown.add(idname)
            info = ops.get(idname, {"label": idname, "mode": "?", "modal": False})
            lines.append("| %s | %s | `%s` | %s | %s | |" % (where, _esc(info["label"]), idname[len(PREFIX):],
                                                         info["mode"], "ビュー" if info["modal"] else "即時"))
    lines.append("")
    rest = sorted(set(ops) - shown)
    lines += ["## パネルやメニューに出ないもの", "",
              "ほかの操作の中から呼ばれる、または一覧の行の中のボタンとして出るものです。",
              "", "| 操作 | 名前 | 説明 |", "|---|---|---|"]
    lines += ["| `%s` | %s | %s |" % (i[len(PREFIX):], _esc(ops[i]["label"]), _esc(ops[i]["desc"])) for i in rest]
    if not rest:
        lines += ["| （なし：すべての操作がパネルかメニューに出ています） | | |"]
    lines += ["", "合計：%d の操作（パネル・メニューに出るもの %d）" % (len(ops), len(set(ops) & shown)), ""]
    return "\n".join(lines)


def main():
    text = build()
    if "--check" in sys.argv:
        old = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if old != text:
            print("docs/COMMANDS.md is out of date: run  py tools/list_commands.py")
            sys.exit(1)
        print("docs/COMMANDS.md is up to date")
        return
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
