"""Renderers: terminal, markdown, and a self-contained HTML report card.

The HTML exists because the gap between two numbers is a thing you should be
able to point at in a lab meeting.  It embeds its own hand-written SVG -- no
libraries, no fonts, no network -- so it survives being emailed to a
collaborator or committed next to the results it describes.
"""
from __future__ import annotations

import html
import json

import numpy as np

__all__ = ["to_text", "to_markdown", "to_html", "erase_text"]

BLOCK, EMPTY = "█", "░"


def _fmt(v, nd=3):
    return "  n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _bar(v, width=26):
    if not np.isfinite(v):
        return " " * width
    n = int(round(max(0.0, min(1.0, v)) * width))
    return BLOCK * n + EMPTY * (width - n)


# --------------------------------------------------------------------- terminal
def to_text(r, color: bool = True) -> str:
    C = {"dim": "\033[2m", "b": "\033[1m", "red": "\033[31m", "yel": "\033[33m",
         "grn": "\033[32m", "off": "\033[0m"} if color else \
        {k: "" for k in ("dim", "b", "red", "yel", "grn", "off")}
    tone = {"SEVERE": C["red"], "MODERATE": C["yel"], "LOW": C["grn"]}[r.level]
    L, out = 72, []
    add = out.append

    add(f"{C['b']}leakcheck{C['off']} {r.version}  {C['dim']}{r.created}{C['off']}")
    add(f"{r.dataset}   label={r.label_name} ({r.design.n_classes} classes)   "
        f"group={r.group_name} ({r.design.n_groups})   {r.design.n_clips} clips")
    add(C["dim"] + "─" * L + C["off"])

    rows = [("shuffled", r.shuffled.accuracy, r.shuffled.chance, r.shuffled.split,
             "what a random split reports"),
            ("honest", r.honest.accuracy, r.honest.chance, r.honest.split,
             "what you get on a new " + r.group_name),
            ("identity", r.identity.accuracy, r.identity.chance,
             f"StratifiedKFold({r.config.folds})",
             f"how well the same features name the {r.group_name}")]
    for name, val, chance, split, why in rows:
        add(f"  {C['b']}{name:<9}{C['off']} {_fmt(val)}  {_bar(val)}  "
            f"{C['dim']}chance {_fmt(chance)}{C['off']}")
        add(f"  {C['dim']}{'':<9} {split} — {why}{C['off']}")

    add(C["dim"] + "─" * L + C["off"])
    lf = r.leak_fraction
    frac = "" if not np.isfinite(lf) else f"   {max(0.0, min(1.0, lf)):.0%} of the apparent skill"
    add(f"  {C['b']}inflation{C['off']} {r.inflation:+.3f}   {tone}{C['b']}{r.level}{C['off']}{C['dim']}{frac}{C['off']}")
    add("")

    for line in r.verdict():
        add(_wrap(line, L - 4, "  • ", "    "))

    if r.design.warnings:
        add("")
        add(f"  {C['b']}study design{C['off']}")
        for wline in r.design.warnings:
            add(_wrap(wline, L - 4, "  ! ", "    "))

    pg = r.honest.per_group
    if len(pg) >= 3:
        add("")
        add(f"  {C['b']}per-{r.group_name}{C['off']} {C['dim']}(held-out score, worst first){C['off']}")
        show = pg[:5] + ([None] + pg[-2:] if len(pg) > 7 else pg[5:])
        for g in show:
            if g is None:
                add(f"  {C['dim']}      ⋮{C['off']}")
                continue
            flag = "" if g.metric == "balanced_accuracy" else f"  {C['dim']}(single-class: recall){C['off']}"
            add(f"    {g.group:<18} {_fmt(g.score)}  {_bar(g.score, 18)}  "
                f"{C['dim']}n={g.n}{C['off']}{flag}")

    add("")
    add(f"  {C['dim']}probe: logistic regression C={r.config.C}"
        f"{', class_weight=balanced' if r.config.balanced else ''}, "
        f"scaler fitted per fold  |  {r.environment}{C['off']}")
    return "\n".join(out)


def _wrap(text, width, first, cont):
    words, lines, cur = text.split(), [], first
    for w in words:
        if len(cur) + len(w) + 1 > width + len(cont) and cur.strip() not in (first.strip(),):
            lines.append(cur.rstrip()); cur = cont
        cur += w + " "
    lines.append(cur.rstrip())
    return "\n".join(lines)


# --------------------------------------------------------------------- markdown
def to_markdown(r) -> str:
    o = [f"# leakcheck report — {r.dataset}", "",
         f"`label={r.label_name}` · `group={r.group_name}` · "
         f"{r.design.n_clips} clips · {r.design.n_classes} classes · "
         f"{r.design.n_groups} groups · leakcheck {r.version}, {r.created}", "",
         "| | score | chance | split | what it is |", "|---|---|---|---|---|",
         f"| **shuffled** | {_fmt(r.shuffled.accuracy)} | {_fmt(r.shuffled.chance)} | "
         f"{r.shuffled.split} | what a random split reports |",
         f"| **honest** | {_fmt(r.honest.accuracy)} | {_fmt(r.honest.chance)} | "
         f"{r.honest.split} | what you get on a new {r.group_name} |",
         f"| **identity** | {_fmt(r.identity.accuracy)} | {_fmt(r.identity.chance)} | "
         f"StratifiedKFold({r.config.folds}) | how well the same features name the {r.group_name} |",
         "", f"**inflation {r.inflation:+.3f} — {r.level}**", ""]
    o += [f"- {v}" for v in r.verdict()]
    if r.design.warnings:
        o += ["", "## Study design", ""] + [f"- ⚠️ {w}" for w in r.design.warnings]
    if len(r.honest.per_group) >= 3:
        o += ["", f"## Per-{r.group_name}", "",
              f"| {r.group_name} | n | held-out score | metric |", "|---|---|---|---|"]
        o += [f"| {g.group} | {g.n} | {_fmt(g.score)} | {g.metric} |" for g in r.honest.per_group]
    return "\n".join(o) + "\n"


# --------------------------------------------------------------------- html
_CSS = """
/* Neutrals carry a slight slate bias toward the accent, so the greys read as
   chosen rather than inherited. Semantic colours (severity) are deliberately
   separate from the accent, which is reserved for "the mean". */
:root{--bg:#f8f9fb;--card:#ffffff;--fg:#111721;--mut:#5a6675;--line:#dfe4ea;
      --sev:#bc3b2e;--mod:#a8781a;--low:#2c7a53;--accent:#2f6fbd}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
      --bg:#0e1216;--card:#151b22;--fg:#e4e8ee;--mut:#93a0af;--line:#232a33;
      --sev:#ef7263;--mod:#dfb04a;--low:#5cbf88;--accent:#77abe2}}
:root[data-theme="dark"]{--bg:#0e1216;--card:#151b22;--fg:#e4e8ee;--mut:#93a0af;
      --line:#232a33;--sev:#ef7263;--mod:#dfb04a;--low:#5cbf88;--accent:#77abe2}

*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:0 1.25rem 5rem;
     font:16px/1.62 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;
     -webkit-font-smoothing:antialiased}
main{max-width:52rem;margin:0 auto;display:flex;flex-direction:column;gap:0}
.stripe{height:4px;margin:0 -1.25rem 2.5rem;background:var(--line)}
.stripe.SEVERE{background:var(--sev)}.stripe.MODERATE{background:var(--mod)}
.stripe.LOW{background:var(--low)}
.eyebrow{font-size:.68rem;font-weight:650;letter-spacing:.1em;text-transform:uppercase;
     color:var(--mut);margin:0 0 .5rem}
h1{font-size:1.75rem;line-height:1.2;margin:0 0 .6rem;letter-spacing:-.018em;text-wrap:balance}
h2{font-size:.95rem;font-weight:650;letter-spacing:.02em;margin:2.75rem 0 .5rem;
   padding-bottom:.4rem;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);font-size:.9rem;margin:0 0 1.75rem;line-height:1.7}
code,.mono,td.num{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
     font-variant-numeric:tabular-nums}
code{font-size:.85em;background:var(--card);border:1px solid var(--line);
     border-radius:4px;padding:.08em .38em}
figure{margin:1.25rem 0 0;padding:1.4rem 1.25rem;background:var(--card);
       border:1px solid var(--line);border-radius:8px;overflow-x:auto}
figure svg{display:block;max-width:100%;height:auto;color:var(--fg)}
figcaption{color:var(--mut);font-size:.855rem;margin-top:1rem;line-height:1.6;
       max-width:46rem}
.badge{display:inline-block;padding:.18rem .55rem;border-radius:3px;font-size:.7rem;
       font-weight:700;letter-spacing:.07em;color:var(--bg)}
.badge.SEVERE{background:var(--sev)}.badge.MODERATE{background:var(--mod)}
.badge.LOW{background:var(--low)}
ul{padding-left:1.1rem;margin:.75rem 0 0}
li{margin:.55rem 0;max-width:46rem}
.verdict li::marker{color:var(--accent)}
.warn li::marker{color:var(--mod)}
table{border-collapse:collapse;width:100%;font-size:.885rem;margin-top:.75rem}
th,td{text-align:left;padding:.42rem .6rem;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-size:.68rem;font-weight:650;letter-spacing:.08em;
   text-transform:uppercase}
td.num{text-align:right}
tbody tr:last-child td{border-bottom:none}
pre.mono{background:var(--card);border:1px solid var(--line);border-radius:8px;
     padding:1rem 1.1rem;overflow-x:auto;font-size:.8rem;line-height:1.55;margin-top:.75rem}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);
       color:var(--mut);font-size:.8rem;line-height:1.65}
"""


def _svg_three_numbers(r) -> str:
    """The whole argument in one picture: two bars for the same task under two
    splits, the gap between them measured, and the group probe that explains it."""
    x0, w, W = 168, 420, 640
    def px(v):
        return x0 + w * max(0.0, min(1.0, v))
    ys = {"shuffled": 46, "honest": 96, "identity": 176}
    tone = {"SEVERE": "var(--sev)", "MODERATE": "var(--mod)", "LOW": "var(--low)"}[r.level]
    p = []
    p.append(f'<svg viewBox="0 0 {W} 250" role="img" xmlns="http://www.w3.org/2000/svg" '
             f'aria-label="Shuffled split scores {r.shuffled.accuracy:.3f}, groups held out '
             f'{r.honest.accuracy:.3f}, a gap of {r.inflation:.3f}; the same features identify '
             f'the {r.group_name} at {r.identity.accuracy:.3f} against chance '
             f'{r.identity.chance:.3f}.">')
    p.append('<defs><marker id="lc-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
             'markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>'
             '</marker></defs>')
    p.append('<g font-size="12" fill="currentColor">')

    for name, key, val, chance, note in [
            ("shuffled split", "shuffled", r.shuffled.accuracy, r.shuffled.chance, "reported"),
            ("groups held out", "honest", r.honest.accuracy, r.honest.chance, "deployed"),
            (f"{r.group_name} identity", "identity", r.identity.accuracy, r.identity.chance,
             "the shortcut")]:
        y = ys[key]
        col = tone if key == "identity" else "currentColor"
        p.append(f'<text x="{x0 - 12}" y="{y + 12}" text-anchor="end" font-weight="600">'
                 f'{html.escape(name)}</text>')
        p.append(f'<text x="{x0 - 12}" y="{y + 27}" text-anchor="end" fill="var(--mut)" '
                 f'font-size="10.5">{note}</text>')
        p.append(f'<rect x="{x0}" y="{y}" width="{w}" height="17" rx="3" fill="currentColor" '
                 f'opacity="0.07"/>')
        if np.isfinite(val):
            p.append(f'<rect x="{x0}" y="{y}" width="{px(val) - x0:.1f}" height="17" rx="3" '
                     f'fill="{col}" opacity="{0.85 if key != "identity" else 1}"/>')
            p.append(f'<text x="{px(val) + 8:.1f}" y="{y + 13}" font-weight="600" '
                     f'font-family="ui-monospace,Menlo,monospace">{val:.3f}</text>')
        if np.isfinite(chance):
            cx = px(chance)
            p.append(f'<line x1="{cx:.1f}" y1="{y - 5}" x2="{cx:.1f}" y2="{y + 22}" '
                     f'stroke="currentColor" stroke-width="1.5" stroke-dasharray="3 3" '
                     f'opacity="0.55"/>')
            p.append(f'<text x="{cx:.1f}" y="{y - 9}" text-anchor="middle" font-size="10" '
                     f'fill="var(--mut)">chance</text>')

    # the gap, measured
    xs, xh = px(r.shuffled.accuracy), px(r.honest.accuracy)
    ymid = ys["honest"] + 8
    p.append(f'<line x1="{xh:.1f}" y1="{ys["shuffled"] + 17}" x2="{xh:.1f}" y2="{ymid + 22}" '
             f'stroke="{tone}" stroke-width="1" stroke-dasharray="2 3"/>')
    p.append(f'<line x1="{xh:.1f}" y1="{ymid + 22}" x2="{xs:.1f}" y2="{ymid + 22}" '
             f'stroke="{tone}" stroke-width="2" marker-end="url(#lc-a)"/>')
    p.append(f'<text x="{(xh + xs) / 2:.1f}" y="{ymid + 38}" text-anchor="middle" '
             f'font-size="11.5" font-weight="700" fill="{tone}">'
             f'inflation {r.inflation:+.3f}</text>')

    p.append(f'<line x1="{x0 - 150}" y1="152" x2="{W - 12}" y2="152" stroke="currentColor" '
             f'opacity="0.18"/>')
    p.append('</g></svg>')
    return "".join(p)


def _svg_per_group(r) -> str:
    pg = r.honest.per_group
    if len(pg) < 3:
        return ""
    x0, w, W, y = 40, 560, 640, 74
    def px(v):
        return x0 + w * max(0.0, min(1.0, v))
    worst, best = pg[0], pg[-1]
    p = [f'<svg viewBox="0 0 {W} 132" role="img" xmlns="http://www.w3.org/2000/svg" '
         f'aria-label="One dot per {r.group_name}. The mean held-out score is '
         f'{r.honest.accuracy:.3f} but the worst {r.group_name}, {html.escape(worst.group)}, '
         f'scores {worst.score:.3f}.">',
         '<g font-size="11" fill="currentColor">']
    p.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + w}" y2="{y}" stroke="currentColor" opacity="0.25"/>')
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        p.append(f'<line x1="{px(t):.1f}" y1="{y - 4}" x2="{px(t):.1f}" y2="{y + 4}" '
                 f'stroke="currentColor" opacity="0.3"/>')
        p.append(f'<text x="{px(t):.1f}" y="{y + 20}" text-anchor="middle" fill="var(--mut)">'
                 f'{t:.2f}</text>')
    cx = px(r.honest.chance)
    p.append(f'<line x1="{cx:.1f}" y1="{y - 30}" x2="{cx:.1f}" y2="{y + 8}" stroke="currentColor" '
             f'stroke-dasharray="3 3" opacity="0.6"/>')
    p.append(f'<text x="{cx:.1f}" y="{y - 35}" text-anchor="middle" fill="var(--mut)">chance</text>')
    mx = px(r.honest.accuracy)
    p.append(f'<line x1="{mx:.1f}" y1="{y - 30}" x2="{mx:.1f}" y2="{y + 8}" stroke="var(--accent)" '
             f'stroke-width="2"/>')
    p.append(f'<text x="{mx:.1f}" y="{y - 35}" text-anchor="middle" fill="var(--accent)" '
             f'font-weight="600">mean {r.honest.accuracy:.3f}</text>')
    for g in pg:
        p.append(f'<circle cx="{px(g.score):.1f}" cy="{y}" r="4.5" fill="currentColor" '
                 f'opacity="0.45"><title>{html.escape(g.group)}: {g.score:.3f} (n={g.n})</title>'
                 f'</circle>')
    p.append(f'<circle cx="{px(worst.score):.1f}" cy="{y}" r="8.5" fill="none" '
             f'stroke="var(--sev)" stroke-width="2"/>')
    p.append(f'<text x="{px(worst.score):.1f}" y="{y + 38}" text-anchor="middle" '
             f'fill="var(--sev)" font-weight="600">{html.escape(worst.group)} {worst.score:.3f}</text>')
    p.append(f'<text x="{px(best.score):.1f}" y="{y - 14}" text-anchor="middle" fill="var(--mut)">'
             f'{html.escape(best.group)} {best.score:.3f}</text>')
    p.append('</g></svg>')
    return "".join(p)


def to_html(r) -> str:
    lf = r.leak_fraction
    frac = "" if not np.isfinite(lf) else \
        f" &middot; {max(0.0, min(1.0, lf)):.0%} of the apparent skill"
    o = [f"<title>{html.escape(r.dataset)}</title>",
         f"<style>{_CSS}</style>", f'<div class="stripe {r.level}"></div>', "<main>",
         '<p class="eyebrow">leakcheck report</p>',
         f"<h1>{html.escape(r.dataset)}</h1>",
         f'<p class="sub"><span class="badge {r.level}">{r.level}</span>&nbsp; '
         f'inflation {r.inflation:+.3f}{frac}<br>'
         f'<code>label={html.escape(r.label_name)}</code> '
         f'<code>group={html.escape(r.group_name)}</code> &middot; '
         f'{r.design.n_clips:,} clips &middot; {r.design.n_classes} classes &middot; '
         f'{r.design.n_groups} groups</p>',
         "<figure>", _svg_three_numbers(r),
         f"<figcaption>The same features and the same probe, scored two ways. "
         f"A random split of the clips reports {r.shuffled.accuracy:.3f}; holding out whole "
         f"{html.escape(r.group_name)}s gives {r.honest.accuracy:.3f}. The bottom bar is the "
         f"explanation: those features identify the {html.escape(r.group_name)} itself at "
         f"{r.identity.accuracy:.3f} where chance is {r.identity.chance:.3f}.</figcaption>",
         "</figure>", "<h2>What this means</h2>", '<div class="verdict"><ul>']
    o += [f"<li>{html.escape(v)}</li>" for v in r.verdict()]
    o += ["</ul></div>"]

    if len(r.honest.per_group) >= 3:
        o += ["<h2>Per-" + html.escape(r.group_name) + "</h2>", "<figure>", _svg_per_group(r),
              f"<figcaption>One dot per {html.escape(r.group_name)}, held-out score. "
              f"The mean is a summary of this spread, not a promise to any single one of "
              f"them. Hover a dot for its name and clip count.</figcaption>", "</figure>"]

    if r.design.warnings:
        o += ["<h2>Study design</h2>", '<ul class="warn">']
        o += [f"<li>{html.escape(w)}</li>" for w in r.design.warnings]
        o += ["</ul>"]

    if len(r.honest.per_group) >= 3:
        o += ["<h2>Full table</h2>",
              f"<table><thead><tr><th>{html.escape(r.group_name)}</th><th>clips</th>"
              f"<th>held-out score</th><th>metric</th></tr></thead><tbody>"]
        o += [f"<tr><td>{html.escape(g.group)}</td><td class='num'>{g.n}</td>"
              f"<td class='num'>{g.score:.3f}</td><td>{g.metric}</td></tr>"
              for g in r.honest.per_group]
        o += ["</tbody></table>"]

    o += ["<h2>Reproducing this</h2>",
          f"<pre class='mono'>{html.escape(json.dumps(r.to_dict()['config'], indent=2))}</pre>",
          f"<footer>leakcheck {html.escape(r.version)} &middot; {html.escape(r.created)} "
          f"&middot; {html.escape(r.environment)}<br>Probe: logistic regression, scaler fitted "
          f"inside each fold. Task scores are balanced accuracy; identity is plain accuracy "
          f"with chance = 1/groups.</footer>", "</main>"]
    return "\n".join(o)


# --------------------------------------------------------------------- erase
def erase_text(e, group_name: str = "group", color: bool = True) -> str:
    C = {"b": "\033[1m", "dim": "\033[2m", "off": "\033[0m"} if color else \
        {"b": "", "dim": "", "off": ""}
    o = [f"{C['b']}{e.method}{C['off']}  rank {e.rank}",
         "─" * 72,
         f"  task      {_fmt(e.task_before)} → {_fmt(e.task_after)}   "
         f"{C['dim']}(chance {_fmt(e.chance_task)}) must hold{C['off']}",
         f"  {group_name:<9} {_fmt(e.identity_before)} → {_fmt(e.identity_after)}   "
         f"{C['dim']}(floor {_fmt(e.floor_identity)}: chance {_fmt(e.chance_identity)}, "
         f"majority {_fmt(e.majority_identity)}) must fall{C['off']}",
         "",
         f"  {C['dim']}identity on groups the eraser never saw   {_fmt(e.identity_after_unseen_groups)}{C['off']}",
         f"  {C['dim']}identity with a curved boundary (RBF)     {_fmt(e.identity_after_nonlinear)}{C['off']}",
         "", f"  {C['b']}{e.verdict}{C['off']}"]
    for n in e.notes:
        o.append(_wrap(n, 68, "  ! ", "    "))
    return "\n".join(o)
