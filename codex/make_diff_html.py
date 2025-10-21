#!/usr/bin/env python3
# Simple static HTML generator for side-by-side unified diffs.
# Input: codex/nccl_vs_ncclx_v2_27.diff
# Output: codex/diff_viewer.html

import html
import os
import re
import sys

DIFF_PATH = os.path.join(os.path.dirname(__file__), 'nccl_vs_ncclx_v2_27.diff')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'diff_viewer.html')

file_header_re = re.compile(r'^\-\-\-\s+(.*)')
file_header_new_re = re.compile(r'^\+\+\+\s+(.*)')
hunk_re = re.compile(r'^@@\s+\-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@')

class HunkRow:
    __slots__ = ('lno','rno','ltext','rtext','ltag','rtag')
    def __init__(self, lno=None, rno=None, ltext='', rtext='', ltag='ctx', rtag='ctx'):
        self.lno, self.rno = lno, rno
        self.ltext, self.rtext = ltext, rtext
        self.ltag, self.rag = ltag, rtag

def sanitize(s: str) -> str:
    return html.escape(s, quote=False)

def render_hunk(rows, left_start, right_start):
    out = []
    out.append('<table class="hunk">')
    out.append('<thead><tr><th class="ln">L</th><th class="code">upstream (thirdparty/nccl)</th><th class="ln">R</th><th class="code">ncclx (comms/ncclx/v2_27)</th></tr></thead>')
    out.append('<tbody>')
    for lno, rno, ltxt, rtxt, cls in rows:
        # cls: one of ctx, add, del, pair
        ldisp = '' if lno is None else str(lno)
        rdisp = '' if rno is None else str(rno)
        lcls = 'ctx' if cls=='ctx' else ('del' if cls in ('del','pair_del') else 'ctx')
        rcls = 'ctx' if cls=='ctx' else ('add' if cls in ('add','pair_add') else 'ctx')
        out.append('<tr>')
        out.append(f'<td class="ln {lcls}">{ldisp}</td>')
        out.append(f'<td class="code {lcls}"><pre>{sanitize(ltxt)}</pre></td>')
        out.append(f'<td class="ln {rcls}">{rdisp}</td>')
        out.append(f'<td class="code {rcls}"><pre>{sanitize(rtxt)}</pre></td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)

def main():
    if not os.path.exists(DIFF_PATH):
        print(f"Diff not found: {DIFF_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(DIFF_PATH, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    files = []  # list of (left_path, right_path, hunks)
    left = right = None
    i = 0
    while i < len(lines):
        line = lines[i]
        # skip noise lines like 'diff -ruN ...'
        if line.startswith('diff '):
            i += 1
            continue
        m_old = file_header_re.match(line)
        if m_old:
            left = m_old.group(1).strip()
            # read next lines until +++
            i += 1
            if i < len(lines) and file_header_new_re.match(lines[i]):
                right = file_header_new_re.match(lines[i]).group(1).strip()
                i += 1
            else:
                right = ''
            hunks = []
            # collect hunks until next file or EOF
            while i < len(lines):
                if lines[i].startswith('diff ') or lines[i].startswith('--- '):
                    break
                mh = hunk_re.match(lines[i])
                if not mh:
                    i += 1
                    continue
                # parse hunk header
                old_start = int(mh.group(1))
                new_start = int(mh.group(3))
                i += 1
                lno = old_start
                rno = new_start
                # accumulate pending deletions to pair with additions
                pending_del = []  # list of (lno, text)
                rows = []
                while i < len(lines):
                    if lines[i].startswith('diff ') or lines[i].startswith('--- ') or hunk_re.match(lines[i]):
                        # flush pending deletions
                        for dlno, dtext in pending_del:
                            rows.append((dlno, None, dtext, '', 'del'))
                        pending_del.clear()
                        break
                    ch = lines[i][:1]
                    content = lines[i][1:].rstrip('\n')
                    if ch == ' ':
                        # flush pending deletions first
                        for dlno, dtext in pending_del:
                            rows.append((dlno, None, dtext, '', 'del'))
                        pending_del.clear()
                        rows.append((lno, rno, content, content, 'ctx'))
                        lno += 1
                        rno += 1
                    elif ch == '-':
                        pending_del.append((lno, content))
                        lno += 1
                    elif ch == '+':
                        if pending_del:
                            dlno, dtext = pending_del.pop(0)
                            rows.append((dlno, rno, dtext, content, 'ctx'))
                        else:
                            rows.append((None, rno, '', content, 'add'))
                        rno += 1
                    else:
                        # metadata line; ignore
                        pass
                    i += 1
                hunks.append((old_start, new_start, rows))
            files.append((left, right, hunks))
            continue
        i += 1

    # Write HTML
    with open(OUT_PATH, 'w', encoding='utf-8') as out:
        out.write("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>NCCL vs NCCLX — Side-by-side Diff</title>
  <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 0; padding: 0; }
    header { position: sticky; top: 0; background: #111; color: #eee; padding: 8px 16px; z-index: 10; }
    .container { padding: 12px 16px; }
    details { border: 1px solid #ddd; border-radius: 6px; margin: 12px 0; background: #fafafa; }
    summary { cursor: pointer; padding: 8px 12px; font-weight: 600; }
    summary .file { color: #333; }
    table.hunk { width: 100%; border-collapse: collapse; table-layout: fixed; margin: 0 0 8px 0; }
    table.hunk thead th { position: sticky; top: 40px; background: #f0f0f0; z-index: 5; }
    td, th { border-bottom: 1px solid #eee; vertical-align: top; }
    th.ln, td.ln { width: 4em; color: #666; text-align: right; padding: 2px 6px; }
    td.code { white-space: pre-wrap; word-break: break-word; padding: 2px 6px; }
    td.code pre { margin: 0; }
    .ctx { background: #fff; }
    .add { background: #eaffea; }
    .del { background: #ffecec; }
    .file-header { font-weight: 600; color: #444; }
    .paths { color: #666; font-size: 12px; }
    .stats { color: #888; font-size: 12px; margin-left: 8px; }
  </style>
</head>
<body>
  <header>
    <div>NCCL (thirdparty/nccl) vs NCCLX (comms/ncclx/v2_27) — Side-by-side Diff</div>
  </header>
  <div class="container">
""")
        for (lpath, rpath, hunks) in files:
            # compute simple stats
            adds=deletes=contexts=0
            for _, _, rows in hunks:
                for lno, rno, ltxt, rtxt, cls in rows:
                    if cls=='add': adds+=1
                    elif cls=='del': deletes+=1
                    else: contexts+=1
            title = f"{html.escape(lpath)} ⟷ {html.escape(rpath)}"
            out.write('<details>\n')
            out.write(f'<summary><span class="file">{title}</span><span class="stats">+{adds} −{deletes}</span></summary>\n')
            out.write('<div class="paths">')
            out.write(f'Upstream: {html.escape(lpath)}<br/>NCCLX: {html.escape(rpath)}')
            out.write('</div>\n')
            for old_start, new_start, rows in hunks:
                out.write(render_hunk(rows, old_start, new_start))
            out.write('</details>\n')
        out.write("""
  </div>
</body>
</html>
""")

    print(f"Wrote {OUT_PATH}")

if __name__ == '__main__':
    main()

