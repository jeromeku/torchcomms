#!/usr/bin/env python3
import os, re, sys

MD_DIR = os.path.dirname(__file__)

code_link = re.compile(r"`([\w./-]+):(\d+)`")
code_link_range = re.compile(r"`([\w./-]+):(\d+)-(\d+)`")

def convert_links(s: str) -> str:
    # First handle ranges like `path:12-34` -> [path:12](path#L12)–[34](path#L34)
    def normalize_target(p: str) -> str:
        # Make paths clickable from codex/*.md by referencing repo-root paths
        if p.startswith(('codex/', './', '../', '/')):
            return p
        return f"../{p}"
    def repl_range(m):
        p, a, b = m.group(1), m.group(2), m.group(3)
        t = normalize_target(p)
        return f"[{p}:{a}]({t}#L{a})–[{b}]({t}#L{b})"
    s = code_link_range.sub(repl_range, s)
    # Then single-line references `path:12` -> [path:12](path#L12)
    def repl_single(m):
        p, a = m.group(1), m.group(2)
        t = normalize_target(p)
        return f"[{p}:{a}]({t}#L{a})"
    s = code_link.sub(repl_single, s)
    # Also normalize existing markdown link targets like ](comms/...#L42)
    md_target = re.compile(r"\]\(((?!\.{1,2}/|/|codex/)[\w./-]+)#L(\d+)\)")
    def repl_target(m):
        p, a = m.group(1), m.group(2)
        t = f"../{p}"
        return f"]({t}#L{a})"
    s = md_target.sub(repl_target, s)
    return s

def main():
    for name in os.listdir(MD_DIR):
        if not name.endswith('.md'): continue
        path = os.path.join(MD_DIR, name)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        new = convert_links(content)
        if new != content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new)
            print(f"Updated links in {name}")

if __name__ == '__main__':
    main()
