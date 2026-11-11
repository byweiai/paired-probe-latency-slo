# -*- coding: utf-8 -*-
"""Report duplicate macro definitions in paper/numbers.tex and optionally drop the earlier ones.

Both definition forms are recognised:  \\def\\numX{value}  and  \\newcommand{\\numX}{value}.
Read-only by default; pass --write to rewrite the file keeping the last definition of each name.

Exit status: 0 when every name has a single value (or the file was repaired with --write),
1 when a name is defined more than once with different values, 2 when the file is missing.
"""
import collections
import io
import re
import sys

P = "paper/numbers.tex"
DEF = re.compile(r"^\s*\\def\\(num[A-Za-z0-9]+)\{(.*)\}\s*$")
NEW = re.compile(r"^\s*\\newcommand\{\\(num[A-Za-z0-9]+)\}\{(.*)\}\s*$")


def parse(line):
    m = DEF.match(line) or NEW.match(line)
    return (m.group(1), m.group(2)) if m else (None, None)


def main(argv):
    write = "--write" in argv
    try:
        lines = io.open(P, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        print("MISSING", P)
        return 2

    seen = collections.OrderedDict()
    dup = collections.Counter()
    last_idx = {}
    for i, ln in enumerate(lines):
        name, val = parse(ln)
        if not name:
            continue
        if name in seen and seen[name] != val:
            dup[name] += 1
        seen[name] = val
        last_idx[name] = i

    print("macros: %d | conflicting redefinitions: %s" % (len(seen), dict(dup) if dup else "none"))

    if not dup:
        print("no repair needed")
        return 0

    if not write:
        print("run with --write to keep the last definition of each conflicting name")
        return 1

    out, emitted = [], set()
    for i, ln in enumerate(lines):
        name, _ = parse(ln)
        if name:
            if last_idx[name] != i or name in emitted:
                continue
            emitted.add(name)
        out.append(ln)
    io.open(P, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    print("rewritten: %d lines (was %d); removed %d definitions"
          % (len(out), len(lines), len(lines) - len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
