import sys, io, re, pathlib

path = pathlib.Path("asgi.py")
src = path.read_text(encoding="utf-8", errors="replace")

# Normalize tabs to 4 spaces
src = src.expandtabs(4)
lines = src.splitlines(True)

def indent_level(s: str) -> int:
    return len(s) - len(s.lstrip(" "))

def is_code_line(s: str) -> bool:
    t = s.strip()
    return t != "" and not t.startswith("#")

insertions = 0
i = 0
while i < len(lines):
    line = lines[i]
    # Detect block starters
    if line.rstrip().endswith(":"):
        base_indent = indent_level(line)
        # Find next non-blank line
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        need_pass = False
        if j >= len(lines):
            need_pass = True
        else:
            next_line = lines[j]
            # If next line is not more indented, the block is empty/invalid
            if indent_level(next_line) <= base_indent:
                need_pass = True
        if need_pass:
            # Insert 'pass' one level deeper
            lines.insert(i + 1, " " * (base_indent + 4) + "pass\n")
            insertions += 1
            i += 1  # skip over the inserted line
    i += 1

fixed = "".join(lines)
if fixed != src:
    path.write_text(fixed, encoding="utf-8")
print(f"Inserted {insertions} 'pass' lines.")
