"""Read chosen keys from a .env file without a shell. Used by snap.sh.

Usage: readenv.py ENV_FILE KEY...
Prints "KEY=value" NUL-terminated for each key found (last one wins).
Understands blank lines, # comments, an "export " prefix and one pair of
surrounding single or double quotes. Nothing is expanded or evaluated.
"""
import sys


def parse_env(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[7:].lstrip()
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


if __name__ == "__main__":
    env = parse_env(open(sys.argv[1], encoding="utf-8").read())
    for k in sys.argv[2:]:
        if k in env:
            sys.stdout.write(f"{k}={env[k]}\0")
