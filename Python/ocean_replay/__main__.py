"""python -m ocean_replay validate FILE   |   python -m ocean_replay extract ARCHIVE --spec S.json --out OUT.h5"""
import sys

from .extract import main as _extract_main
from .schema import validate


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "validate":
        issues = validate(sys.argv[2])
        print("valid" if not issues else "\n".join(issues))
        sys.exit(1 if issues else 0)
    if len(sys.argv) >= 2 and sys.argv[1] == "extract":
        _extract_main(sys.argv[2:])
        return
    print(__doc__)


if __name__ == "__main__":
    main()
