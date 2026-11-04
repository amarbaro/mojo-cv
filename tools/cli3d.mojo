# mojo-cv3d OP OUT IN... [k=v]: the 3D/CAD command line (a package cannot hold a main(), so the entry lives here).
# Build: mojo build -I src tools/cli3d.mojo -o .bin/mojo-cv3d
from std.sys import argv
from mojocv.cli3d import dispatch3d


def main() raises:
    var a = List[String]()
    var av = argv()
    for i in range(1, len(av)):
        a.append(String(av[i]))
    if not dispatch3d(a):
        print("FAIL cli3d: unknown op or too few arguments")
        raise Error("cli3d")
