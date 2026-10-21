"""Section 3 — the use case that was evaluated and rejected.

Blind reconstruction from measurements alone. Works on clean data, but is
measurement-limited under realistic meter noise, which is why the service
uses the documented topology as a starting point instead.
"""

from _common import SMALL, load, save, short
from topology import data, reconstruct, evaluate

METERS = ["clean", "good+angle", "class_0.5+angle", "class_0.5"]


def main():
    rows = []
    for code in SMALL:
        tg, V, A, nodes = load(code)
        row = {"feeder": short(code), "buses": len(nodes)}
        for meters in METERS:
            Vn, An = data.add_meter_noise(V, A, meters)
            G = reconstruct(Vn, nodes, angle=An)
            row[meters] = round(evaluate(G, tg)["reconstruction_accuracy"], 3)
        rows.append(row)

    save(rows, "blind.csv")
    print("\nBlind reconstruction degrades sharply with meter noise;")
    print("see section 3 of the report for why this use case was rejected.")


if __name__ == "__main__":
    main()
