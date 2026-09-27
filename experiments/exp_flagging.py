"""Section 5 — flagging without a list of possible connections.

When the DSO cannot say which lines physically exist, the service does not
correct anything. It ranks the documented edges by how poorly the
measurements support them and flags the worst for inspection.
"""

from _common import SMALL, ERROR_RATE, load, save, short
from topology import data, errors, flag_suspicious_edges, evaluate


def main():
    rows = []
    for code in SMALL[1:] + ["1-LV-rural2--0-sw"]:
        tg, V, A, nodes = load(code)
        n_err = max(1, round(tg.number_of_edges() * ERROR_RATE))
        documented = errors.mixed(tg, n_err)

        truth = {frozenset(e) for e in tg.edges()}
        wrong = {frozenset(e) for e in documented.edges()} - truth
        n_doc = documented.number_of_edges()

        for meters in ["class_0.5", "class_0.5+angle"]:
            Vn, An = data.add_meter_noise(V, A, meters)
            flagged = flag_suspicious_edges(Vn, documented, k=len(wrong),
                                            nodes=nodes, angle=An)
            hits = sum(1 for e in flagged if frozenset(e) in wrong)
            rows.append({
                "feeder": short(code), "meters": meters,
                "wrong_edges": len(wrong), "documented_edges": n_doc,
                "random_baseline": f"{len(wrong)/n_doc:.0%}",
                "precision": f"{hits/max(len(flagged),1):.0%}",
                "lift": round((hits / max(len(flagged), 1)) / (len(wrong) / n_doc), 1),
            })

    save(rows, "flagging.csv")


if __name__ == "__main__":
    main()
