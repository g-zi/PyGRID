from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Point:
    x: float
    y: float
    identifier: str


def read_fault_file(filename):

    faults = defaultdict(list)

    with open(filename) as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("*"):
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            x = float(parts[0])
            y = float(parts[1])
            fault_id = parts[2]

            faults[fault_id].append(
                Point(
                    x,
                    y,
                    fault_id
                )
            )

    return faults


def pair_faults(top_file, bottom_file):

    top = read_fault_file(top_file)
    bottom = read_fault_file(bottom_file)

    print("TOP FAULTS")
    for f in top:
        print(f, len(top[f]))

    print()
    print("BOTTOM FAULTS")
    for f in bottom:
        print(f, len(bottom[f]))


    pairs = []

    for top_id in top:

        if top_id.startswith("T_"):

            base = top_id[2:]
            bottom_id = "B_" + base

            if bottom_id in bottom:

                pairs.append(
                    (
                        top_id,
                        bottom_id
                    )
                )


    print()
    print("FAULT PAIRS")

    for t,b in pairs:
        print(t,"<->",b)

    return top,bottom,pairs



if __name__ == "__main__":

    pair_faults(
        "top_fault.txt",
        "bottom_fault.txt"
    )