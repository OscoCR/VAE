import argparse
import csv


def main():
    parser = argparse.ArgumentParser(description="Select the highest-error fraction of train images")
    parser.add_argument('--csv', required=True, type=str)
    parser.add_argument('--top_fraction', default=0.25, type=float)
    parser.add_argument('--out', required=True, type=str)
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = [(int(r['index']), float(r['mse'])) for r in csv.DictReader(f)]

    rows.sort(key=lambda r: r[1], reverse=True)
    n_hard = int(len(rows) * args.top_fraction)
    hard = rows[:n_hard]
    threshold = hard[-1][1]

    with open(args.out, "w") as f:
        for index, _ in sorted(hard):
            f.write(f"{index}\n")

    print(f"{len(rows)} images, top {args.top_fraction:.0%} = {n_hard} hard images")
    print(f"MSE threshold (lowest hard image): {threshold:.5f}")
    print(f"Saved indices to {args.out}")


if __name__ == "__main__":
    main()
