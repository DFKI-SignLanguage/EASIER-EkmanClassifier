from pathlib import Path

import pandas

RESULTS_DIR = Path("results")

out_df = pandas.DataFrame()

for e in sorted(RESULTS_DIR.iterdir()):

    if not e.is_dir():
        print("Skipping non-directory", e)
        continue

    results_file = e / "test_results.csv"

    if not results_file.exists():
        print("No test_results.csv file in directory {}. Skipping...".format(e))
        continue

    res_df = pandas.read_csv(results_file)
    assert len(res_df) == 1
    res_df["Architecture"] = e.stem

    out_df = out_df.append(res_df)

    # print(results_file)

out_df.to_csv("out.csv", header=True, index=False)

