from pathlib import Path

import pandas

RESULTS_DIR = Path("results")

print("Aggregating results for dir '{}'".format(RESULTS_DIR))


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

    # Rework columns
    columns_to_drop = ["Timestamp", "Training set split", "Hyper-params", "Epochs", "Validation set split", "Validation Accuracy", "Validation Balanced Accuracy", "Test set split" ]
    res_df = res_df.drop(labels=columns_to_drop, axis='columns')

    result_file_splitted = e.stem.split('-')
    img_preproc = result_file_splitted[-1]
    testset = result_file_splitted[-2]
    architecture = result_file_splitted[0]

    res_df["Architecture"] = architecture

    # Insert the columns just after the Architecture
    res_df.insert(loc=1, column="Img Proc Test", value=img_preproc)
    res_df.insert(1, "Test Set", testset)
    res_df.insert(1, "Img Proc Training", "???")
    res_df.insert(1, "Training Set", "???")
    # Insert the directory at the beginning of everything
    res_df.insert(0, "ModelDir", e.stem)

    # Append to results
    out_df = out_df.append(res_df)


aggregated_table = "aggregated_results.csv"

print("Writing '{}'".format(aggregated_table))
out_df.to_csv(aggregated_table, header=True, index=False)

print("All done.")
