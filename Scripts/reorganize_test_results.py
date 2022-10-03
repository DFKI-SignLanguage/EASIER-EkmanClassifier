import pandas as pd
from pandas_ods_reader import read_ods

# df = pd.read_csv("../aggregated_results-220906a.csv", sep=",")
# df = pd.read_csv("../aggregated_results-220926-only_MobileNet.csv", sep=",")
df = pd.read_csv("../../AffectDetectionComparison-aggregated_results-220926-only_MobileNet.csv", sep=",")
# df = read_ods("../aggregated_results-220906a.ods", 1)

# idx = df['Architecture'].str.split('-', expand=True).sort_values([1,0]).index
# df = df.reset_index(drop=True)
# df2 = df.reindex(idx).reset_index(drop=True)

df_cols = df['Architecture'].str.split('-', expand=True)
df_cols.columns = ["Architecture", "Test Set", "Img Proc Test"]
df = df.drop(["Test Set", "Img Proc Test"], axis=1)
# df = df.drop(["Architecture", "Architecture.1", "Test Set", "Img Proc Test"], axis=1)
# df = df.drop(["Architecture"], axis=1)
df = df_cols.merge(df, left_index=True, right_index=True)
# df2 = df.sort_values(['Img Proc Training', 'Test Set', 'Accuracy'], ascending=[True, True, True])
df2 = df.sort_values(['Test Set', 'Img Proc Training'], ascending=[False, False])
# df2 = df.sort_values(['Test Set'], ascending=[False])
temp_cols = df2.columns.tolist()
new_cols = [temp_cols[0]] + [temp_cols[3]] + temp_cols[1:3] + temp_cols[4:-1]
df2 = df2[new_cols]
# df2 = df.sort_values(['Test Set'], ascending=[True])
df2 = df2.reset_index(drop=True)

# df2.to_csv("../aggregated_results-220906a_byTestSet.csv")
# df2.to_csv("../aggregated_results-220926-only_MobileNet_byTestSet.csv")
df2.to_csv("../AffectDetectionComparison-aggregated_results-220926-only_MobileNet_byTestSet.csv")

# df3 = pd.read_csv("../aggregated_results-220906a_byTestSet.csv")
# df3 = pd.read_csv("../aggregated_results-220926-only_MobileNet_byTestSet.csv")
df3 = pd.read_csv("../../AffectDetectionComparison-aggregated_results-220926-only_MobileNet_byTestSet.csv")
print("###")
