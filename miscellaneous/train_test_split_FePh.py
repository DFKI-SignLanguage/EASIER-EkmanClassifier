import os
import pandas as pd

data_path = "C:\\Users\\chira\\OneDrive - uni-saarland.de\\Codebases\\DFKI_HiWi\\ACG\\Easier\\Datasets\\FePh"
x_dir_path = os.path.join(data_path, 'FePh_images')
y_csv_path = os.path.join(data_path, 'FePh_labels.csv')
test_set_fraction = 0.2  # Fraction of FePh_labels.csv to use for test set

y_df = pd.read_csv(y_csv_path)
# Removing all data points with 'Face_not_visible' i.e no labels
y_df.dropna(inplace=True)
# Extracting multiple labels
y_df['Facial_label'] = y_df['Facial_label'].apply(lambda x: [int(i) for i in x])
y_df['num_labels'] = y_df['Facial_label'].apply(lambda x: len(x))
y_df['Facial_label'] = y_df['Facial_label'].apply(lambda x: x[0])

# Removing all data points with more than one labels ==> Ambiguous
y_df = y_df[y_df["num_labels"] == 1]
n_r, n_c = y_df.shape
n_test_set = round(test_set_fraction * n_r)


def get_class_weight_col(df):
    counts = df["Facial_label"].value_counts()
    prop_normalized = counts.apply(lambda x: x / n_r)
    prop_normalized = dict(prop_normalized)
    df["class_weight"] = df["Facial_label"].apply(lambda label: prop_normalized[label])
    return df, prop_normalized, counts


y_df, prop_full, counts_full = get_class_weight_col(y_df)

test_df = y_df.sample(n=n_test_set, weights='class_weight', random_state=1)
test_df = test_df.reset_index()
test_df = test_df.rename(columns={'index': 'index_before_sampling', "class_weight": "class_weight_before_sampling"})

train_df = y_df.drop(test_df["index_before_sampling"])
train_df = train_df.reset_index()
train_df = train_df.rename(columns={'index': 'index_before_sampling', "class_weight": "class_weight_before_sampling"})

test_df, prop_test, counts_test = get_class_weight_col(test_df)
train_df, prop_train, counts_train = get_class_weight_col(train_df)

for col in train_df.columns:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

train_df.to_csv(os.path.join(data_path, 'FePh_train.csv'))
test_df.to_csv(os.path.join(data_path, 'FePh_test.csv'))

print(prop_full)
print(prop_train)
print(prop_test)

print(counts_full)
print(counts_train)
print(counts_test)
