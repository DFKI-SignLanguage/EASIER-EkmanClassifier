import argparse
import pandas
import os

args = argparse.ArgumentParser(description='Analyse a dataset and sevs some statistics.')
args.add_argument('-l', '--labels', default=None, type=str, required=True,
                  help="path to the directory containing the dataset (labels.csv, images/, ...")
args.add_argument('-o', '--out_stats', default=None, type=str, required=True,
                  help="Output text file with statistics")
#args.add_argument('-t', '--ground_truths', default=None, type=str,
#                  help="path to csv file with emotion ground truths")

args = args.parse_args()

outstats_filename = args.out_stats

labels_filename = args.labels


df = pandas.read_csv(labels_filename)

with open(outstats_filename, "w") as outstats_file:

    outstats_file.write("Number of samples: {}\n".format(len(df)))

    outstats_file.write("Labels distribution:\n")
    outstats_file.write(str(df.ClassName.value_counts()))
    outstats_file.write("\n")





