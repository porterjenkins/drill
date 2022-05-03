import argparse
import os
import pandas as pd


def get_labels(xls_path, output_dir):
    labs = pd.read_excel(xls_path)
    # filter missing labels "type"
    labs = labs[~pd.isnull(labs["Type"])]
    # filter: Pumping Time (Should all be >0)
    labs = labs[labs["Pumping Time (Should all be >0)"] > 0]
    # filter: Time Between (Should be >0)
    labs = labs[labs["Pumping Time (Should all be >0)"] > 0]

    labs.to_csv(os.path.join(output_dir, "label-data.csv"), index=False)





if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--label-xls', type=str, help='Path to label excel file')
    parser.add_argument('--output-dir', type=str, help='output directory')

    args = parser.parse_args()

    get_labels(args.label_xls, args.output_dir)



