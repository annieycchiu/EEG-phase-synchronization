import argparse

import pandas as pd

from src.connectivity import verify_inputs, create_distinct_values_dic, \
    generate_connectivity_values, generate_connectivity_tables, save_df_to_excel

def run(
    data_filepath, 
    pivot_filter, 
    pivot_subcol, 
    pivot_idx='Wavelet',
    save_excel=True,
    output_filepath=None
):
    df = pd.read_csv(data_filepath)
    verify_inputs(df, pivot_filter, pivot_idx, pivot_subcol)
    distinct_vals_dic = create_distinct_values_dic(df, pivot_filter, pivot_idx, pivot_subcol)
    conn_vals_dic = generate_connectivity_values(df, distinct_vals_dic)
    conn_pivs = generate_connectivity_tables(conn_vals_dic, distinct_vals_dic)
    
    if save_excel:
        save_df_to_excel(conn_pivs, distinct_vals_dic, output_filepath)

    return conn_pivs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str, help="Path to the EEG data.")
    parser.add_argument("--pivot_filter", type=str, help="The name of the desired column used as the filter for the output pivot table.")
    parser.add_argument("--pivot_subcol", type=str, help="The name of the desired column used as the sub-column for the output pivot table.")
    parser.add_argument("--pivot_idx", type=str, default="Wavelet", help="The name of the desired column used as the index for the output pivot table.")
    parser.add_argument("--save_excel", type=bool, default=True, help="Save output pivot table to Excel file.")
    parser.add_argument("--output_filepath", type=str, default=None, help="Path of output pivot tables.")

    opt = parser.parse_args()

    return opt

def main(opt):
    run(**vars(opt))
    
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)