import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def verify_inputs(df, pivot_filter, pivot_idx, pivot_subcol):
    
    # check input arguement - df
    if not isinstance(df, pd.core.frame.DataFrame):
        raise TypeError(f'Expected "pd.core.frame.DataFrame" for df, but got {type(df).__name__} instead.')
        
    columns = set(df.columns)
    
    
    # check required columns
    required_columns = ['Channel', 'Feature', 'Value']
    for col in required_columns:
        if col not in columns:
            raise Exception(f'{col} is not found in dataframe column names.')
            
    channels = sorted(list(df['Channel'].unique()))
    features = sorted(list(df['Feature'].unique()))
    
    if len(channels) != len(features):
        raise Exception(f'Mismatch of channel names in Channel and Feature columns')
        
    for C, F in zip(channels, features):
        if C != F.split('_')[1]:
            raise Exception(f'Mismatch of channel names in Channel and Feature columns')
            
    
    # check input arguement - pivot_filter
    if (not isinstance(pivot_filter, str)) and (not isinstance(pivot_filter, list)):
        raise TypeError(f'Expected "str" or "list" for filters, but got {type(pivot_filter).__name__} instead.')
        
    if isinstance(pivot_filter, str):
        if pivot_filter not in columns:
            raise Exception(f'Filters is not found in dataframe column names.')
            
    if isinstance(pivot_filter, list):
        for fil in pivot_filter:
            if fil not in columns:
                raise Exception(f'Filters is not found in dataframe column names.')
    
    
    # check input arguement - pivot_idx
    if not isinstance(pivot_idx, str):
        raise TypeError(f'Expected "str" for idx, but got {type(pivot_idx).__name__} instead.')
        
    if pivot_idx not in columns:
        raise Exception(f'Idx is not found in dataframe column names.')
    
    
    # check input arguement - pivot_subcol
    if not isinstance(pivot_subcol, str):
        raise TypeError(f'Expected "str" for subcol, but got {type(pivot_subcol).__name__} instead.')
        
    if pivot_subcol not in columns:
        raise Exception(f'Subcol is not found in dataframe column names.')
        
    if df[pivot_subcol].nunique() != 2:
        raise Exception(f'The number of unique values in subcol needs to be 2.')
        

def create_distinct_values_dic(df, pivot_filter, pivot_idx, pivot_subcol):
    distinct_vals_dic = {}
    distinct_vals_dic['pivot_filter'] = [pivot_filter, sorted(df[pivot_filter].unique())]
    distinct_vals_dic['pivot_idx'] = [pivot_idx, sorted(df[pivot_idx].unique())]
    distinct_vals_dic['pivot_subcol'] = [pivot_subcol, sorted(df[pivot_subcol].unique())]
    
    return distinct_vals_dic


def create_corr_table(df, F, I, S, distinct_vals_dic):
    
    filter_col_name = distinct_vals_dic['pivot_filter'][0]
    idx_col_name = distinct_vals_dic['pivot_idx'][0]
    subcol_col_name = distinct_vals_dic['pivot_subcol'][0]
    
    # fliter out sub-dataset
    sub_df = df[(df[filter_col_name] == F) & 
                (df[idx_col_name] == I) & 
                (df[subcol_col_name] == S)]
    
    # create correlation table
    df_corr = pd.DataFrame(sub_df.groupby(['Feature', 'Channel'])['Value'].mean())\
                .reset_index(level=[0, 1])\
                .pivot(index='Feature', columns='Channel', values=['Value'])
    
    # replace feature names by channel names
    df_corr.index = df_corr.columns.get_level_values(1)
    df_corr.index.names = ['Feature']

    return df_corr


def generate_connectivity_dict(df_corr):
    channels = list(df_corr.index)
    left = [x for x in channels if x[-1].isdigit() and int(x[-1]) % 2 == 1]
    right = [x for x in channels if x[-1].isdigit() and int(x[-1]) % 2 == 0]
    
    AC = list(combinations(channels, 2))
    NC = [
        ('C3', 'F3'), ('C3', 'F7'), ('C3', 'Fz'),  ('C3', 'P3'), ('C3', 'P7'), ('C3', 'Pz'),('C3', 'T7'),
        ('C4', 'F4'), ('C4', 'F8'), ('C4', 'Fz'), ('C4', 'P4'), ('C4', 'P8'), ('C4', 'Pz'), ('C4', 'T8'),
        ('F3', 'F7'), ('F3', 'Fp1'), ('F3', 'Fp2'), ('F3', 'Fz'), ('F3', 'T7'),
        ('F4', 'F8'), ('F4', 'Fp1'), ('F4', 'Fp2'), ('F4', 'Fz'), ('F4', 'T8'),
        ('F7', 'Fp1'), ('F7', 'T7'),
        ('F8', 'Fp2'), ('F8', 'T8'),
        ('Fp1', 'Fp2'), ('Fp1', 'Fz'),
        ('O1', 'O2'), ('O1', 'P3'), ('O1', 'P7'), ('O1', 'Pz'),
        ('O2', 'P4'), ('O2', 'P8'), ('O2', 'Pz'),
        ('P3', 'P7'), ('P3', 'Pz'), ('P3', 'T7'),
        ('P4', 'P8'), ('P4', 'Pz'), ('P4', 'T8'),
        ('P7', 'T7'),
        ('P8', 'T8'),
    ]
    FC = sorted(list(set(AC) - set(NC)))
    LC = list(combinations(left, 2))
    RC = list(combinations(right, 2))
    IC = [(L, R) for L in left for R in right]
    
    values = {
        'AC': [df_corr.loc[i, ('Value', c)] for (i, c) in AC], 
        'NC': [df_corr.loc[i, ('Value', c)] for (i, c) in NC], 
        'FC': [df_corr.loc[i, ('Value', c)] for (i, c) in FC], 
        'LC': [df_corr.loc[i, ('Value', c)] for (i, c) in LC], 
        'RC': [df_corr.loc[i, ('Value', c)] for (i, c) in RC], 
        'IC': [df_corr.loc[i, ('Value', c)] for (i, c) in IC]
    } 
   
    return values


def unnest_dict(nested_dict, parent_key='', sep='_'):
    unnested_dict = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            unnested_dict.update(unnest_dict(value, new_key, sep))
        else:
            unnested_dict[new_key] = value
            
    return unnested_dict


def generate_connectivity_values(df, distinct_vals_dic):
    
    filter_vals = distinct_vals_dic['pivot_filter'][1]
    idx_vals = distinct_vals_dic['pivot_idx'][1]
    subcol_vals = distinct_vals_dic['pivot_subcol'][1]
    
    conn_vals_dic = {}
    for F in filter_vals:
        for I in idx_vals:
            for S  in subcol_vals:
                sub_corr = create_corr_table(df, F, I, S, distinct_vals_dic)
                values = generate_connectivity_dict(sub_corr)
                key = f'{F}_{I}_{S}'
                conn_vals_dic[key] = values

    conn_vals_dic = unnest_dict(conn_vals_dic) 
    
    return conn_vals_dic


def generate_connectivity_tables(conn_vals_dic, distinct_vals_dic):
    global col_values_dict
    
    filter_vals = distinct_vals_dic['pivot_filter'][1]
    idx_vals = distinct_vals_dic['pivot_idx'][1]
    subcol_vals = distinct_vals_dic['pivot_subcol'][1]
    
    # Initialize a DataFrame filled with zeros
    connectivities = ['AC', 'NC', 'FC', 'LC', 'RC', 'IC']
    idx_names = idx_vals
    subcol_names = subcol_vals + ['p-value']
    col_names = pd.MultiIndex.from_tuples([(C, S) for C in connectivities for S in subcol_names])

    conn_pivs = []
    for F in filter_vals:
        conn_piv = pd.DataFrame(0, index=idx_names, columns=col_names)
        for C in connectivities:
            for I in idx_vals:
                t_test_vals = []
                for S in subcol_vals:
                    key = f'{F}_{I}_{S}_{C}'
                    mean = np.mean(conn_vals_dic[key])
                    conn_piv.loc[I, (C, S)] = mean
                    t_test_vals.append(conn_vals_dic[key])
                t_statistic, p_value = stats.ttest_ind(t_test_vals[0], t_test_vals[1])
                conn_piv.loc[I, (C, 'p-value')] = p_value

        conn_pivs.append(conn_piv)
    
    return conn_pivs


def save_df_to_excel(conn_pivs, distinct_vals_dic, output_filepath=None):
    filter_col_name = distinct_vals_dic['pivot_filter'][0]
    filter_vals = distinct_vals_dic['pivot_filter'][1]

    if output_filepath is None:
        if not os.path.exists('output'):
            os.makedirs('output')
        output_filepath = f'output/connectivity_result_filter_{filter_col_name}.xlsx'
    
    excel_writer = pd.ExcelWriter(output_filepath, engine='xlsxwriter')
    
    for i in range(len(filter_vals)):
        # Save each DataFrame to a separate sheet in the Excel file
        conn_pivs[i].to_excel(excel_writer, sheet_name=f'{filter_vals[i]}')

    excel_writer.save()
