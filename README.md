# EEG-phase-synchronization

## Usage

To perform phase sychronization analysis, you can either run the python command or call the python function.

#### - run python command

```bash
python phase_synchronization.py 
    --data_filepath <filepath of EEG data>
    --pivot_filter <chosen column name as the filter of output pivot table>
    --pivot_subcol <chosen column name as the sub-column of output pivot table> 
    --pivot_subcol <chosen column name as the index of output pivot table>
    --save_excel <save output pivot table to Excel file or not>
    --output_filepath <desired output filepath>
```

#### - call python function

```python
pivot_tbls = phase_synchronization.run(
    data_filepath='data/test_connectivity_data.csv', 
    pivot_filter='Label', 
    pivot_idx='Wavelet', 
    pivot_subcol='Sleep',
    save_excel=True,
    output_filepath='connectivity_result_filter_Label.xlsx'
)
```
