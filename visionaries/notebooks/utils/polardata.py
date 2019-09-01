"""
Useful functions for wrangling with polar data
"""

def cell_df_to_median_polar(cell_df):
    """
    Input:
    - cell_df (DataFrame) - each row is one cell. columns = [angle, magnitude].
    Output:
    - list<(angle in degree, median intensity for this angle)>
    """
    grouped_df = cell_df.groupby('angle').median()
    result = []
    for index, row in grouped_df.iterrows():
        result.append((index, row['magnitude']))
    return result