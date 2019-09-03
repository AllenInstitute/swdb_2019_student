import pandas as pd

def cell_types_ratio(brain_areas, good_units, size_cut_off):
    """sort cells using waveform duration into Fast Spiking and 
    Regular Spiking groups and then computes the ratio of FS cells to 
    RS cells in desired brain regions
    
    
    Parameters
    ==========
    
    brain_areas : list 
        contains brain areas of interest

    good_units : pandas.Dataframe 
        contains column of waveform duration information

    size_cut_off : value, float
        duration calculated for best split of waveforms(recommended between 0.4 and 0.5)

    Returns
    =======
    pandas.Dataframe

    """
    
    ratio_dict = {}
    
    for i,name in enumerate(brain_areas):
        area_units=good_units[good_units.structure_acronym==name]
        fs = area_units["waveform_duration"] < size_cut_off
        fs_df = area_units[fs].index
        fs_count=len(fs_df)
        rs_df = area_units[~(fs)].index
        rs_count=len(rs_df)
        ratio=fs_count/rs_count
        ratio_dict[name]=(ratio)
    return (ratio_dict)




