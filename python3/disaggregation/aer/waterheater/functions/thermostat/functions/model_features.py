"""
Author - Nikhil
Date - 10/10/2018
This file contains the list of features used for non-timed water heater detection
"""

# Train variable is the one currently in use for detection algorithm

train = ['std_percent', 'all_wh_laps', 'wtr_wh_laps', 'itr_wh_laps', 'smr_wh_laps',
         'wtr_peaks_count', 'smr_peaks_count', 'all_laps_count', 'wtr_laps_count', 'itr_laps_count',
         'smr_laps_count', 'max_laps_count', 'all_peaks_count_night', 'wtr_peaks_count_night',
         'itr_peaks_count_night', 'smr_peaks_count_night', 'wtr_consistency', 'itr_consistency', 'smr_consistency',
         'all_valid_lap_days', 'smr_valid_lap_days', 'wtr_peak_factor', 'itr_peak_factor',
         'smr_peak_factor', 'all_peak_factor_std']

train_old = ['std_percent', 'all_wh_laps', 'smr_peaks_count', 'max_peaks_count', 'wtr_laps_count', 'smr_laps_count',
             'max_laps_count',  'all_peaks_count_night', 'wtr_peaks_count_night', 'itr_peaks_count_night',
             'smr_peaks_count_night', 'all_consistency', 'wtr_consistency', 'itr_consistency', 'smr_consistency',
             'max_valid_lap_days', 'itr_valid_lap_days', 'smr_valid_lap_days', 'wtr_peak_factor', 'itr_peak_factor',
             'smr_peak_factor', 'all_peak_factor_std']

train_nve = ['std_percent', 'all_wh_laps', 'max_peaks_count', 'wtr_laps_count', 'smr_laps_count', 'max_laps_count',
             'all_peaks_count_night', 'wtr_peaks_count_night', 'itr_peaks_count_night', 'smr_peaks_count_night',
             'all_consistency', 'wtr_consistency', 'itr_consistency', 'smr_consistency', 'max_valid_lap_days',
             'smr_valid_lap_days', 'wtr_wh_days', 'smr_wh_days', 'wtr_peak_factor', 'itr_peak_factor',
             'smr_peak_factor', 'all_peak_factor_std']

nve = ['std_percent', 'all_wh_laps', 'wtr_peaks_count', 'max_peaks_count', 'all_laps_count', 'wtr_laps_count',
       'smr_laps_count', 'max_laps_count', 'all_peaks_count_night', 'wtr_peaks_count_night', 'itr_peaks_count_night',
       'smr_peaks_count_night', 'all_consistency', 'wtr_consistency', 'itr_consistency', 'smr_consistency',
       'max_valid_lap_days', 'smr_valid_lap_days', 'smr_wh_days', 'wtr_peak_factor', 'itr_peak_factor',
       'smr_peak_factor', 'all_peak_factor_std']
