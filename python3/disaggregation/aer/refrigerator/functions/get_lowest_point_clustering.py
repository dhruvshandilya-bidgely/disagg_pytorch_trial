"""
Date Created - 13 Nov 2018
Author name - Pratap
Lowest point and cap value for water heater is calculated in this function
"""
import numpy as np
from sklearn.cluster import KMeans

from python3.disaggregation.aer.refrigerator.LapColumns import LapColumns


def sign_check(new_series, new_series2):
    """ check same sign points """
    j_1 = 0
    k_1 = 0
    for i_1 in range(np.shape(new_series)[0] - 1):
        if j_1 >= (np.shape(new_series)[0] - 2):
            break
        elif (new_series[j_1, 2] == new_series[j_1 + 1, 2]):
            new_series2[k_1, 0] = new_series[j_1, 1]
            j_1 = j_1 + 2
            k_1 = k_1 + 1
        else:
            new_series2[k_1, 0] = new_series[j_1, 1]
            j_1 = j_1 + 1
            k_1 = k_1 + 1

    return new_series2


def fix_negative_transition(new_series10, correct_array, lowest_point, incr_val, config):
    """ fix negative transitions """
    for i in range(np.shape(new_series10)[0]):
        if new_series10[i, 0] <= -config['Estimation']['AmpTrnsLimit']:
            correct_array = np.vstack((correct_array, new_series10[i, 1]))
            new_series10[i, 1] = lowest_point + incr_val

    return new_series10


def fix_positive_transition(new_series11, correct_array, lowest_point, incr_val, config):
    """ fix positive transitions """
    for i in range(np.shape(new_series11)[0] - 1):
        if new_series11[i, 0] >= config['Estimation']['AmpTrnsLimit']:
            correct_array = np.vstack((correct_array, new_series11[i + 1, 1]))
            new_series11[i + 1, 1] = lowest_point + incr_val

    return new_series11


def get_lowest_point_clustering(enr_low_pnt_data, lap_j, config, abc_test, lowest_point):
    """ calculates lowest point from clustering"""
    enr_low_lap_data = enr_low_pnt_data[(enr_low_pnt_data[:, 1] >= lap_j[:, LapColumns.LAP_START_EPOCH_IDX][0]) &
                                        (enr_low_pnt_data[:, 1] <= lap_j[:, LapColumns.LAP_END_EPOCH_IDX][0]), :]
    enr_low_points = enr_low_lap_data[enr_low_lap_data[:, 2] == 1, 4]
    enr_low_points = enr_low_points - lowest_point
    med_enr_low_pnts = np.median(enr_low_points)
    std_enr_low_pnts = np.std(enr_low_points, ddof=1)
    min_enr_lmt = med_enr_low_pnts - (config['Estimation']['StdThrEst'] * std_enr_low_pnts)
    max_enr_lmt = med_enr_low_pnts + (config['Estimation']['StdThrEst'] * std_enr_low_pnts)

    if min_enr_lmt < config['Estimation']['MinEnrThrLmt']:
        min_enr_lmt = 0
    fltrd_enr_pnt = enr_low_points[(enr_low_points >= min_enr_lmt) & (enr_low_points <= max_enr_lmt)]

    if np.shape(fltrd_enr_pnt)[0] > 0:
        srtd_fltrd_pnt = np.sort(fltrd_enr_pnt)
        top_srt_flt_pnt = srtd_fltrd_pnt[:3]
        new_lowst_pnt = np.mean(top_srt_flt_pnt)
    else:
        new_lowst_pnt = np.min(enr_low_points)

    new_lowst_pnt = new_lowst_pnt + lowest_point
    d_abctest = np.diff(abc_test)[:, np.newaxis]
    d_abctest = np.vstack((d_abctest, 0))
    abc_test = abc_test[:, np.newaxis]

    new_series = np.hstack((d_abctest, abc_test))
    new_series = np.hstack((new_series, np.sign(new_series[:, 0])[:, np.newaxis]))
    new_series = np.vstack((new_series, np.zeros((3, 3))))
    new_series2 = np.zeros((np.shape(new_series)[0], 1))

    # First check for sign
    new_series2 = sign_check(new_series, new_series2)

    new_series3 = new_series2[new_series2 > 0][:, np.newaxis]
    d_abctest2 = np.diff(new_series3[:, 0])
    d_abctest2 = np.vstack((d_abctest2[:, np.newaxis], 0))
    new_series4 = np.hstack((d_abctest2, new_series3))
    new_series4 = np.hstack((new_series4, np.sign(new_series4[:, 0][:, np.newaxis])))
    new_series4 = np.vstack((new_series4, np.zeros((3, 3))))
    new_series5 = np.zeros((np.shape(new_series4)[0], 1))

    # Second check for sign
    new_series5 = sign_check(new_series4, new_series5)

    new_series6 = new_series5[new_series5 > 0][:, np.newaxis]
    d_abctest3 = np.diff(new_series6[:, 0])
    d_abctest3 = np.vstack((d_abctest3[:, np.newaxis], 0))
    new_series7 = np.hstack((d_abctest3, new_series6))
    new_series7 = np.hstack((new_series7, np.sign(new_series7[:, 0][:, np.newaxis])))
    new_series7 = np.vstack((new_series7, np.zeros((3, 3))))
    new_series8 = np.zeros((np.shape(new_series7)[0], 1))

    # Third check for sign
    new_series8 = sign_check(new_series7, new_series8)

    new_series9 = new_series8[new_series8 > 0][:, np.newaxis]
    clus_diffval = np.abs(np.diff(new_series9[:, 0]))
    clus_diffval = clus_diffval[clus_diffval > config['Estimation']['MinClustLimit']]
    clus_diffval2 = clus_diffval[clus_diffval < percentile_1d(clus_diffval, 90)]

    if (np.shape(clus_diffval2)[0] > 2):
        k_means_fit = KMeans(n_clusters=2, init=np.array([[np.min(clus_diffval2)], [np.max(clus_diffval2)]]),
                             n_init=1, max_iter=100)
        kmeans_model = k_means_fit.fit(clus_diffval2.reshape(-1, 1))
        clus_seg = kmeans_model.predict(clus_diffval2.reshape(-1, 1))

        if kmeans_model.cluster_centers_[0][0] == kmeans_model.cluster_centers_[1][0]:
            clus_seg[0] = 1 - clus_seg[0]

        clus_seg_vals = np.hstack((clus_diffval2[:, np.newaxis], clus_seg[:, np.newaxis]))
        sortd_clsegval = clus_seg_vals[clus_seg_vals[:, 1].argsort()]

        first_clus_medn = percentile_1d(sortd_clsegval[sortd_clsegval[:, 1] == 0, 0], 90)
        second_clus_medn = percentile_1d(sortd_clsegval[sortd_clsegval[:, 1] == 1, 0], 90)
        first_clus_per = (np.shape(sortd_clsegval[sortd_clsegval[:, 1] == 0, 0])[0]) / (np.shape(sortd_clsegval)[0])
        second_clus_per = (np.shape(sortd_clsegval[sortd_clsegval[:, 1] == 1, 0])[0]) / (np.shape(sortd_clsegval)[0])

        if (first_clus_medn > second_clus_medn):
            max_clus_val = first_clus_medn
            min_clus_val = second_clus_medn
            max_clus_per = first_clus_per
            min_clus_per = second_clus_per
        else:
            max_clus_val = second_clus_medn
            min_clus_val = first_clus_medn
            max_clus_per = second_clus_per
            min_clus_per = first_clus_per

        if ((max_clus_val < config['Estimation']['AmpTrnsLimit']) & (max_clus_per > 0.25)):
            incr_val = max_clus_val
        elif ((min_clus_val > config['Estimation']['MinAmptorplcWHplsLimit']) & (min_clus_per > 0.25)):
            incr_val = min_clus_val
        else:
            incr_val = config['Estimation']['MinAmptorplcWHplsLimit']
    else:
        incr_val = config['Estimation']['MinAmptorplcWHplsLimit']

    d_abctest4 = np.diff(abc_test[:, 0])[:, np.newaxis]
    d_abctest4 = np.vstack((d_abctest4, 0))
    new_series10 = np.hstack((d_abctest4, abc_test))
    correct_array = np.zeros((1, 1))

    # Fix negative transition
    new_series10 = fix_negative_transition(new_series10, correct_array, lowest_point, incr_val, config)

    d_abctest5 = np.diff(new_series10[:, 1])[:, np.newaxis]
    d_abctest5 = np.vstack((d_abctest5, 0))
    new_series11 = np.hstack((d_abctest5, new_series10[:, 1][:, np.newaxis]))

    # Fix positive transition
    new_series11 = fix_positive_transition(new_series11, correct_array, lowest_point, incr_val, config)

    if (np.shape(correct_array)[0] > 1):
        correct_array = correct_array[1:]
        min_corr_val = np.min(correct_array)
        trncated_series = new_series11[:, 1]
        trncated_series[trncated_series > min_corr_val] = lowest_point + incr_val
    else:
        trncated_series = new_series11[:, 1]

    return new_lowst_pnt, incr_val, trncated_series


def percentile_1d(arr, ptile):
    """1d percentile based on MATLAB implemented for python"""
    arr_clean = np.sort(arr[~np.isnan(arr)])
    num_el = len(arr_clean)

    if num_el > 0:
        p_rank = 100.0 * (np.arange(num_el) + 0.5) / num_el
        return np.interp(ptile, p_rank, arr_clean)
    else:
        return np.nan

