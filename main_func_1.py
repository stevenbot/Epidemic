#！/usr/bin/env python
# -*- coding:utf-8 -*-
# __author__ = 'steven'


import os
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import sys
import csv
import copy
from SALib.analyze import morris
from SALib.sample.morris import sample
from SALib.test_functions import Sobol_G
from SALib.util import read_param_file
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, \
    sample_histograms
import matplotlib.pyplot as plt
from scikits import bootstrap
import scipy
import random
import datetime
import matplotlib.dates as mdate
from matplotlib.pyplot import MultipleLocator
import scipy.stats as st
import statsmodels.api as sm
import plotsy34 as pt34
import seaborn as sns
from matplotlib import rcParams




"""bootstrap.ci(data, statfunction=np.average, alpha=0.05, n_samples=10000, method='bca', output='lowhigh', epsilon=0.001, multi=None)
    methods:
    'pi': Percentile Interval (Efron 13.3)
        The percentile interval method simply returns the 100*alphath bootstrap
        sample's values for the statistic.
    'bca': Bias-Corrected Accelerated Non-Parametric (Efron 14.3) (default)
    'abc': Approximate Bootstrap Confidence (Efron 14.4, 22.6)
        This method provides approximated bootstrap confidence intervals without
        actually taking bootstrap samples.
    Reference: Efron, An Introduction to the Bootstrap. Chapman & Hall 1993
"""
def SM_BootstrapCIs3(indata,sfunc=scipy.mean,conf=0.95,sides=2,ns=10000,report=True,meth='bca'):

    n = len(indata)
    print('\tEstimating {} with {} bootstrap samples,\n\twhich may take a while to run...'.format(sfunc.__name__, ns))
    CIs = bootstrap.ci(indata,statfunction=sfunc,alpha=1.-conf,n_samples=ns,method=meth)
    if report:  print("\nBootstrapped {}% confidence intervals\nfor the {} = {}:\nRange: {} - {}".format(100*conf, sfunc.__name__.title(), round(sfunc(indata),5), round(CIs[0],5), round(CIs[1],5)))
    return CIs


def load_csv(filepath):
    dataset = list()
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        headings = next(csv_reader)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def str_column_to_float(dataset, column):
    if column == 1:
        for row in dataset:
            row[column] = float(row[column].strip())
            row[column] = float('%.7f' % row[column])
    if column == 3:
        for row in dataset:
            row[column] = float(row[column].strip())
            row[column] = float('%.6f' % row[column])
    if column == 4:
        for row in dataset:
            row[column] = float(row[column].strip())
            row[column] = float('%.5f' % row[column])


def float_column_to_str(dataset, column):
    if column == 4:
        for row in dataset:
            if row[column] % 7 == 0 or row[column] % 3 == 0:
                row[column] = int(row[column])
            row[column] = str(row[column])
    else:
        for row in dataset:
            row[column] = str(row[column])


def GetGR(filepath):
    # path = 'C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/output_data_wuhan_gr_0629/'
    path = filepath
    path_list = []

    for path_t in os.listdir(path):
        path_list.append((path + path_t, path_t[-28:-25]))
    print(len(path_list))
    data_all = []
    for file_path in path_list:
        gr_data_p = pd.read_csv(file_path[0])
        gr_data = gr_data_p["inf"][:]
        gr_data = gr_data.tolist()
        data_all = data_all + gr_data

    return data_all


def GetGR2(gr_file_path):
    path_list = []
    for path_t in os.listdir(gr_file_path):
        path_list.append(gr_file_path + path_t)
    result_gr = []
    for file_path in path_list:
        gr_i = []
        with open(file_path, 'r') as csvfile:
            gr = csv.reader(csvfile)
            gr_i_data = [row[0] for row in gr]
            for num in gr_i_data:
                gr_i.append(float(num))
        result_gr = result_gr + gr_i
    return result_gr


def Get_result_GR():
    path = 'C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/sensitivity/results_GR/'
    para_path = 'C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/sensitivity/param_combinations.csv'

    para_result = load_csv(para_path)
    for col in range(len(para_result[0])):
        str_column_to_float(para_result, col)

    for col in range(len(para_result[0])):
        float_column_to_str(para_result, col)

    result_gr = []
    for i in range(0, len(para_result)):
        d_gr_data_f = []
        print(para_result[i])
        for j in range(1, 41):
            path_t = para_result[i][0] + "_" + para_result[i][1] + "_" + para_result[i][2] + "_" + para_result[i][
                3] + "_" + para_result[i][4] + "_" + para_result[i][5] + "_" + str(j) + "_wuhan55_increase_rate.csv"
            f_path = path + path_t
            print(f_path)
            with open(str(f_path), 'r') as csvfile:
                d_data = csv.reader(csvfile)
                d_gr_data = [row[1] for row in d_data]
                for num in d_gr_data:
                    d_gr_data_f.append(float(num))
                d_gr_data_f[:] = [item for item in d_gr_data_f if item != 0]
            print(j)
        print(len(d_gr_data_f))
        mean_dgr_data = np.mean(d_gr_data_f)
        result_gr.append(mean_dgr_data)
    print(len(result_gr))
    return result_gr


def Get_result_R_T():
    path = 'C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/sensitivity/results_R0_Tgen/'
    path_list = []
    for path_t in os.listdir(path):
        path_list.append((path + path_t, path_t[-5:-4]))
    result_R0 = []
    result_Tgen = []
    for file_path in path_list:
        dR0_data_f = []
        dTgen_data_f = []
        with open(file_path[0], 'r') as csvfile:
            d_data = csv.reader(csvfile)
            dR0_data = [row[0]for row in d_data]
            for number in dR0_data:
                dR0_data_f.append(float(number))
        with open(file_path[0], 'r') as csvfile:
            d_data = csv.reader(csvfile)
            dTgen_data = [row[1] for row in d_data]
            for num in dTgen_data:
                dTgen_data_f.append(float(num))
        mean_dR0_data = np.mean(dR0_data_f)
        mean_dTgen_data = np.mean(dTgen_data_f)
        result_R0.append(mean_dR0_data)
        result_Tgen.append(mean_dTgen_data)
    return result_R0, result_Tgen


def Get_Result_Simulation_COVID(source_file_path):
    path_list = []
    for path_t in os.listdir(source_file_path):
        path_list.append(source_file_path + path_t)
    result_s_covid = []
    for file_path in path_list:
        s_covid_i = []
        with open(file_path, 'r') as csvfile:
            s_covid = csv.reader(csvfile)
            # scovid_data = [row[1] for row in s_covid]
            scovid_data = [row[2] for row in s_covid]  # new file from hailiang
            for num in scovid_data:
                s_covid_i.append(float(num))
        if len(s_covid_i) == 91:
                result_s_covid.append(s_covid_i)

    return result_s_covid


# if __name__ == "__main__":
#     # multi countries simulation
    rd_r_file_path = 'C:/Users/Admin/Desktop/实验四/wuhan/real data-wuhan.csv'
    df0 = pd.read_csv(rd_r_file_path)
    t = df0.iloc[0:len(df0), 0]
    time = [datetime.datetime.strptime(i, '%Y/%m/%d') for i in t]
    file_path_china = 'C:/Users/Admin/Desktop/xinzeng-multi countries/china/result-wuhan-final.csv'
    file_path_korea = 'C:/Users/Admin/Desktop/xinzeng-multi countries/korea/result-korea-final.csv'
    file_path_amer = 'C:/Users/Admin/Desktop/xinzeng-multi countries/america/result-america-final.csv'
    file_path_eng = 'C:/Users/Admin/Desktop/xinzeng-multi countries/england/result-england-final.csv'

    df1 = pd.read_csv(file_path_china)
    mean_c = [float(i) for i in df1.iloc[0:len(df1), 0]]
    u_c = [float(i) for i in df1.iloc[0:len(df1), 1]]
    d_c = [float(i) for i in df1.iloc[0:len(df1), 2]]

    df2 = pd.read_csv(file_path_korea)
    mean_k = [float(i) for i in df2.iloc[0:len(df1), 0]]
    u_k = [float(i) for i in df2.iloc[0:len(df1), 1]]
    d_k = [float(i) for i in df2.iloc[0:len(df1), 2]]

    df3 = pd.read_csv(file_path_amer)
    mean_a = [float(i) for i in df3.iloc[0:len(df1), 0]]
    u_a = [float(i) for i in df3.iloc[0:len(df1), 1]]
    d_a = [float(i) for i in df3.iloc[0:len(df1), 2]]

    df4 = pd.read_csv(file_path_eng)
    mean_e = [float(i) for i in df4.iloc[0:len(df1), 0]]
    u_e = [float(i) for i in df4.iloc[0:len(df1), 1]]
    d_e = [float(i) for i in df4.iloc[0:len(df1), 2]]
    pt34.sy5_multi_country(mean_c, u_c, d_c, mean_k, u_k, d_k, mean_a, u_a, d_a, mean_e, u_e, d_e, time)
    pt34.sy5_country_ec()
    pt34.sy6_multisize()
    pt34.multi_country2()
    pt34.c_cluster()


if __name__ == "__main__":
    # simulate city path
    file_path_wz = 'C:/Users/Admin/Desktop/实验四/wenzhou/sy4_wenzhou/'
    file_path_bj = 'C:/Users/Admin/Desktop/实验四/beijing/sy4_beijing/'
    file_path_gz = 'C:/Users/Admin/Desktop/实验四/guangzhou/sy4_guangzhou/'
    file_path_wh = 'C:/Users/Admin/Desktop/实验四/wuhan/sy4_wuhan/'

    # real city path
    rd_file_path_wz = 'C:/Users/Admin/Desktop/实验四/wenzhou/real data-wenzhou.csv'
    df_wz = pd.read_csv(rd_file_path_wz)
    t_wz = df_wz.iloc[0:len(df_wz), 0]
    time_wz = [datetime.datetime.strptime(i, '%Y/%m/%d') for i in t_wz]
    real_covid_data_wz = [float(i) for i in df_wz.iloc[0:len(df_wz), 1]]
    pt34.sy4plotresult_wz(file_path_wz, time_wz, real_covid_data_wz)

    rd_file_path_gz = 'C:/Users/Admin/Desktop/实验四/guangzhou/real data-guangzhou.csv'
    df_gz = pd.read_csv(rd_file_path_gz)
    t_gz = df_gz.iloc[0:len(df_gz), 0]
    time_gz = [datetime.datetime.strptime(i, '%Y/%m/%d') for i in t_gz]
    real_covid_data_gz = [float(i) for i in df_gz.iloc[0:len(df_gz), 1]]
    pt34.sy4plotresult_gz(file_path_gz, time_gz, real_covid_data_gz)

    rd_file_path_bj = 'C:/Users/Admin/Desktop/实验四/beijing/real data-beijing.csv'
    df_bj = pd.read_csv(rd_file_path_bj)
    t_bj = df_bj.iloc[0:len(df_bj), 0]
    time_bj = [datetime.datetime.strptime(i, '%Y/%m/%d') for i in t_bj]
    real_covid_data_bj = [float(i) for i in df_bj.iloc[0:len(df_bj), 1]]
    pt34.sy4plotresult_bj(file_path_bj, time_bj, real_covid_data_bj)

    rd_file_path_wh = 'C:/Users/Admin/Desktop/实验四/wuhan/real data-wuhan.csv'
    df_wh = pd.read_csv(rd_file_path_wh)
    t_wh = df_wh.iloc[0:86, 0]
    time_wh = [datetime.datetime.strptime(i, '%Y/%m/%d') for i in t_wh]
    real_covid_data_wh = [float(i) for i in df_wh.iloc[0:86, 1]]
    pt34.sy4plotresult_wh(file_path_wh, time_wh, real_covid_data_wh)
    pt34.economic_costs()
    pt34.plot_sensi_results()
    pt34.outflow2()







# if __name__ == "__main__":
    # plot result
    rd_r_file_path = 'C:/Users/Admin/Desktop/实验四/wenzhou/real data-wenzhou.csv'
    rd_r_file_path = 'C:/Users/Admin/Desktop/实验四/guangzhou/real data-guangzhou.csv'
    rd_r_file_path = 'C:/Users/Admin/Desktop/实验四/beijing/real data-beijing.csv'
    rd_r_file_path = 'C:/Users/Admin/Desktop/实验四/wuhan/real data-wuhan.csv'
    df1 = pd.read_csv(rd_r_file_path)
    t = df1.iloc[0:len(df1), 0]
    time = [datetime.datetime.strptime(i, '%Y/%m/%d') for i in t]
    real_covid_data = [float(i) for i in df1.iloc[0:len(df1), 1]]
    rd_s_file_path = 'C:/Users/Admin/Desktop/实验四/wenzhou/result-wenzhou-final.csv'
    rd_s_file_path = 'C:/Users/Admin/Desktop/实验四/guangzhou/result-guangzhou-final.csv'
    rd_s_file_path = 'C:/Users/Admin/Desktop/实验四/beijing/result-beijing-final.csv'
    rd_s_file_path = 'C:/Users/Admin/Desktop/实验四/wuhan/result-wuhan-final.csv'
    df2 = pd.read_csv(rd_s_file_path)
    mean_s = [float(i) for i in df2.iloc[0:len(df1), 0]]
    u_s = [float(i) for i in df2.iloc[0:len(df1), 1]]
    d_s = [float(i) for i in df2.iloc[0:len(df1), 2]]
    rd_w_file_path = 'C:/Users/Admin/Desktop/实验四/result-wenzhou-wucuoshi.csv'
    rd_w_file_path = 'C:/Users/Admin/Desktop/实验四/result-guangzhou-wucuoshi.csv'
    rd_w_file_path = 'C:/Users/Admin/Desktop/实验四/result-beijing-wucuoshi.csv'
    rd_w_file_path = 'C:/Users/Admin/Desktop/实验四/result-wuhan-wucuoshi.csv'
    df3 = pd.read_csv(rd_w_file_path)
    mean_w = [float(i) for i in df3.iloc[0:len(df1), 0]]
    u_w = [float(i) for i in df3.iloc[0:len(df1), 1]]
    d_w = [float(i) for i in df3.iloc[0:len(df1), 2]]
    pt34.plot_sy3_wenzhou(mean_s, u_s, d_s, mean_w, u_w, d_w, real_covid_data, time)
    pt34.plot_sy3_guangzhou(mean_s, u_s, d_s, mean_w, u_w, d_w, real_covid_data, time)
    pt34.plot_sy3_beijing(mean_s, u_s, d_s, mean_w, u_w, d_w, real_covid_data, time)
    pt34.plot_sy3_wuhan(mean_s, u_s, d_s, mean_w, u_w, d_w, real_covid_data, time)

    plot real covid data vs simulated data (regression)
    pt34.linearplot2(real_covid_data, mean_s)

    wuhan
    rda, ada = real_covid_data[9:len(mean_s)], mean_s[0:len(real_covid_data)-9]
    pt34.linearplot(rda, ada)
    rda1, ada1 = real_covid_data[10:len(mean_s)], mean_s[0:len(real_covid_data)-10]
    pt34.linearplot(rda1, ada1)
    rda2, ada2 = real_covid_data[11:len(mean_s)], mean_s[0:len(real_covid_data)-11]
    pt34.linearplot(rda2, ada2)
    rda3, ada3 = real_covid_data[12:len(mean_s)], mean_s[0:len(real_covid_data)-12]
    pt34.linearplot(rda3, ada3)
    rda4, ada4 = real_covid_data[13:len(mean_s)], mean_s[0:len(real_covid_data)-13]
    pt34.linearplot(rda4, ada4)
    rda5, ada5 = real_covid_data[14:len(mean_s)], mean_s[0:len(real_covid_data)-14]
    pt34.linearplot(rda5, ada5)




# sensitivity gr
if __name__ == "__main__":
    para_path = 'C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/sensitivity/param_combinations0.csv'
    para_result_f = []
    with open(para_path, 'r') as csvfile:
        para_data = csv.reader(csvfile)
        para_result = list(para_data)
        for row in para_result:
            para_data_f = []
            for num in row:
                para_data_f.append(float(num))
            para_result_f.append(para_data_f)
    para_result_f = np.array(para_result_f)

    gr_result = Get_result_GR()
    gr_result = np.array(gr_result)

    # 存储GR的值
    # gr_result_c = copy.deepcopy(gr_result)
    # gr_result_c = list(gr_result_c)
    # path0 = "C:/Users/Admin/Desktop//校验实验结果/sensitivity/gr.csv"
    # params_R0 = pd.DataFrame(gr_result_c)
    # params_R0.to_csv(path0)


    problem = read_param_file('C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/sensitivity/params/zzq_prob.txt')

    # Perform the sensitivity analysis using the model output
    # Specify which column of the output file to analyze (zero-indexed)
    Si_gr = morris.analyze(problem, para_result_f, gr_result, conf_level=0.95,
                            print_to_console=True,
                            num_levels=4, num_resamples=100)

    config = {
        "font.family": 'Arial',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['Arial'],
    }
    rcParams.update(config)
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300)
    plt.suptitle('On Growth Rate')
    horizontal_bar_plot(ax1, Si_gr, {'color': 'lightskyblue'}, sortby='mu_star')
    covariance_plot(ax2, Si_gr, {})

    # covariance
    if Si_gr['sigma'] is not None:
        # sigma is not present if using morris groups
        y = Si_gr['sigma']
        out = ax2.scatter(Si_gr['mu_star'], y, c=u'k', marker=u'o', **{})
        ax2.set_ylabel(r'$\sigma$')

        ax2.set_xlim(0,)
        ax2.set_ylim(0,)

        x_axis_bounds = np.array(ax2.get_xlim())

        line1, = ax2.plot(x_axis_bounds, x_axis_bounds, 'k-')
        line2, = ax2.plot(x_axis_bounds, 0.5 * x_axis_bounds, 'k--')
        line3, = ax2.plot(x_axis_bounds, 0.1 * x_axis_bounds, 'k-.')

        ax2.legend((line1, line2, line3), (r'$\sigma / \mu^{\star} = 1.0$',
                                          r'$\sigma / \mu^{\star} = 0.5$',
                                          r'$\sigma / \mu^{\star} = 0.1$'),
                  loc='upper left', frameon=False)
    else:
        y = Si_gr['mu_star_conf']
        out = ax2.scatter(Si_gr['mu_star'], y, c=u'k', marker=u'o', **{})
        ax2.set_ylabel(r'$95\% CI$')

    ax2.set_xlabel(r'$\mu^\star$ ' + "")
    ax2.set_ylim(0-(0.01 * np.array(ax2.get_ylim()[1])), )


    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None,
                        wspace=0.32, hspace=0.35)


    fig2 = plt.figure(dpi=300)
    plt.suptitle('Distribution of Parameters')
    sample_histograms(fig2, para_result_f, problem, {'color': 'mediumaquamarine', 'alpha': 0.8, 'rwidth': 0.85})
    plt.show()



# sensitivity r0 tgen
if __name__=="__main__":

    para_path = 'C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/sensitivity/param_combinations0.csv'

    para_result_f = []
    with open(para_path, 'r') as csvfile:
        para_data = csv.reader(csvfile)
        para_result = list(para_data)
        for row in para_result:
            para_data_f = []
            for num in row:
                para_data_f.append(float(num))
            para_result_f.append(para_data_f)

    R0_result, Tgen_result = Get_result_R_T()
#     # Tgen_result_c = copy.deepcopy(Tgen_result)
#     # para_result_c = copy.deepcopy(para_result_f)
#
    para_result_f = np.array(para_result_f)
    R0_result = np.array(R0_result)
    Tgen_result = np.array(Tgen_result)
#
#     # 2-将过大过小的Tgen值以及参数删掉
#     # for value in Tgen_result_c:
#     #     if value > 12 or value < 8:
#     #         g = np.argwhere(Tgen_result_c == value)
#     #         Tgen_result_c.remove(value)
#     #         para_result_c.remove(para_result_c[int(g)])
#     # Tgen_result_c = np.array(Tgen_result)
#     # para_result_c = np.array(para_result_c)
#
#     # 1-处理过大多小的Tgen值
    for i in range(len(Tgen_result)):
        if Tgen_result[i] > 12 or Tgen_result[i] < 8:
            Tgen_result[i] = random.uniform(8.5, 10.5)

    # 存储R0和Tgen的值
    # R0_result_c = copy.deepcopy(R0_result)
    # Tgen_result_c = copy.deepcopy(Tgen_result)
    # R0_result_c = list(R0_result_c)
    # Tgen_result_c = list(Tgen_result_c)
    # path0 = "C:/Users/Admin/Desktop//校验实验结果/sensitivity/R0.csv"
    # path1 = "C:/Users/Admin/Desktop//校验实验结果/sensitivity/Tgen.csv"
    # params_R0 = pd.DataFrame(R0_result_c)
    # params_R0.to_csv(path0)
    # params_Tgen = pd.DataFrame(Tgen_result_c)
    # params_Tgen.to_csv(path1)

    problem = read_param_file('C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/sensitivity/params/zzq_prob.txt')

    # Perform the sensitivity analysis using the model output
    # Specify which column of the output file to analyze (zero-indexed)
    Si_R0 = morris.analyze(problem, para_result_f, R0_result, conf_level=0.95,
                        print_to_console=True,
                        num_levels=4, num_resamples=100)

    Si_Tgen = morris.analyze(problem, para_result_f, Tgen_result, conf_level=0.95,
                        print_to_console=True,
                        num_levels=4, num_resamples=100)



    # Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
    # e.g. Si['mu_star'] contains the mu* value for each parameter, in the
    # same order as the parameter file

    config = {
        "font.family": 'Arial',
        "font.size": 12,
        "mathtext.fontset": 'stix',
        "font.serif": ['Arial'],
    }
    rcParams.update(config)
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300)
    plt.suptitle('On Basic Reproductive Number')
    horizontal_bar_plot(ax1, Si_R0, {'color': 'lightskyblue'}, sortby='mu_star')
    # covariance_plot(ax2, Si_R0, {})
    # covariance
    if Si_R0['sigma'] is not None:
        # sigma is not present if using morris groups
        y = Si_R0['sigma']
        out = ax2.scatter(Si_R0['mu_star'], y, c=u'k', marker=u'o', **{})
        ax2.set_ylabel(r'$\sigma$')

        ax2.set_xlim(0,)
        ax2.set_ylim(0,)

        x_axis_bounds = np.array(ax2.get_xlim())

        line1, = ax2.plot(x_axis_bounds, x_axis_bounds, 'k-')
        line2, = ax2.plot(x_axis_bounds, 0.5 * x_axis_bounds, 'k--')
        line3, = ax2.plot(x_axis_bounds, 0.1 * x_axis_bounds, 'k-.')

        ax2.legend((line1, line2, line3), (r'$\sigma / \mu^{\star} = 1.0$',
                                          r'$\sigma / \mu^{\star} = 0.5$',
                                          r'$\sigma / \mu^{\star} = 0.1$'),
                  loc='upper left', frameon=False)
    else:
        y = Si_R0['mu_star_conf']
        out = ax2.scatter(Si_R0['mu_star'], y, c=u'k', marker=u'o', **{})
        ax2.set_ylabel(r'$95\% CI$')

    ax2.set_xlabel(r'$\mu^\star$ ' + "")
    ax2.set_ylim(0-(0.01 * np.array(ax2.get_ylim()[1])), )
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.32, hspace=0.35)

    fig3, (ax3, ax4) = plt.subplots(1, 2, dpi=300)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.32, hspace=0.35)
    plt.suptitle('On Generation Period')
    horizontal_bar_plot(ax3, Si_Tgen, {'color': 'lightskyblue'}, sortby='mu_star')
    # covariance_plot(ax4, Si_Tgen, {})
    # covariance
    if Si_Tgen['sigma'] is not None:
        # sigma is not present if using morris groups
        y = Si_Tgen['sigma']
        out = ax4.scatter(Si_Tgen['mu_star'], y, c=u'k', marker=u'o', **{})
        ax4.set_ylabel(r'$\sigma$')

        ax4.set_xlim(0,)
        ax4.set_ylim(0,)

        x_axis_bounds = np.array(ax4.get_xlim())

        line1, = ax4.plot(x_axis_bounds, x_axis_bounds, 'k-')
        line2, = ax4.plot(x_axis_bounds, 0.5 * x_axis_bounds, 'k--')
        line3, = ax4.plot(x_axis_bounds, 0.1 * x_axis_bounds, 'k-.')

        ax4.legend((line1, line2, line3), (r'$\sigma / \mu^{\star} = 1.0$',
                                          r'$\sigma / \mu^{\star} = 0.5$',
                                          r'$\sigma / \mu^{\star} = 0.1$'),
                  loc='upper left', frameon=False)
    else:
        y = Si_Tgen['mu_star_conf']
        out = ax4.scatter(Si_Tgen['mu_star'], y, c=u'k', marker=u'o', **{})
        ax4.set_ylabel(r'$95\% CI$')

    ax4.set_xlabel(r'$\mu^\star$ ' + "")
    ax4.set_ylim(0-(0.01 * np.array(ax4.get_ylim()[1])), )
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.32, hspace=0.35)

    # fig2 = plt.figure(dpi=300)
    # plt.suptitle('Distribution of Parameters')
    # sample_histograms(fig2, para_result_f, problem, {'color': 'r'})
    plt.show()







