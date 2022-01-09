# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import scipy
import datetime
import matplotlib.dates as mdate
from matplotlib.pyplot import MultipleLocator
import scipy.stats as st
import statsmodels.api as sm
import seaborn as sns
import os
import numpy as np
import csv
from labellines import labelLines
from matplotlib.dates import date2num
import matplotlib.ticker
import math
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import matplotlib.ticker as mtick
from matplotlib import ticker


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


def to_percent(temp, position):
    return '%1.0f'%(10*temp) + '%'


def plot_sy3_wuhan(adata, udata, ddata, wadata, wudata, wddata, rdata, time):
    aver_d, up_d, down_d, real_d = adata, udata, ddata, rdata
    fig = plt.figure(dpi=300, figsize=(8, 5))
    plt.ylim(-5000, 70000)

    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }
    plt.title('Wuhan', font1)
    plt.xlabel(u'Date/day', font1, verticalalignment='top', labelpad=10)
    plt.ylabel(u'Cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center', labelpad=15)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))

    y_major_locator = MultipleLocator(10000)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xticks(pd.date_range('2020-01-07', '2020-03-31', freq='12D'), rotation=0)

    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax.plot(time, aver_d, 's-', color='k', linewidth=1, label='Average simulated cumulative incidence',
            markersize=3)
    ax.fill_between(time, down_d, up_d, color='darkgrey', alpha=0.5)
    ax.plot(time, real_d, 'o-', color='red', linewidth=1, label='Official reported cumulative incidence', markersize=3)
    ax.vlines(['2020-01-07'], -5000, 54.6, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/05'), '%Y/%m/%d'), 200, "ST1",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('China CDC Level 2 \n emergency response \n activated',
                xy=(datetime.datetime.strptime(('2020/01/07'), '%Y/%m/%d'), 54.6),
                xytext=(datetime.datetime.strptime(('2020/01/07'), '%Y/%m/%d'), 7000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='center', fontname="Arial")
    ax.vlines(['2020-01-13'], -5000, 333.17, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/11'), '%Y/%m/%d'), 1400, "ST2",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('PCR diagnostic reagents \n were put into use',
                xy=(datetime.datetime.strptime(('2020/01/13'), '%Y/%m/%d'), 333.17),
                xytext=(datetime.datetime.strptime(('2020/01/13'), '%Y/%m/%d'), 12000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='center', fontname="Arial")
    ax.text(datetime.datetime.strptime(('2020/01/13'), '%Y/%m/%d'), 15500, "Wuhan implemented the control \n on outbound passengers",
            fontsize=5, color="k", horizontalalignment='center', fontname="Arial")
    ax.vlines(['2020-01-22'], -5000, 7492.72, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/20'), '%Y/%m/%d'), 8000, "ST3",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('launched the level-2 response to \n major public health emergencies',
                xy=(datetime.datetime.strptime(('2020/01/22'), '%Y/%m/%d'), 7492.72),
                xytext=(datetime.datetime.strptime(('2020/01/22'), '%Y/%m/%d'), 20000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-01-24'], -5000, 12970.23, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/26'), '%Y/%m/%d'), 13200, "ST4",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('updated to level-1 response and \n full lockdown measure',
                xy=(datetime.datetime.strptime(('2020/01/24'), '%Y/%m/%d'), 12970.23),
                xytext=(datetime.datetime.strptime(('2020/01/24'), '%Y/%m/%d'), 26000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-02-02'], -5000, 35865.61, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/31'), '%Y/%m/%d'), 36000, "ST5",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Wuhan centrally treated and isolated \n four types of personnel',
                xy=(datetime.datetime.strptime(('2020/02/02'), '%Y/%m/%d'), 35865.61),
                xytext=(datetime.datetime.strptime(('2020/02/02'), '%Y/%m/%d'), 42000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-02-06'], -5000, 43990.36, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/02/04'), '%Y/%m/%d'), 44000, "ST6",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Wuhan implemented temperature monitoring \n for the whole population once a day',
                xy=(datetime.datetime.strptime(('2020/02/06'), '%Y/%m/%d'), 43990.36),
                xytext=(datetime.datetime.strptime(('2020/02/06'), '%Y/%m/%d'), 50000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-02-11'], -5000, 48930.31, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/02/09'), '%Y/%m/%d'), 50000, "ST7",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Wuhan implemented full lockdown \n in all residential districts',
                xy=(datetime.datetime.strptime(('2020/02/11'), '%Y/%m/%d'), 48930.31),
                xytext=(datetime.datetime.strptime(('2020/02/11'), '%Y/%m/%d'), 54000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='right', fontname="Arial")
    fontt = {'family': 'Arial',
             'weight': 'normal',
             'size': 10,
             }
    ax.legend(loc='upper left', frameon=False, prop=fontt)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    i_r = np.diff(rdata)
    s_r = np.diff(adata)
    w_r = np.diff(wadata)
    time2 = time[1:len(time)]
    left, bottom, width, height = 0.6, 0.25, 0.35, 0.35
    ax2 = fig.add_axes([left, bottom, width, height])
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 8,
             }
    ax2.set_xlabel(u'Date/day', font2, verticalalignment='top')
    ax2.set_ylabel(u'incidence', font2, verticalalignment='baseline', horizontalalignment='center')
    ax2.set_title('Comparison to incidence of no intervention', font2)
    ax2.plot(time2, i_r, 'o-.', color='r', linewidth=1, label='Official reported incidence', markersize=2)
    ax2.plot(time2, s_r, 's-.', color='k', linewidth=1, label='Average simulated incidence of intervention', markersize=2)
    ax2.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax2.set_xticks(pd.date_range('2020-01-08', '2020-03-31', freq='12D'))

    ax2.tick_params(labelsize=6)
    labels2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]

    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 5,
             }
    ax2.legend(loc='upper left', frameon=False, prop=font3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    left2, bottom2, width2, height2 = 0.8, 0.35, 0.12, 0.12
    ax3 = fig.add_axes([left2, bottom2, width2, height2])
    # ax3.set_yscale('log')
    # ax3.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=100.0, numticks=5))
    ax3.plot(time2, w_r, 'p-.', color='saddlebrown', linewidth=1, label='Average simulated incidence of no intervention', markersize=1)
    # ax3.xaxis.set_major_locator(plt.NullLocator())
    ax3.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax3.set_xticks(pd.date_range('2020-01-08', '2020-03-31', freq='18D'))
    ax3.tick_params(labelsize=5)
    labels3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Arial') for label in labels3]
    ax3.set_title('Incidence of no intervention', font3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.show()


def plot_sy3_wenzhou(adata, udata, ddata, wadata, wudata, wddata, rdata, time):
    aver_d, up_d, down_d, real_d = adata, udata, ddata, rdata
    fig = plt.figure(dpi=300, figsize=(8, 5))
    plt.ylim(-50, 700)
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }
    plt.title('Wenzhou', font1)
    plt.xlabel(u'Date/day', font1, verticalalignment='top', labelpad=10)
    plt.ylabel(u'Cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center', labelpad=15)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    y_major_locator = MultipleLocator(100)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xticks(pd.date_range('2020-01-20', '2020-03-31', freq='10D'), rotation=0)


    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax.plot(time, aver_d, 's-', color='k', linewidth=1, label='Average simulated cumulative incidence',
            markersize=3)
    ax.fill_between(time, down_d, up_d, color='darkgrey', alpha=0.5)
    ax.plot(time, real_d, 'o-', color='red', linewidth=1, label='Official reported cumulative incidence', markersize=3)
    ax.vlines(['2020-01-21'], -50, 2.98, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/20'), '%Y/%m/%d'), 20, "ST1",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Wenzhou \n started to \n response',
                xy=(datetime.datetime.strptime(('2020/01/21'), '%Y/%m/%d'), 2.98),
                xytext=(datetime.datetime.strptime(('2020/01/21'), '%Y/%m/%d'), 50),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=6, horizontalalignment='center', fontname="Arial")
    ax.vlines(['2020-01-23'], -50, 8.58, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/24'), '%Y/%m/%d'), 40, "ST2",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Wenzhou launched the \n level-1 response',
                xy=(datetime.datetime.strptime(('2020/01/23'), '%Y/%m/%d'), 8.58),
                xytext=(datetime.datetime.strptime(('2020/01/23'), '%Y/%m/%d'), 130),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=6, horizontalalignment='center', fontname="Arial")
    ax.vlines(['2020-02-01'], -50, 282.05, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/31'), '%Y/%m/%d'), 290, "ST3",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Wenzhou implemented the \n full lockdown measures',
                xy=(datetime.datetime.strptime(('2020/02/01'), '%Y/%m/%d'), 282.05),
                xytext=(datetime.datetime.strptime(('2020/02/01'), '%Y/%m/%d'), 350),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=6, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-02-03'], -50, 347.79, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/02/02'), '%Y/%m/%d'), 355, "ST4",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('46 high-speed tollbooths \n were temporarily closed ',
                xy=(datetime.datetime.strptime(('2020/02/03'), '%Y/%m/%d'), 347.79),
                xytext=(datetime.datetime.strptime(('2020/02/03'), '%Y/%m/%d'), 420),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=6, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-02-08'], -50, 443.21, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/02/07'), '%Y/%m/%d'), 465, "ST5",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Wenzhou lifted restrictions on travel \n and only implemented partial lockdown',
                xy=(datetime.datetime.strptime(('2020/02/08'), '%Y/%m/%d'), 443.21),
                xytext=(datetime.datetime.strptime(('2020/02/08'), '%Y/%m/%d'), 500),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=6, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-03-02'], -50, 510.88, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/03/01'), '%Y/%m/%d'), 540, "ST6",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Wenzhou updated to level-2 response \n to major public health emergencies',
                xy=(datetime.datetime.strptime(('2020/03/02'), '%Y/%m/%d'), 510.88),
                xytext=(datetime.datetime.strptime(('2020/03/02'), '%Y/%m/%d'), 580),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=6, horizontalalignment='right', fontname="Arial")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fontt = {'family': 'Arial',
             'weight': 'normal',
             'size': 10,
             }
    ax.legend(loc='upper left', frameon=False, prop=fontt)


    i_r = np.diff(rdata)
    s_r = np.diff(adata)
    w_r = np.diff(wadata)
    time2 = time[1:len(time)]
    time3 = time2[0:len(time2)-1]
    left, bottom, width, height = 0.65, 0.25, 0.3, 0.3
    ax2 = fig.add_axes([left, bottom, width, height])
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 8,
             }
    ax2.set_ylim(0, 85)
    ax2.set_xlabel(u'Date/day', font2, verticalalignment='top')
    ax2.set_ylabel(u'incidence', font2, verticalalignment='baseline', horizontalalignment='center')
    ax2.set_title('Comparison to incidence of no intervention', font2)
    ax2.plot(time2, i_r, 'o-.', color='r', linewidth=1, label='Official reported incidence', markersize=1.5)
    ax2.plot(time2, s_r, 's-.', color='k', linewidth=1, label='Average simulated incidence of intervention',
             markersize=1.5)
    ax2.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax2.set_xticks(pd.date_range('2020-01-21', '2020-03-31', freq='12D'))

    ax2.tick_params(labelsize=6)
    labels2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]
    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 5,
             }
    ax2.legend(loc='upper left', frameon=False, prop=font3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    left2, bottom2, width2, height2 = 0.8, 0.3, 0.12, 0.15
    ax3 = fig.add_axes([left2, bottom2, width2, height2])
    ax3.plot(time3, w_r, 'p-.', color='saddlebrown', linewidth=1,
             label='Average simulated incidence of no intervention', markersize=1)
    # ax3.xaxis.set_major_locator(plt.NullLocator())
    ax3.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax3.set_xticks(pd.date_range('2020-01-21', '2020-03-31', freq='18D'))
    ax3.tick_params(labelsize=5)
    labels3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Arial') for label in labels3]
    ax3.set_title('Incidence of no intervention', font3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.show()


def plot_sy3_guangzhou(adata, udata, ddata, wadata, wudata, wddata, rdata, time):
    aver_d, up_d, down_d, real_d = adata, udata, ddata, rdata
    fig = plt.figure(dpi=300, figsize=(8, 5))
    plt.ylim(-50, 500)
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }
    plt.title('Guangzhou', font1)
    plt.xlabel(u'Date/day', font1, verticalalignment='top',labelpad=10)
    plt.ylabel(u'Cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center', labelpad=15)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    y_major_locator = MultipleLocator(100)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xticks(pd.date_range('2020-01-21', '2020-03-31', freq='10D'), rotation=0)


    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax.plot(time, aver_d, 's-', color='k', linewidth=1, label='Average simulated cumulative incidence',
            markersize=3)
    ax.fill_between(time, down_d, up_d, color='darkgrey', alpha=0.5)
    ax.plot(time, real_d, 'o-', color='red', linewidth=1, label='Official reported cumulative incidence', markersize=3)

    ax.vlines(['2020-01-21'], -50, 2.0, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/20'), '%Y/%m/%d'), 10, "ST1",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Guangzhou \n started to \n  response',
                xy=(datetime.datetime.strptime(('2020/01/21'), '%Y/%m/%d'), 2.0),
                xytext=(datetime.datetime.strptime(('2020/01/21'), '%Y/%m/%d'), 35),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='center', fontname="Arial")
    ax.vlines(['2020-01-23'], -50, 5.302, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/24'), '%Y/%m/%d'), 22, "ST2",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Guangzhou launched the \n level-1 response',
                xy=(datetime.datetime.strptime(('2020/01/23'), '%Y/%m/%d'), 5.302),
                xytext=(datetime.datetime.strptime(('2020/01/23'), '%Y/%m/%d'), 100),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='center', fontname="Arial")
    ax.vlines(['2020-01-27'], -50, 34.459, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/26'), '%Y/%m/%d'), 50, "ST3",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Citizens were required to wear \n face masks in public places \n and  travel was restriction',
                xy=(datetime.datetime.strptime(('2020/01/27'), '%Y/%m/%d'), 34.459),
                xytext=(datetime.datetime.strptime(('2020/01/27'), '%Y/%m/%d'), 150),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-01-30'], -50, 102.9678, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/29'), '%Y/%m/%d'), 110, "ST4",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Guangzhou carried out  specific \n screening  of outpatients with fever',
                xy=(datetime.datetime.strptime(('2020/01/30'), '%Y/%m/%d'), 102.9678),
                xytext=(datetime.datetime.strptime(('2020/01/30'), '%Y/%m/%d'), 210),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-02-07'], -50, 304.561, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/02/06'), '%Y/%m/%d'), 310, "ST5",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Guangzhou implemented the full lockdown measures',
                xy=(datetime.datetime.strptime(('2020/02/07'), '%Y/%m/%d'), 304.561),
                xytext=(datetime.datetime.strptime(('2020/02/07'), '%Y/%m/%d'), 350),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-02-24'], -50, 348.0025, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/02/23'), '%Y/%m/%d'), 360, "ST6",
            fontsize=7, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Guangzhou updated to level-2 response \n to major public health emergencies',
                xy=(datetime.datetime.strptime(('2020/02/24'), '%Y/%m/%d'), 348.0025),
                xytext=(datetime.datetime.strptime(('2020/02/24'), '%Y/%m/%d'), 390),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5, horizontalalignment='right', fontname="Arial")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fontt = {'family': 'Arial',
             'weight': 'normal',
             'size': 10,
             }
    ax.legend(loc='upper left', frameon=False, prop=fontt)


    i_r = np.diff(rdata)
    s_r = np.diff(adata)
    w_r = np.diff(wadata)
    time2 = time[1:len(time)]
    time3 = time2[0:len(time2) - 1]
    left, bottom, width, height = 0.65, 0.25, 0.3, 0.3
    ax2 = fig.add_axes([left, bottom, width, height])
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 8,
             }
    ax2.set_ylim(0, 60)
    ax2.set_xlabel(u'Date/day', font2, verticalalignment='top')
    ax2.set_ylabel(u'incidence', font2, verticalalignment='baseline', horizontalalignment='center')
    ax2.set_title('Comparison to incidence of no intervention', font2)
    ax2.plot(time2, i_r, 'o-.', color='r', linewidth=1, label='Official reported incidence', markersize=1.5)
    ax2.plot(time2, s_r, 's-.', color='k', linewidth=1, label='Average simulated incidence of intervention',
             markersize=1.5)
    ax2.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax2.set_xticks(pd.date_range('2020-01-22', '2020-03-31', freq='12D'))

    ax2.tick_params(labelsize=6)
    labels2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]
    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 5,
             }
    ax2.legend(loc='upper left', frameon=False, prop=font3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    left2, bottom2, width2, height2 = 0.8, 0.3, 0.12, 0.15
    ax3 = fig.add_axes([left2, bottom2, width2, height2])
    ax3.plot(time3, w_r, 'p-.', color='saddlebrown', linewidth=1,
             label='Average simulated incidence of no intervention', markersize=1)
    # ax3.xaxis.set_major_locator(plt.NullLocator())
    ax3.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax3.set_xticks(pd.date_range('2020-01-22', '2020-03-31', freq='18D'))
    ax3.tick_params(labelsize=5)
    labels3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Arial') for label in labels3]
    ax3.set_title('Incidence of no intervention', font3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.show()


def plot_sy3_beijing(adata, udata, ddata, wadata, wudata, wddata, rdata, time):
    aver_d, up_d, down_d, real_d = adata, udata, ddata, rdata
    fig = plt.figure(dpi=300, figsize=(8, 5))
    plt.ylim(-50, 700)
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }
    plt.title('Beijing', font1)
    plt.xlabel(u'Date/day', font1, verticalalignment='top',labelpad=10)
    plt.ylabel(u'Cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center', labelpad=15)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    y_major_locator = MultipleLocator(100)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xticks(pd.date_range('2020-01-20', '2020-03-31', freq='10D'), rotation=0)


    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax.plot(time, aver_d, 's-', color='k', linewidth=1, label='Average simulated cumulative incidence',
            markersize=3)
    ax.fill_between(time, down_d, up_d, color='darkgrey', alpha=0.5)
    ax.plot(time, real_d, 'o-', color='red', linewidth=1, label='Official reported cumulative incidence', markersize=3)
    ax.vlines(['2020-01-20'], -50, 6.24, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/19'), '%Y/%m/%d'), 10, "ST1",
            fontsize=6.5, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Beijing \n started \n to response',
                xy=(datetime.datetime.strptime(('2020/01/20'), '%Y/%m/%d'), 6.24),
                xytext=(datetime.datetime.strptime(('2020/01/20'), '%Y/%m/%d'), 45),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5.5, horizontalalignment='center', fontname="Arial")
    ax.vlines(['2020-01-24'], -50, 30.07, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/25'), '%Y/%m/%d'), 65, "ST2",
            fontsize=6.5, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Beijing launched the \n level-1 response',
                xy=(datetime.datetime.strptime(('2020/01/24'), '%Y/%m/%d'), 30.07),
                xytext=(datetime.datetime.strptime(('2020/01/24'), '%Y/%m/%d'), 110),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5.5, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-01-26'], -50, 61.02, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/01/27'), '%Y/%m/%d'), 112, "ST3",
            fontsize=6.5, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('The inter-provincial passenger \n transport in and out of beijing \n were suspended',
                xy=(datetime.datetime.strptime(('2020/01/26'), '%Y/%m/%d'), 61.02),
                xytext=(datetime.datetime.strptime(('2020/01/26'), '%Y/%m/%d'), 175),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=5.5, horizontalalignment='center', fontname="Arial")
    ax.vlines(['2020-02-02'], -50, 219.15, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/02/01'), '%Y/%m/%d'), 235, "ST4",
            fontsize=6.5, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Beijing strengthened the management \n on community',
                xy=(datetime.datetime.strptime(('2020/02/02'), '%Y/%m/%d'), 219.15),
                xytext=(datetime.datetime.strptime(('2020/02/02'), '%Y/%m/%d'), 350),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=6, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-02-10'], -50, 347.64, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/02/09'), '%Y/%m/%d'), 365, "ST5",
            fontsize=6.5, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Full lockdown measures were implemented on community',
                xy=(datetime.datetime.strptime(('2020/02/10'), '%Y/%m/%d'), 347.64),
                xytext=(datetime.datetime.strptime(('2020/02/10'), '%Y/%m/%d'), 420),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=6, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-02-14'], -50, 373.38, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/02/13'), '%Y/%m/%d'), 390, "ST6",
            fontsize=6.5, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('A 14-day self-quarantine has been imposed \n on all those returning to Beijing',
                xy=(datetime.datetime.strptime(('2020/02/14'), '%Y/%m/%d'), 373.38),
                xytext=(datetime.datetime.strptime(('2020/02/14'), '%Y/%m/%d'), 460),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=6, horizontalalignment='right', fontname="Arial")
    ax.vlines(['2020-03-10'], -50, 430.62, linestyles='dashed', colors='darkgrey')
    ax.text(datetime.datetime.strptime(('2020/03/09'), '%Y/%m/%d'), 460, "ST7",
            fontsize=6.5, color="red", horizontalalignment='center', fontname="Arial")
    ax.annotate('Strengthened quarantine measures for inbound international flights',
                xy=(datetime.datetime.strptime(('2020/03/10'), '%Y/%m/%d'), 430.62),
                xytext=(datetime.datetime.strptime(('2020/03/10'), '%Y/%m/%d'), 530),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4), fontsize=6, horizontalalignment='right', fontname="Arial")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fontt = {'family': 'Arial',
             'weight': 'normal',
             'size': 10,
             }
    ax.legend(loc='upper left', frameon=False, prop=fontt)


    i_r = np.diff(rdata)
    s_r = np.diff(adata)
    w_r = np.diff(wadata)
    time2 = time[1:len(time)]
    left, bottom, width, height = 0.47, 0.22, 0.23, 0.28
    ax2 = fig.add_axes([left, bottom, width, height])
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 8,
             }
    ax2.set_ylim(0, 60)
    ax2.set_xlabel(u'Date/day', font2, verticalalignment='top')
    ax2.set_ylabel(u'incidence', font2, verticalalignment='baseline', horizontalalignment='center')
    ax2.set_title('Comparison to incidence of no intervention', font2)
    ax2.plot(time2, i_r, 'o-.', color='r', linewidth=1, label='Official reported incidence', markersize=1.5)
    ax2.plot(time2, s_r, 's-.', color='k', linewidth=1, label='Average simulated incidence of intervention',
             markersize=1.5)
    ax2.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax2.set_xticks(pd.date_range('2020-01-22', '2020-03-31', freq='12D'))

    ax2.tick_params(labelsize=6)
    labels2 = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]
    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 5,
             }
    ax2.legend(loc='upper left', frameon=False, prop=font3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    left2, bottom2, width2, height2 = 0.55, 0.30, 0.10, 0.10
    ax3 = fig.add_axes([left2, bottom2, width2, height2])
    ax3.plot(time2, w_r, 'p-.', color='saddlebrown', linewidth=1,
             label='Average simulated incidence of no intervention', markersize=1)
    # ax3.xaxis.set_major_locator(plt.NullLocator())
    ax3.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax3.set_xticks(pd.date_range('2020-01-21', '2020-03-31', freq='18D'))
    ax3.tick_params(labelsize=5)
    labels3 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Arial') for label in labels3]
    ax3.set_title('Incidence of no intervention', font3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.show()


def linearplot2(rd, ad):
    slope, intercept, r_value, p_value, std_err = st.linregress(rd, ad)
    r2 = r_value * r_value
    p_value = '{:.2e}'.format(p_value)
    y = ad
    x = rd
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    print(model.summary())
    predicts = model.predict()
    x = rd
    fig = plt.figure(dpi=300, figsize=(6.25, 5))
    # # wenzhou
    plt.ylim(0, 520)
    plt.xlim(0, 510)
    # # guangzhou
    # plt.ylim(0, 360)
    # plt.xlim(0, 350)
    # beijing
    # plt.ylim(0, 600)
    # plt.xlim(0, 600)
    plt.scatter(x, y, color='white', edgecolors='black', alpha=0.9, label='Scatter points used in the fitting')
    # sns.regplot(x, y, color='tab:red', ci=99, scatter=False)
    plt.plot(x, x, color='r', label='y = x')
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }
    plt.xlabel(u'Official reported cumulative incidence', font1, verticalalignment='top')
    plt.ylabel(u'Average simulated cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center')

    ax = plt.subplot(111)
    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    # wenzhou
    plt.text(348, 150, "$R^2$=", fontsize=12, color="black", style="italic", weight="light",
             horizontalalignment='left', rotation=0)
    plt.text(383, 150, float('%0.3f' % r2), fontsize=12, color="black", style="italic", weight="light",
             horizontalalignment='left', rotation=0)
    plt.text(348, 125, "p_value=", fontsize=12, color="black", style="italic", weight="light",
             horizontalalignment='left', rotation=0)
    plt.text(423, 125, p_value, fontsize=12, color="black", style="italic", weight="light",
             horizontalalignment='left', rotation=0)
    plt.text(260, 20, "Wenzhou", fontsize=10, color="black", weight="heavy",
             horizontalalignment='center', rotation=0)
    # guangzhou
    # plt.text(238, 110, "$R^2$=", fontsize=12, color="black", style="italic", weight="light",
    #          horizontalalignment='left', rotation=0)
    # plt.text(262, 110, float('%0.3f' % r2), fontsize=12, color="black", style="italic", weight="light",
    #          horizontalalignment='left', rotation=0)
    # plt.text(238, 95, "p_value=", fontsize=12, color="black", style="italic", weight="light",
    #          horizontalalignment='left', rotation=0)
    # plt.text(290, 95, float(p_value), fontsize=12, color="black", style="italic", weight="light",
    #          horizontalalignment='left', rotation=0)
    # plt.text(175, 20, "Guangzhou", fontsize=10, color="black", weight="heavy",
    #          horizontalalignment='center', rotation=0)
    # beijing
    # plt.text(408, 170, "$R^2$=", fontsize=10, color="black", style="italic", weight="light",
    #          horizontalalignment='left', rotation=0)
    # plt.text(440, 170, float('%0.3f' % r2), fontsize=10, color="black", style="italic", weight="light",
    #          horizontalalignment='left', rotation=0)
    # plt.text(408, 140, "p_value=", fontsize=10, color="black", style="italic", weight="light",
    #          horizontalalignment='left', rotation=0)
    # plt.text(485, 140, float(p_value), fontsize=10, color="black", style="italic", weight="light",
    #          horizontalalignment='left', rotation=0)
    # plt.text(300, 20, "Beijing", fontsize=10, color="black", weight="heavy",
    #          horizontalalignment='center', rotation=0)

    plt.legend(loc='upper left', frameon=False, prop=font1)
    plt.show()


def linearplot(rd, ad):
    slope, intercept, r_value, p_value, std_err = st.linregress(rd, ad)
    r2 = r_value * r_value
    p_value = '{:.2e}'.format(p_value)
    y = ad
    x = rd
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    print(model.summary())
    predicts = model.predict()
    x = rd
    fig = plt.figure(dpi=300, figsize=(6.25, 5))
    plt.scatter(x, y, color='white', edgecolors='black', alpha=0.9, label='Scatter points used in the fitting')
    t = list(range(51000))
    plt.plot(t, t, color='r', label='y = x')
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }
    plt.xlabel(u'Official reported cumulative incidence', font1, verticalalignment='top')
    plt.ylabel(u'Average simulated cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center')
    plt.tick_params(labelsize=10)

    ax = plt.subplot(111)
    ax.set_xlim(4.4*10**1, 5.1*10**4)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=5))
    ax.set_ylim(4.4*10**1, 5.3*10**4)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=5))
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    # wuhan day 9
    ax.text(7000, 100, "$R^2$=", fontsize=10, color="black", style="italic", weight="light",
            horizontalalignment='left', rotation=0)
    ax.text(9950, 100, float('%0.3f' % r2), fontsize=10, color="black", style="italic", weight="light",
            horizontalalignment='left', rotation=0)
    ax.text(7000, 70, "p_value=", fontsize=10, color="black", style="italic", weight="light",
            horizontalalignment='left', rotation=0)
    ax.text(15850, 70, p_value, fontsize=10, color="black", style="italic", weight="light",
            horizontalalignment='left', rotation=0)
    ax.text(7000, 150, "Adjusting days=", fontsize=10, color="black", style="italic", weight="light",
            horizontalalignment='left', rotation=0)
    ax.text(29500, 150, 9, fontsize=11, color="black", style="italic", weight="light",
            horizontalalignment='left', rotation=0)

    # # wuhan day 10
    # ax.text(7000, 100, "$R^2$=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(9950, 100, float('%0.3f' % r2), fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(7000, 70, "p_value=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(15850, 70, p_value, fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(7000, 150, "Adjusting days=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(29500, 150, 10, fontsize=11, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)

    # # wuhan day 11
    # ax.text(7000, 100, "$R^2$=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(9950, 100, float('%0.3f' % r2), fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(7000, 70, "p_value=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(15850, 70, p_value, fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(7000, 150, "Adjusting days=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(29000, 150, 11, fontsize=11, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)

    # # wuhan day 12
    # ax.text(7000, 100, "$R^2$=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(9950, 100, float('%0.3f' % r2), fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(7000, 70, "p_value=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(15850, 70, p_value, fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(7000, 150, "Adjusting days=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(28700, 150, 12, fontsize=11, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)

    # # wuhan day 13
    # ax.text(7000, 100, "$R^2$=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(9950, 100, float('%0.3f' % r2), fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(7000, 70, "p_value=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(15850, 70, p_value, fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(7000, 150, "Adjusting days=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(28700, 150, 13, fontsize=11, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)

    # # wuhan day 14
    # ax.text(7000, 100, "$R^2$=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(9950, 100, float('%0.3f' % r2), fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(7000, 70, "p_value=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(15800, 70, p_value, fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(7000, 150, "Adjusting days=", fontsize=10, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)
    # ax.text(28700, 150, 14, fontsize=11, color="black", style="italic", weight="light",
    #         horizontalalignment='left', rotation=0)

    plt.legend(loc='upper left', frameon=False, prop=font1)
    plt.show()


def risk_index():
    real_data_risk = 'C:/Users/Admin/Desktop/real data/risk score.csv'
    risk_data = pd.read_csv(real_data_risk)
    time = risk_data["Date"][:].tolist()
    time = [datetime.datetime.strptime(i, '%Y/%m/%d') for i in time]

    # risk
    beijing, shanghai, chongqing, wenzhou = risk_data["beijing "][:].tolist(), risk_data["shanghai"][:].tolist(), \
                                            risk_data["chongqing"][:].tolist(), risk_data["wenzhou"][:].tolist()
    guangzhou, tianjin, harbin, changchun = risk_data["guangzhou"][:].tolist(), risk_data["tianjin"][:].tolist(), \
                                            risk_data["harbin"][:].tolist(), risk_data["changchun"][:].tolist()
    shenyang, sjz, xian, zhengzhou = risk_data["shenyang"][:].tolist(), risk_data["shijiazhuang"][:].tolist(), \
                                     risk_data["xi'an"][:].tolist(), risk_data["zhengzhou"][:].tolist()
    jinan, changsha, nanjing, chengdu = risk_data["jinan"][:].tolist(), risk_data["changsha"][:].tolist(), \
                                        risk_data["nanjing"][:].tolist(), risk_data["chengdou"][:].tolist()
    guiyang, hangzhou, nanchang, haikou = risk_data["guiyang"][:].tolist(), risk_data["hangzhou"][:].tolist(), \
                                          risk_data["nanchang"][:].tolist(), risk_data["haikou"][:].tolist()
    lanzhou, xining, taiyuan, hefei = risk_data["lanzhou"][:].tolist(), risk_data["xining"][:].tolist(), \
                                      risk_data["taiyuan"][:].tolist(), risk_data["hefei"][:].tolist()
    kunming, fuzhou, shenzhen, suizhou = risk_data["kunming"][:].tolist(), risk_data["fuzhou"][:].tolist(), \
                                         risk_data["shenzhen"][:].tolist(), risk_data["suizhou"][:].tolist()
    o_risk = [beijing, shanghai, chongqing, wenzhou, guangzhou, tianjin, harbin, changchun, shenyang, sjz, xian,
              zhengzhou, jinan, changsha, nanjing, chengdu, guiyang, hangzhou, nanchang, haikou, lanzhou, xining,
              taiyuan, hefei, kunming, fuzhou, shenzhen, suizhou]
    label_name_risk = ['Beijing', 'Shanghai', 'Chongqing', 'Wenzhou', 'Guangzhou', 'Tianjin', 'Harbin', 'Changchun',
                       'Shenyang', 'Shijiazhuang', 'Xian', 'Zhengzhou', 'Jinan', 'Changsha', 'Nanjing', 'Chengdu',
                       'Guiyang',
                       'Hangzhou', 'Nanchang', 'Haikou', 'Lanzhou', 'Xining', 'Taiyuan', 'Hefei', 'Kunming', 'Fuzhou',
                       'Shenzhen', 'Suizhou']


    fig = plt.figure(dpi=300, figsize=(5, 4))
    plt.ylim(-10, 12)
    ax = plt.subplot(111)

    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }

    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 6,
             }

    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    ax.set_xticks(pd.date_range('2020-01-27', '2020-02-20', freq='8D'))
    ax.set_xlabel(u'Date/day', font1, verticalalignment='top')
    ax.set_ylabel(u'Risk score', font1, verticalalignment='baseline', horizontalalignment='center')
    plt.title('Risk score in main cities of China', font1)

    colors = list(mcolors.CSS4_COLORS.keys())
    colors2 = ['lightcoral', 'rosybrown', 'firebrick', 'brown', 'mistyrose', 'salmon', 'tomato', 'darksalmon', 'coral',
               'lightsalmon', 'sienna', 'seashell', 'chocolate', 'saddlebrown', 'sandybrown', 'peachpuff', 'peru',
               'linen', 'bisque', 'darkorange', 'burlywood', 'tan', 'orange', 'wheat', 'gold', 'khaki', 'darkkhaki', 'y']
    ls = []
    style = ['-', '--', '-.', ':']
    ss = ['o', '+', 's', '<', 'p', '*', 'd']
    for i in range(len(ss)):
        for j in range(len(style)):
            ls0 = ss[i] + style[j]
            ls.append(ls0)

    # mcolors.CSS4_COLORS[colors[i]] colors2[i]
    for i in range(len(o_risk)):
        ax.plot(time[0:len(o_risk[i])], o_risk[i], ls[i], color=mcolors.CSS4_COLORS[colors[i]],
                linewidth=1, label=str(label_name_risk[i]), markersize=1.5)

    ax.text(time[3], o_risk[0][3] + 0.3, str(label_name_risk[0]), fontsize=8, color="black",
             fontname="Arial")
    ax.text(time[19], o_risk[3][19] + 0.3, str(label_name_risk[3]), fontsize=8, color="black",
             fontname="Arial")
    ax.text(time[9], o_risk[4][9] + 0.3, str(label_name_risk[4]), fontsize=8, color="black",
             fontname="Arial")
    ax.text(time[5], o_risk[26][5] + 0.8, str(label_name_risk[26]), fontsize=8, color="black",
             fontname="Arial")
    ax.text(time[19], o_risk[27][19] + 0.3, str(label_name_risk[27]), fontsize=8, color="black",
             fontname="Arial")
    ax.text(time[19], o_risk[21][19] + 0.3, str(label_name_risk[21]), fontsize=8, color="black",
             fontname="Arial")

    ax.axhline(y=0, ls="-", c="black", linewidth=1)
    ax.hlines(1.645, ['2020-02-07'], ['2020-02-11'], linestyles='-', linewidth=2, color='red')
    ax.hlines(1.960, ['2020-02-11'], ['2020-02-15'], linestyles='-', linewidth=2, color='r')
    ax.hlines(2.576, ['2020-02-15'], ['2020-02-19'], linestyles='-', linewidth=2, color='darkred')

    ax.annotate('90% CI, 3rd-level risk=1.645', xy=(datetime.datetime.strptime(('2020/02/07'), '%Y/%m/%d'), 1.645),
                xytext=(datetime.datetime.strptime(('2020/01/29'), '%Y/%m/%d'), 2.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=2.5,
                                headlength=1), fontname="Arial", fontsize=6)
    ax.annotate('95% CI, 2nd-level risk=1.960', xy=(datetime.datetime.strptime(('2020/02/11'), '%Y/%m/%d'), 1.96),
                xytext=(datetime.datetime.strptime(('2020/02/02'), '%Y/%m/%d'), 3.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=2.5,
                                headlength=1), fontname="Arial", fontsize=6)
    ax.annotate('99% CI, 1st-level risk=2.576', xy=(datetime.datetime.strptime(('2020/02/15'), '%Y/%m/%d'), 2.576),
                xytext=(datetime.datetime.strptime(('2020/02/06'), '%Y/%m/%d'), 4.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=2.5,
                                headlength=1), fontname="Arial", fontsize=6)

    ax.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0, frameon=False, prop=font2, labelspacing=0.1,
               markerscale=0.5, ncol=1)
    # bbox_to_anchor=(1.05, 1), loc=3,
    plt.show('tight')


def risk_index2():
    real_data_risk = 'C:/Users/Admin/Desktop/real data/risk score.csv'
    risk_data = pd.read_csv(real_data_risk)
    time = risk_data["Date"][:].tolist()
    time = [datetime.datetime.strptime(i, '%Y/%m/%d') for i in time]

    # risk
    beijing, shanghai, chongqing, wenzhou = risk_data["beijing "][:].tolist(), risk_data["shanghai"][:].tolist(), \
                                            risk_data["chongqing"][:].tolist(), risk_data["wenzhou"][:].tolist()
    guangzhou, tianjin, harbin = risk_data["guangzhou"][:].tolist(), risk_data["tianjin"][:].tolist(), \
                                            risk_data["harbin"][:].tolist()
    xian, zhengzhou = risk_data["xi'an"][:].tolist(), risk_data["zhengzhou"][:].tolist()
    changsha, nanjing, chengdu = risk_data["changsha"][:].tolist(), \
                                        risk_data["nanjing"][:].tolist(), risk_data["chengdou"][:].tolist()
    guiyang, hangzhou = risk_data["guiyang"][:].tolist(), risk_data["hangzhou"][:].tolist(),

    taiyuan, hefei = risk_data["taiyuan"][:].tolist(), risk_data["hefei"][:].tolist()
    shenzhen, suizhou = risk_data["shenzhen"][:].tolist(), risk_data["suizhou"][:].tolist()
    o_risk = [beijing, shanghai, chongqing, wenzhou, guangzhou, tianjin, harbin, xian,
              zhengzhou, changsha, nanjing, chengdu, guiyang, hangzhou, taiyuan, hefei, shenzhen, suizhou]
    label_name_risk = ['Beijing', 'Shanghai', 'Chongqing', 'Wenzhou', 'Guangzhou', 'Tianjin', 'Harbin',
                    'Xian', 'Zhengzhou', 'Changsha', 'Nanjing', 'Chengdu', 'Guiyang', 'Hangzhou',
                       'Taiyuan', 'Hefei',  'Shenzhen', 'Suizhou']


    fig = plt.figure(dpi=300, figsize=(5, 4))
    plt.ylim(-10, 12.5)
    ax = plt.subplot(111)

    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }

    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 6,
             }

    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    ax.set_xticks(pd.date_range('2020-01-27', '2020-02-20', freq='8D'))
    ax.set_xlabel(u'Date/day', font1, verticalalignment='top')
    ax.set_ylabel(u'Risk score', font1, verticalalignment='baseline', horizontalalignment='center')
    plt.title('Risk score in main cities of China', font1)

    colors = list(mcolors.CSS4_COLORS.keys())
    colors2 = ['lightcoral', 'rosybrown', 'firebrick', 'brown', 'mistyrose', 'salmon', 'tomato', 'darksalmon', 'coral',
               'lightsalmon', 'sienna', 'seashell', 'chocolate', 'saddlebrown', 'sandybrown', 'peachpuff', 'peru',
               'linen', 'bisque', 'darkorange', 'burlywood', 'tan', 'orange', 'wheat', 'gold', 'khaki', 'darkkhaki', 'y']
    ls = []
    style = ['-', '--', '-.', ':']
    ss = ['o', '+', 's', '<', 'p', '*', 'd']
    for i in range(len(ss)):
        for j in range(len(style)):
            ls0 = ss[i] + style[j]
            ls.append(ls0)

    # mcolors.CSS4_COLORS[colors[i]] colors2[i]
    for i in range(len(o_risk)):
        ax.plot(time[0:len(o_risk[i])], o_risk[i], ls[i], color=mcolors.CSS4_COLORS[colors[i]],
                linewidth=1, label=str(label_name_risk[i]), markersize=1.5)

    ax.text(time[3], o_risk[0][3] + 0.3, str(label_name_risk[0]), fontsize=8, color="black",
             fontname="Arial")
    ax.text(time[19], o_risk[3][19] + 0.3, str(label_name_risk[3]), fontsize=8, color="black",
             fontname="Arial")
    ax.text(time[9], o_risk[4][9] + 0.3, str(label_name_risk[4]), fontsize=8, color="black",
             fontname="Arial")
    ax.text(time[5], o_risk[16][5] + 0.8, str(label_name_risk[16]), fontsize=8, color="black",
             fontname="Arial")
    ax.text(time[19], o_risk[17][19] + 0.3, str(label_name_risk[17]), fontsize=8, color="black",
             fontname="Arial")


    ax.axhline(y=0, ls="-", c="black", linewidth=1)
    ax.hlines(1.645, ['2020-02-07'], ['2020-02-11'], linestyles='-', linewidth=2, color='red')
    ax.hlines(1.960, ['2020-02-11'], ['2020-02-15'], linestyles='-', linewidth=2, color='r')
    ax.hlines(2.576, ['2020-02-15'], ['2020-02-19'], linestyles='-', linewidth=2, color='darkred')

    ax.annotate('90% CI, 3rd-level risk=1.645', xy=(datetime.datetime.strptime(('2020/02/07'), '%Y/%m/%d'), 1.645),
                xytext=(datetime.datetime.strptime(('2020/01/29'), '%Y/%m/%d'), 2.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=2.5,
                                headlength=1), fontname="Arial", fontsize=6)
    ax.annotate('95% CI, 2nd-level risk=1.960', xy=(datetime.datetime.strptime(('2020/02/11'), '%Y/%m/%d'), 1.96),
                xytext=(datetime.datetime.strptime(('2020/02/02'), '%Y/%m/%d'), 3.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=2.5,
                                headlength=1), fontname="Arial", fontsize=6)
    ax.annotate('99% CI, 1st-level risk=2.576', xy=(datetime.datetime.strptime(('2020/02/15'), '%Y/%m/%d'), 2.576),
                xytext=(datetime.datetime.strptime(('2020/02/06'), '%Y/%m/%d'), 4.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=2.5,
                                headlength=1), fontname="Arial", fontsize=6)

    ax.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0, frameon=False, prop=font2, labelspacing=0.1,
    #            markerscale=0.5, ncol=1)
    # bbox_to_anchor=(1.05, 1), loc=3,
    plt.show('tight')


def outflow2():
    real_data_fp = 'C:/Users/Admin/Desktop/real data/outbound flow.csv'

    flow_data = pd.read_csv(real_data_fp)

    time = flow_data["Date"][:].tolist()
    time = [datetime.datetime.strptime(i, '%Y/%m/%d') for i in time]

    # flow
    beijing, shanghai, chongqing, wenzhou = flow_data["beijing "][:].tolist(), flow_data["shanghai"][:].tolist(), \
                                            flow_data["chongqing"][:].tolist(), flow_data["wenzhou"][:].tolist()
    guangzhou, tianjin, harbin = flow_data["guangzhou"][:].tolist(), flow_data["tianjin"][:].tolist(), \
                                            flow_data["harbin"][:].tolist()
    xian, zhengzhou = flow_data["xi'an"][:].tolist(), flow_data["zhengzhou"][:].tolist()
    changsha, nanjing, chengdu = flow_data["changsha"][:].tolist(), \
                                        flow_data["nanjing"][:].tolist(), flow_data["chengdou"][:].tolist()
    guiyang, hangzhou = flow_data["guiyang"][:].tolist(), flow_data["hangzhou"][:].tolist()
    taiyuan, hefei = flow_data["taiyuan"][:].tolist(), flow_data["hefei"][:].tolist()
    shenzhen, suizhou = flow_data["shenzhen"][:].tolist(), flow_data["suizhou"][:].tolist()
    o_flow = [beijing, shanghai, chongqing, wenzhou, guangzhou, tianjin, harbin, xian,
              zhengzhou, changsha, nanjing, chengdu, guiyang, hangzhou, taiyuan, hefei, shenzhen, suizhou]
    label_name = ['Beijing', 'Shanghai', 'Chongqing', 'Wenzhou', 'Guangzhou', 'Tianjin', 'Harbin',
                    'Xian', 'Zhengzhou', 'Changsha', 'Nanjing', 'Chengdu', 'Guiyang', 'Hangzhou',
                       'Taiyuan', 'Hefei',  'Shenzhen', 'Suizhou']

    fig = plt.figure(dpi=300, figsize=(5, 4))
    plt.ylim(0, 40000)
    ax = plt.subplot(111)
    y_major_locator = MultipleLocator(10000)
    ax.yaxis.set_major_locator(y_major_locator)

    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }

    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 6,
             }

    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    ax.set_xticks(pd.date_range('2020-01-01', '2020-01-31', freq='10D'))


    ax.set_xlabel(u'Date/day', font1, verticalalignment='top')
    ax.set_ylabel(u'Number of outbound flows', font1, verticalalignment='baseline', horizontalalignment='center')
    plt.title('Outbound flows from wuhan to main cities of China', font1)

    colors = list(mcolors.CSS4_COLORS.keys())
    colors2 = ['lightcoral', 'rosybrown', 'firebrick', 'brown', 'mistyrose', 'salmon', 'tomato', 'darksalmon', 'coral',
               'lightsalmon', 'sienna', 'seashell', 'chocolate', 'saddlebrown', 'sandybrown', 'peachpuff', 'peru',
               'linen', 'bisque', 'darkorange', 'burlywood', 'tan', 'orange', 'wheat', 'gold', 'khaki', 'darkkhaki',
               'y']
    ls = []
    style = ['-', '--', '-.', ':']
    ss = ['o', '+', 's', '<', 'p', '*', 'd']
    for i in range(len(ss)):
        for j in range(len(style)):
            ls0 = ss[i] + style[j]
            ls.append(ls0)

    for i in range(len(o_flow)):
        ax.plot(time[0:len(o_flow[i])], o_flow[i], ls[i], color=mcolors.CSS4_COLORS[colors[i]], linewidth=1.0, label=str(label_name[i]), markersize=1.5)

    ax.vlines(['2020-01-23'], 0, 18000, linestyles='dashed', colors='red', alpha=0.7, linewidth=1)
    ax.annotate('The date of quarantine: 23 Jan', xy=(datetime.datetime.strptime(('2020/01/23'), '%Y/%m/%d'), 18000),
                xytext=(datetime.datetime.strptime(('2020/01/25'), '%Y/%m/%d'), 18050),
                arrowprops=dict(facecolor='red', color='red', shrink=0.05, width=1, headwidth=2.5,
                                headlength=1), fontname="Arial", fontsize=6, c='red')
    ax.tick_params(labelsize=10)
    labels1 = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels1]


    ax.legend(loc='best', frameon=False, prop=font2, labelspacing=0.2, markerscale=0.5, ncol=2)
    # plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0, frameon=False, prop=font2, labelspacing=0.1,
    #            markerscale=0.5, ncol=1)
    # ax.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0, frameon=False, prop=font2, labelspacing=0.2, markerscale=1, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def outflow():
    real_data_fp = 'C:/Users/Admin/Desktop/real data/outbound flow.csv'

    flow_data = pd.read_csv(real_data_fp)

    time = flow_data["Date"][:].tolist()
    time = [datetime.datetime.strptime(i, '%Y/%m/%d') for i in time]

    # flow
    beijing, shanghai, chongqing, wenzhou = flow_data["beijing "][:].tolist(), flow_data["shanghai"][:].tolist(), \
                                            flow_data["chongqing"][:].tolist(), flow_data["wenzhou"][:].tolist()
    guangzhou, tianjin, harbin, changchun = flow_data["guangzhou"][:].tolist(), flow_data["tianjin"][:].tolist(), \
                                            flow_data["harbin"][:].tolist(), flow_data["changchun"][:].tolist()
    shenyang, sjz, xian, zhengzhou = flow_data["shenyang"][:].tolist(), flow_data["shijiazhuang"][:].tolist(), \
                                     flow_data["xi'an"][:].tolist(), flow_data["zhengzhou"][:].tolist()
    jinan, changsha, nanjing, chengdu = flow_data["jinan"][:].tolist(), flow_data["changsha"][:].tolist(), \
                                        flow_data["nanjing"][:].tolist(), flow_data["chengdou"][:].tolist()
    guiyang, hangzhou, nanchang, haikou = flow_data["guiyang"][:].tolist(), flow_data["hangzhou"][:].tolist(), \
                                          flow_data["nanchang"][:].tolist(), flow_data["haikou"][:].tolist()
    lanzhou, xining, taiyuan, hefei = flow_data["lanzhou"][:].tolist(), flow_data["xining"][:].tolist(), \
                                      flow_data["taiyuan"][:].tolist(), flow_data["hefei"][:].tolist()
    kunming, fuzhou, shenzhen, suizhou = flow_data["kunming"][:].tolist(), flow_data["fuzhou"][:].tolist(), \
                                         flow_data["shenzhen"][:].tolist(), flow_data["suizhou"][:].tolist()
    o_flow = [beijing, shanghai, chongqing, wenzhou, guangzhou, tianjin, harbin, changchun, shenyang, sjz, xian,
              zhengzhou, jinan, changsha, nanjing, chengdu, guiyang, hangzhou, nanchang, haikou, lanzhou, xining,
              taiyuan, hefei, kunming, fuzhou, shenzhen, suizhou]
    label_name = ['Beijing', 'Shanghai', 'Chongqing', 'Wenzhou', 'Guangzhou', 'Tianjin', 'Harbin', 'Changchun',
                  'Shenyang', 'Shijiazhuang', 'Xian', 'Zhengzhou', 'Jinan', 'Changsha', 'Nanjing', 'Chengdu', 'Guiyang',
                  'Hangzhou', 'Nanchang', 'Haikou', 'Lanzhou', 'Xining', 'Taiyuan', 'Hefei', 'Kunming', 'Fuzhou',
                  'Shenzhen', 'Suizhou']

    fig = plt.figure(dpi=300, figsize=(5, 4))
    plt.ylim(0, 40000)
    ax = plt.subplot(111)
    y_major_locator = MultipleLocator(10000)
    ax.yaxis.set_major_locator(y_major_locator)

    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }

    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 6,
             }

    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    ax.set_xticks(pd.date_range('2020-01-01', '2020-01-31', freq='10D'))


    ax.set_xlabel(u'Date/day', font1, verticalalignment='top')
    ax.set_ylabel(u'Number of outbound flows', font1, verticalalignment='baseline', horizontalalignment='center')
    plt.title('Outbound flows from wuhan to main cities of China', font1)

    colors = list(mcolors.CSS4_COLORS.keys())
    colors2 = ['lightcoral', 'rosybrown', 'firebrick', 'brown', 'mistyrose', 'salmon', 'tomato', 'darksalmon', 'coral',
               'lightsalmon', 'sienna', 'seashell', 'chocolate', 'saddlebrown', 'sandybrown', 'peachpuff', 'peru',
               'linen', 'bisque', 'darkorange', 'burlywood', 'tan', 'orange', 'wheat', 'gold', 'khaki', 'darkkhaki',
               'y']
    ls = []
    style = ['-', '--', '-.', ':']
    ss = ['o', '+', 's', '<', 'p', '*', 'd']
    for i in range(len(ss)):
        for j in range(len(style)):
            ls0 = ss[i] + style[j]
            ls.append(ls0)

    for i in range(len(o_flow)):
        ax.plot(time[0:len(o_flow[i])], o_flow[i], ls[i], color=mcolors.CSS4_COLORS[colors[i]], linewidth=1.0, label=str(label_name[i]), markersize=1.5)

    ax.vlines(['2020-01-23'], 0, 18000, linestyles='dashed', colors='red', alpha=0.7, linewidth=1)
    ax.annotate('The date of quarantine: 23 Jan', xy=(datetime.datetime.strptime(('2020/01/23'), '%Y/%m/%d'), 18000),
                xytext=(datetime.datetime.strptime(('2020/01/25'), '%Y/%m/%d'), 18050),
                arrowprops=dict(facecolor='red', color='red', shrink=0.05, width=1, headwidth=2.5,
                                headlength=1), fontname="Arial", fontsize=6, c='red')
    ax.tick_params(labelsize=10)
    labels1 = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels1]


    ax.legend(loc='best', frameon=False, prop=font2, labelspacing=0.2, markerscale=0.5, ncol=2)
    # plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0, frameon=False, prop=font2, labelspacing=0.1,
    #            markerscale=0.5, ncol=1)
    # ax.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0, frameon=False, prop=font2, labelspacing=0.2, markerscale=1, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def sy4plotresult_wz(file_path, time, real_covid_data):
    path_list = []
    for path_t in os.listdir(file_path):
        path_list.append(file_path + path_t)
    str_result_data = []
    for i in range(len(path_list)):
        s_covid = load_csv(path_list[i])
        str_i = np.array(s_covid)
        str_result_data.append(str_i)
    out_data = np.array(str_result_data)
    ii, jj, kk = out_data.shape

    s_covid_tc1_a, s_covid_tc1_u, s_covid_tc1_d = [], [], []
    s_covid_tc2_a, s_covid_tc2_u, s_covid_tc2_d = [], [], []
    s_covid_tq1_a, s_covid_tq1_u, s_covid_tq1_d = [], [], []
    s_covid_tq2_a, s_covid_tq2_u, s_covid_tq2_d = [], [], []
    s_covid_tq3_a, s_covid_tq3_u, s_covid_tq3_d = [], [], []
    s_covid_tq4_a, s_covid_tq4_u, s_covid_tq4_d = [], [], []
    s_covid_tq5_a, s_covid_tq5_u, s_covid_tq5_d = [], [], []
    for j in range(jj-19):
        s_covid_tc1_a.append(float(out_data[0][j][0]))
        s_covid_tc1_u.append(float(out_data[0][j][1]))
        s_covid_tc1_d.append(float(out_data[0][j][2]))
        s_covid_tc2_a.append(float(out_data[1][j][0]))
        s_covid_tc2_u.append(float(out_data[1][j][1]))
        s_covid_tc2_d.append(float(out_data[1][j][2]))
        s_covid_tq1_a.append(float(out_data[2][j][0]))
        s_covid_tq1_u.append(float(out_data[2][j][1]))
        s_covid_tq1_d.append(float(out_data[2][j][2]))
        s_covid_tq2_a.append(float(out_data[3][j][0]))
        s_covid_tq2_u.append(float(out_data[3][j][1]))
        s_covid_tq2_d.append(float(out_data[3][j][2]))
        s_covid_tq3_a.append(float(out_data[4][j][0]))
        s_covid_tq3_u.append(float(out_data[4][j][1]))
        s_covid_tq3_d.append(float(out_data[4][j][2]))
        s_covid_tq4_a.append(float(out_data[5][j][0]))
        s_covid_tq4_u.append(float(out_data[5][j][1]))
        s_covid_tq4_d.append(float(out_data[5][j][2]))
        s_covid_tq5_a.append(float(out_data[6][j][0]))
        s_covid_tq5_u.append(float(out_data[6][j][1]))
        s_covid_tq5_d.append(float(out_data[6][j][2]))

    rd_s_file_path = 'C:/Users/Admin/Desktop/实验四/wenzhou/result-wenzhou-final.csv'
    df2 = pd.read_csv(rd_s_file_path)
    u_s = [float(i) for i in df2.iloc[0:len(df2), 1]]
    d_s = [float(i) for i in df2.iloc[0:len(df2), 2]]

    # plot
    fig = plt.figure(dpi=300, figsize=(8, 5))
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }

    ax = plt.Axes(fig, [.10, .10, .45, 0.8])
    ax.set_title('Wenzhou', font1)
    ax.set_xlabel(u'Date/day', font1, verticalalignment='top')
    ax.set_ylabel(u'Cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center')
    ax.set_ylim(0, 1800)
    ax1 = plt.Axes(fig, [.62, .10, .30, 0.36])
    fig.add_axes(ax)
    fig.add_axes(ax1)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    y_major_locator = MultipleLocator(300)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xticks(pd.date_range('2020-01-21', '2020-03-31', freq='15D'))


    ax.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax.plot(time[0:71], real_covid_data[0:71], '-.', color='teal', linewidth=2.5, label='Official reported cumulative incidence',
            markersize=2)
    ax.fill_between(time, d_s, u_s, color='skyblue', alpha=0.8)

    ax.plot(time, s_covid_tc1_a, 'o-', color='crimson', linewidth=1, label='Lockdown 1 day delay',
            markersize=2)
    ax.fill_between(time, s_covid_tc1_d, s_covid_tc1_u, color='salmon', alpha=0.8)
    ax.plot(time, s_covid_tc2_a, 'p-', color='crimson', linewidth=1, label='Lockdown 2 day delay',
            markersize=2)
    ax.fill_between(time, s_covid_tc2_d, s_covid_tc2_u, color='salmon', alpha=0.8)
    ax.plot(time, s_covid_tq1_a, '+-', color='forestgreen', linewidth=1, label='Lockdown 1 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq1_d, s_covid_tq1_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq2_a, 's-', color='forestgreen', linewidth=1, label='Lockdown 2 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq2_d, s_covid_tq2_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq3_a, 'H-', color='forestgreen', linewidth=1, label='Lockdown 3 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq3_d, s_covid_tq3_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq4_a, 'x-', color='forestgreen', linewidth=1, label='Lockdown 4 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq4_d, s_covid_tq4_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq5_a, '2-', color='forestgreen', linewidth=1, label='Lockdown 5 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq5_d, s_covid_tq5_u, color='lightgreen', alpha=0.8)

    ax.vlines(['2020-02-01'], 0, 591.46, linestyles='dashed', colors='black')

    ax.text(datetime.datetime.strptime(('2020/01/18'), '%Y/%m/%d'), 650, "lockdown measure \n in advance \n or delay",
            fontsize=7, color="black", style="italic", weight="light", horizontalalignment='left', rotation=0, fontname="Arial")
    ax.annotate('', xy=(datetime.datetime.strptime(('2020/02/01'), '%Y/%m/%d'), 0),
                xytext=(datetime.datetime.strptime(('2020/01/25'), '%Y/%m/%d'), 640),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4))
    ax.annotate('in advance', xy=(datetime.datetime.strptime(('2020/02/13'), '%Y/%m/%d'), 330),
                xytext=(datetime.datetime.strptime(('2020/02/13'), '%Y/%m/%d'), 430),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=4,
                                headlength=4), fontname="Arial", fontsize=7)
    ax.annotate('delay', xy=(datetime.datetime.strptime(('2020/02/13'), '%Y/%m/%d'), 660),
                xytext=(datetime.datetime.strptime(('2020/02/13'), '%Y/%m/%d'), 540),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=4,
                                headlength=4), fontname="Arial", fontsize=7)
    ax.annotate('real data', xy=(datetime.datetime.strptime(('2020/03/05'), '%Y/%m/%d'), 515),
                xytext=(datetime.datetime.strptime(('2020/03/05'), '%Y/%m/%d'), 625),
                arrowprops=dict(color='r', shrink=0.05, width=2, headwidth=4,
                                headlength=4), color="red", fontname="Arial", fontsize=7)
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 7,
             }
    ax.legend(loc='upper left', frameon=False, prop=font2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fontt = {'family': 'Arial',
             'weight': 'normal',
             'size': 9,
             }

    tq1_r = np.diff(s_covid_tq1_a)
    tq2_r = np.diff(s_covid_tq2_a)
    tq3_r = np.diff(s_covid_tq3_a)
    tq4_r = np.diff(s_covid_tq4_a)
    tq5_r = np.diff(s_covid_tq5_a)
    time2 = time[1:len(time)]
    ax1.set_ylim(0, 40)
    ax1.set_xlabel(u'Date/day', fontt, verticalalignment='top', labelpad=5)
    ax1.set_ylabel(u'Incidence', fontt, verticalalignment='baseline', horizontalalignment='center', labelpad=14)
    ax1.plot(time2, tq1_r, '+-', color='forestgreen', linewidth=1, label='Lockdown 1 day earlier', markersize=1.5)
    ax1.plot(time2, tq2_r, 's-', color='forestgreen', linewidth=1, label='Lockdown 2 day earlier', markersize=1.5)
    ax1.plot(time2, tq3_r, 'H-', color='forestgreen', linewidth=1, label='Lockdown 3 day earlier', markersize=1.5)
    ax1.plot(time2, tq4_r, 'x-', color='forestgreen', linewidth=1, label='Lockdown 4 day earlier', markersize=1.5)
    ax1.plot(time2, tq5_r, '2-', color='forestgreen', linewidth=1, label='Lockdown 5 day earlier', markersize=1.5)
    ax1.annotate('Local transmission is almost contained \n when lockdown is adopted once \n risk index prompts high-risk',
                 xy=(datetime.datetime.strptime(('2020/01/25'), '%Y/%m/%d'), 4),
                xytext=(datetime.datetime.strptime(('2020/02/18'), '%Y/%m/%d'), 15),
                arrowprops=dict(color='r', shrink=0.05, width=1, headwidth=2,
                                headlength=2), color="red", fontname="Arial", fontsize=6)

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax1.set_xticks(pd.date_range('2020-01-22', '2020-03-31', freq='12D'))
    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 6,
             }
    ax1.legend(loc='upper right', frameon=False, prop=font3)
    ax1.tick_params(labelsize=7)
    labels2 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot barchart-wenzhou
    ax2 = plt.Axes(fig, [.62, .55, .29, 0.36])
    fig.add_axes(ax2)
    x_pos = np.arange(7)
    relative_loss = [-0.915, -1.949, -2.159, -2.434, -2.709, 0.284, 0.705]
    relative_loss = np.array(relative_loss)
    md_loss = [-3.47, -5.16, -7.16, -8.19, -8.84, 7.51, 20.49]
    rl_err = [0.342, 0.576, 0.753, 0.863, 0.913, 0.105, 0.283]
    md_err = [0.904, 0.841, 0.839, 0.967, 0.831, 1.2, 1.187]
    gdp_err = 1.185 / 6.391

    ax2.axhline(y=0, ls="-.", c="black", linewidth=1)
    ax2.bar(x_pos, relative_loss, yerr=rl_err, width=0.3, align='center', color='deepskyblue',
           label='Relative GDP loss variation')
    bar_width = 0.3
    ax2.bar(x_pos + bar_width, md_loss, yerr=md_err, width=bar_width, align='center', color='mediumspringgreen',
           label='Relative medical expenditure variation')
    ax2.set_xticks([])
    ax2.set_ylim(-10, 22)
    ax2.set_xlabel('Adjusting the timing of lockdown measures', fontname="Arial", fontsize=9, labelpad=5)
    ax2.set_ylabel('Relative economic variation due to \n adjustment of the timing of lockdown', fontname="Arial",
                  fontsize=9, labelpad=2)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax2.set_xlim(min(x_pos) - 1, max(x_pos) + 4)
    ax2.tick_params(labelsize=7)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax2.axvspan(xmin=-1, xmax=4.5, facecolor='bisque', alpha=0.5)
    ax2.text(0.1, 0.5, "1 day \n earlier", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(1.1, 0.5, "2 day \n earlier", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(2.1, 0.5, "3 day \n earlier", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(3.1, 0.5, "4 day \n earlier", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(4.1, 0.5, "5 day \n earlier", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(1.2, 11, "Earlier-than-\n actual control", fontsize=7, color="black", weight="heavy", style="italic",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.axvspan(4.5, 7.5, facecolor='aquamarine', alpha=0.35)
    ax2.text(5.1, -2.9, "1 day \n delay", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(6.1, -2.9, "2 day \n delay", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(6.0, -7.5, "Later-than-\n actual control", fontsize=7, color="black", weight="heavy", style="italic",
            horizontalalignment='center', rotation=0, fontname="Arial")
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 6,
             }
    ax2.legend(loc='upper left', frameon=False, prop=font2)

    ax3 = ax2.twinx()
    ax3.set_ylim(-6.8 / 6.391, 15 / 6.391)
    ax3.set_yticks([0, 15 / 6.391])
    ax3.bar(8, 12.601 / 6.391, width=0.8, align='center', color='pink')
    ax3.bar(9, 12.858 / 6.391, yerr=gdp_err, width=0.8, align='center', color='pink')
    ax3.set_ylabel('Estimated GDP loss values \n (billion USD $)', fontname="Arial",
                   fontsize=9)
    ax3.text(8, 1 / 6.391, "Direct GDP \n loss", fontsize=6, color="black", weight="light",
             horizontalalignment='center', rotation=90, fontname="Arial")
    ax3.text(9, 1 / 6.391, "Our model  \n prediction loss", fontsize=6, color="black", weight="light",
             horizontalalignment='center', rotation=90, fontname="Arial")
    ax3.text(8.5, -2.4 / 6.391, "Estimated \n GDP loss", fontsize=6, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax3.tick_params(labelsize=7)
    labels2 = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]
    ax3.text(9, -5.8 / 6.391, "Wenzhou", fontsize=6, color="black", weight="heavy",
             horizontalalignment='center', rotation=0, fontname="Arial")
    plt.show('Tight')


def sy4plotresult_gz(file_path, time, real_covid_data):
    path_list = []
    for path_t in os.listdir(file_path):
        path_list.append(file_path + path_t)
    str_result_data = []
    for i in range(len(path_list)):
        s_covid = load_csv(path_list[i])
        str_i = np.array(s_covid)
        str_result_data.append(str_i)
    out_data = np.array(str_result_data)
    ii, jj, kk = out_data.shape
    s_covid_tc1_a, s_covid_tc1_u, s_covid_tc1_d = [], [], []
    s_covid_tc2_a, s_covid_tc2_u, s_covid_tc2_d = [], [], []
    s_covid_tq1_a, s_covid_tq1_u, s_covid_tq1_d = [], [], []
    s_covid_tq2_a, s_covid_tq2_u, s_covid_tq2_d = [], [], []
    s_covid_tq3_a, s_covid_tq3_u, s_covid_tq3_d = [], [], []
    s_covid_tq4_a, s_covid_tq4_u, s_covid_tq4_d = [], [], []
    s_covid_tq5_a, s_covid_tq5_u, s_covid_tq5_d = [], [], []
    for j in range(jj-20):
        s_covid_tc1_a.append(float(out_data[0][j][0]))
        s_covid_tc1_u.append(float(out_data[0][j][1]))
        s_covid_tc1_d.append(float(out_data[0][j][2]))
        s_covid_tc2_a.append(float(out_data[1][j][0]))
        s_covid_tc2_u.append(float(out_data[1][j][1]))
        s_covid_tc2_d.append(float(out_data[1][j][2]))
        s_covid_tq1_a.append(float(out_data[2][j][0]))
        s_covid_tq1_u.append(float(out_data[2][j][1]))
        s_covid_tq1_d.append(float(out_data[2][j][2]))
        s_covid_tq2_a.append(float(out_data[3][j][0]))
        s_covid_tq2_u.append(float(out_data[3][j][1]))
        s_covid_tq2_d.append(float(out_data[3][j][2]))
        s_covid_tq3_a.append(float(out_data[4][j][0]))
        s_covid_tq3_u.append(float(out_data[4][j][1]))
        s_covid_tq3_d.append(float(out_data[4][j][2]))
        s_covid_tq4_a.append(float(out_data[5][j][0]))
        s_covid_tq4_u.append(float(out_data[5][j][1]))
        s_covid_tq4_d.append(float(out_data[5][j][2]))
        s_covid_tq5_a.append(float(out_data[6][j][0]))
        s_covid_tq5_u.append(float(out_data[6][j][1]))
        s_covid_tq5_d.append(float(out_data[6][j][2]))

    # plot
    fig = plt.figure(dpi=300, figsize=(8, 4.5))
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 14,
             }

    ax = plt.Axes(fig, [.08, .12, .6, 0.8])
    ax.set_title('Guangzhou', font1)
    ax.set_xlabel(u'Date/day', font1, verticalalignment='top')
    ax.set_ylabel(u'Cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center')
    ax.set_ylim(0, 750)
    ax1 = plt.Axes(fig, [.70, .25, .25, 0.4])
    fig.add_axes(ax)
    fig.add_axes(ax1)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    y_major_locator = MultipleLocator(250)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xticks(pd.date_range('2020-01-22', '2020-03-31', freq='15D'))


    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax.plot(time[0:70], real_covid_data[0:70], '-.', color='teal', linewidth=2.5, label='Official reported cumulative incidence',
            markersize=2)
    ax.plot(time, s_covid_tc1_a, 'o-', color='crimson', linewidth=1, label='Lockdown 1 day delay',
            markersize=2)
    ax.fill_between(time, s_covid_tc1_d, s_covid_tc1_u, color='salmon', alpha=0.8)
    ax.plot(time, s_covid_tc2_a, 'p-', color='crimson', linewidth=1, label='Lockdown 2 day delay',
            markersize=2)
    ax.fill_between(time, s_covid_tc2_d, s_covid_tc2_u, color='salmon', alpha=0.8)
    ax.plot(time, s_covid_tq1_a, '+-', color='forestgreen', linewidth=1, label='Lockdown 1 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq1_d, s_covid_tq1_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq2_a, 's-', color='forestgreen', linewidth=1, label='Lockdown 2 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq2_d, s_covid_tq2_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq3_a, 'H-', color='forestgreen', linewidth=1, label='Lockdown 3 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq3_d, s_covid_tq3_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq4_a, 'x-', color='forestgreen', linewidth=1, label='Lockdown 4 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq4_d, s_covid_tq4_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq5_a, '2-', color='forestgreen', linewidth=1, label='Lockdown 5 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq5_d, s_covid_tq5_u, color='lightgreen', alpha=0.8)


    ax.vlines(['2020-02-07'], 0, 482.24, linestyles='dashed', colors='black')
    ax.text(datetime.datetime.strptime(('2020/01/19'), '%Y/%m/%d'), 315, "lockdown measure in \n advance or delay",
            fontsize=7, color="black", style="italic", weight="light", horizontalalignment='left', rotation=0, fontname="Arial")

    ax.annotate('', xy=(datetime.datetime.strptime(('2020/02/07'), '%Y/%m/%d'), 0),
                xytext=(datetime.datetime.strptime(('2020/01/21'), '%Y/%m/%d'), 315),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4))
    ax.annotate('in advance', xy=(datetime.datetime.strptime(('2020/02/25'), '%Y/%m/%d'), 290),
                xytext=(datetime.datetime.strptime(('2020/02/25'), '%Y/%m/%d'), 320),
                arrowprops=dict(facecolor='black',shrink=0.05, width=2, headwidth=4,
                                headlength=2), fontname="Arial", fontsize=7)
    ax.annotate('delay', xy=(datetime.datetime.strptime(('2020/02/25'), '%Y/%m/%d'), 395),
                xytext=(datetime.datetime.strptime(('2020/02/25'), '%Y/%m/%d'), 355),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=4,
                                headlength=2), fontname="Arial", fontsize=7)
    ax.annotate('real data', xy=(datetime.datetime.strptime(('2020/03/05'), '%Y/%m/%d'), 353),
                xytext=(datetime.datetime.strptime(('2020/03/05'), '%Y/%m/%d'), 380),
                arrowprops=dict(color='r', shrink=0.05, width=2, headwidth=4,
                                headlength=2), color="red", fontname="Arial", fontsize=7)
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 8,
             }
    ax.legend(loc='upper left', frameon=False, prop=font2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    tq1_r = np.diff(s_covid_tq1_a)
    tq2_r = np.diff(s_covid_tq2_a)
    tq3_r = np.diff(s_covid_tq3_a)
    tq4_r = np.diff(s_covid_tq4_a)
    tq5_r = np.diff(s_covid_tq5_a)
    time2 = time[1:len(time)]
    ax1.set_ylim(0, 35)
    ax1.set_xlabel(u'Date/day', font2, verticalalignment='top')
    ax1.set_ylabel(u'Incidence', font2, verticalalignment='baseline', horizontalalignment='center')
    ax1.plot(time2, tq1_r, '+-', color='forestgreen', linewidth=1, label='Lockdown 1 day earlier', markersize=1.5)
    ax1.plot(time2, tq2_r, 's-', color='forestgreen', linewidth=1, label='Lockdown 2 day earlier', markersize=1.5)
    ax1.plot(time2, tq3_r, 'H-', color='forestgreen', linewidth=1, label='Lockdown 3 day earlier', markersize=1.5)
    ax1.plot(time2, tq4_r, 'x-', color='forestgreen', linewidth=1, label='Lockdown 4 day earlier', markersize=1.5)
    ax1.plot(time2, tq5_r, '2-', color='forestgreen', linewidth=1, label='Lockdown 5 day earlier', markersize=1.5)
    ax1.annotate(
        'Local transmission is almost contained \n when timing of lockdown is early enough',
        xy=(datetime.datetime.strptime(('2020/01/28'), '%Y/%m/%d'), 5),
        xytext=(datetime.datetime.strptime(('2020/02/18'), '%Y/%m/%d'), 15),
        arrowprops=dict(color='r', shrink=0.05, width=1, headwidth=2,
                        headlength=2), color="red", fontname="Arial", fontsize=6)

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax1.set_xticks(pd.date_range('2020-01-23', '2020-03-31', freq='12D'))
    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 6,
             }
    ax1.legend(loc='upper right', frameon=False, prop=font3)
    ax1.tick_params(labelsize=6)
    labels2 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.show()


def sy4plotresult_bj(file_path, time, real_covid_data):
    path_list = []
    for path_t in os.listdir(file_path):
        path_list.append(file_path + path_t)
    str_result_data = []
    for i in range(len(path_list)):
        s_covid = load_csv(path_list[i])
        str_i = np.array(s_covid)
        str_result_data.append(str_i)
    out_data = np.array(str_result_data)
    ii, jj, kk = out_data.shape
    print(out_data.shape)
    s_covid_tc1_a, s_covid_tc1_u, s_covid_tc1_d = [], [], []
    s_covid_tc2_a, s_covid_tc2_u, s_covid_tc2_d = [], [], []
    s_covid_tq1_a, s_covid_tq1_u, s_covid_tq1_d = [], [], []
    s_covid_tq2_a, s_covid_tq2_u, s_covid_tq2_d = [], [], []
    s_covid_tq3_a, s_covid_tq3_u, s_covid_tq3_d = [], [], []
    s_covid_tq4_a, s_covid_tq4_u, s_covid_tq4_d = [], [], []
    s_covid_tq5_a, s_covid_tq5_u, s_covid_tq5_d = [], [], []
    for j in range(jj-18):
        s_covid_tc1_a.append(float(out_data[0][j][0]))
        s_covid_tc1_u.append(float(out_data[0][j][1]))
        s_covid_tc1_d.append(float(out_data[0][j][2]))
        s_covid_tc2_a.append(float(out_data[1][j][0]))
        s_covid_tc2_u.append(float(out_data[1][j][1]))
        s_covid_tc2_d.append(float(out_data[1][j][2]))
        s_covid_tq1_a.append(float(out_data[2][j][0]))
        s_covid_tq1_u.append(float(out_data[2][j][1]))
        s_covid_tq1_d.append(float(out_data[2][j][2]))
        s_covid_tq2_a.append(float(out_data[3][j][0]))
        s_covid_tq2_u.append(float(out_data[3][j][1]))
        s_covid_tq2_d.append(float(out_data[3][j][2]))
        s_covid_tq3_a.append(float(out_data[4][j][0]))
        s_covid_tq3_u.append(float(out_data[4][j][1]))
        s_covid_tq3_d.append(float(out_data[4][j][2]))
        s_covid_tq4_a.append(float(out_data[5][j][0]))
        s_covid_tq4_u.append(float(out_data[5][j][1]))
        s_covid_tq4_d.append(float(out_data[5][j][2]))
        s_covid_tq5_a.append(float(out_data[6][j][0]))
        s_covid_tq5_u.append(float(out_data[6][j][1]))
        s_covid_tq5_d.append(float(out_data[6][j][2]))

    # plot
    fig = plt.figure(dpi=300, figsize=(8, 4.5))
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }

    ax = plt.Axes(fig, [.08, .12, .6, 0.8])
    ax.set_title('Beijing', font1)
    ax.set_xlabel(u'Date/day', font1, verticalalignment='top')
    ax.set_ylabel(u'Cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center')
    ax.set_ylim(0, 1100)
    ax1 = plt.Axes(fig, [.70, .25, .25, 0.4])
    fig.add_axes(ax)
    fig.add_axes(ax1)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    y_major_locator = MultipleLocator(300)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xticks(pd.date_range('2020-01-20', '2020-03-31', freq='15D'))


    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax.plot(time[0:72], real_covid_data[0:72], '-.', color='teal', linewidth=2.5, label='Official reported cumulative incidence',
            markersize=2)
    ax.plot(time, s_covid_tc1_a, 'o-', color='crimson', linewidth=1, label='Lockdown 1 day delay',
            markersize=2)
    ax.fill_between(time, s_covid_tc1_d, s_covid_tc1_u, color='salmon', alpha=0.8)
    ax.plot(time, s_covid_tc2_a, 'p-', color='crimson', linewidth=1, label='Lockdown 2 day delay',
            markersize=2)
    ax.fill_between(time, s_covid_tc2_d, s_covid_tc2_u, color='salmon', alpha=0.8)
    ax.plot(time, s_covid_tq1_a, '+-', color='forestgreen', linewidth=1, label='Lockdown 1 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq1_d, s_covid_tq1_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq2_a, 's-', color='forestgreen', linewidth=1, label='Lockdown 2 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq2_d, s_covid_tq2_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq3_a, 'H-', color='forestgreen', linewidth=1, label='Lockdown 3 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq3_d, s_covid_tq3_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq4_a, 'x-', color='forestgreen', linewidth=1, label='Lockdown 4 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq4_d, s_covid_tq4_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq5_a, '2-', color='forestgreen', linewidth=1, label='Lockdown 5 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq5_d, s_covid_tq5_u, color='lightgreen', alpha=0.8)

    ax.vlines(['2020-02-10'], 0, 451.4, linestyles='dashed', colors='black')

    ax.text(datetime.datetime.strptime(('2020/01/19'), '%Y/%m/%d'), 351.4, "lockdown measure in \n advance or delay",
            fontsize=7, color="black", style="italic", weight="light", horizontalalignment='left', rotation=0, fontname="Arial")
    ax.annotate('', xy=(datetime.datetime.strptime(('2020/02/10'), '%Y/%m/%d'), 0),
                xytext=(datetime.datetime.strptime(('2020/01/21'), '%Y/%m/%d'), 350),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4))
    ax.annotate('in advance', xy=(datetime.datetime.strptime(('2020/02/25'), '%Y/%m/%d'), 335),
                xytext=(datetime.datetime.strptime(('2020/02/25'), '%Y/%m/%d'), 385),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=4,
                                headlength=2), fontname="Arial", fontsize=7)
    ax.annotate('delay', xy=(datetime.datetime.strptime(('2020/02/25'), '%Y/%m/%d'), 525),
                xytext=(datetime.datetime.strptime(('2020/02/25'), '%Y/%m/%d'), 465),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=4,
                                headlength=2), fontname="Arial", fontsize=7)
    ax.annotate('real data', xy=(datetime.datetime.strptime(('2020/03/21'), '%Y/%m/%d'), 493),
                xytext=(datetime.datetime.strptime(('2020/03/26'), '%Y/%m/%d'), 530),
                arrowprops=dict(color='r', shrink=0.05, width=2, headwidth=4,
                                headlength=2), color="red", fontname="Arial", fontsize=7)
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 8,
             }
    ax.legend(loc='upper left', frameon=False, prop=font2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    tq1_r = np.diff(s_covid_tq1_a)
    tq2_r = np.diff(s_covid_tq2_a)
    tq3_r = np.diff(s_covid_tq3_a)
    tq4_r = np.diff(s_covid_tq4_a)
    tq5_r = np.diff(s_covid_tq5_a)
    time2 = time[1:len(time)]
    ax1.set_ylim(0, 30)
    ax1.set_xlabel(u'Date/day', font2, verticalalignment='top')
    ax1.set_ylabel(u'Incidence', font2, verticalalignment='baseline', horizontalalignment='center')
    ax1.plot(time2, tq1_r, '+-', color='forestgreen', linewidth=1, label='Lockdown 1 day earlier', markersize=1.5)
    ax1.plot(time2, tq2_r, 's-', color='forestgreen', linewidth=1, label='Lockdown 2 day earlier', markersize=1.5)
    ax1.plot(time2, tq3_r, 'H-', color='forestgreen', linewidth=1, label='Lockdown 3 day earlier', markersize=1.5)
    ax1.plot(time2, tq4_r, 'x-', color='forestgreen', linewidth=1, label='Lockdown 4 day earlier', markersize=1.5)
    ax1.plot(time2, tq5_r, '2-', color='forestgreen', linewidth=1, label='Lockdown 5 day earlier', markersize=1.5)
    ax1.annotate(
        'Advancing the timing of lockdown may have \n limited impact on reducing incidence when \n risk score and level of incidence is low',
        xy=(datetime.datetime.strptime(('2020/02/03'), '%Y/%m/%d'), 6),
        xytext=(datetime.datetime.strptime(('2020/02/18'), '%Y/%m/%d'), 13),
        arrowprops=dict(color='r', shrink=0.05, width=1, headwidth=2,
                        headlength=2), color="red", fontname="Arial", fontsize=6)

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax1.set_xticks(pd.date_range('2020-01-23', '2020-03-31', freq='12D'))
    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 6,
             }
    ax1.legend(loc='upper right', frameon=False, prop=font3)
    ax1.tick_params(labelsize=6)
    labels2 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.show()


def sy4plotresult_wh(file_path, time, real_covid_data):
    path_list = []
    for path_t in os.listdir(file_path):
        path_list.append(file_path + path_t)
    str_result_data = []
    for i in range(len(path_list)):
        s_covid = load_csv(path_list[i])
        str_i = np.array(s_covid)
        str_result_data.append(str_i)
    out_data = np.array(str_result_data)
    ii, jj, kk = out_data.shape
    print(out_data)
    s_covid_tc1_a, s_covid_tc1_u, s_covid_tc1_d = [], [], []
    s_covid_tc2_a, s_covid_tc2_u, s_covid_tc2_d = [], [], []
    s_covid_tq1_a, s_covid_tq1_u, s_covid_tq1_d = [], [], []
    s_covid_tq2_a, s_covid_tq2_u, s_covid_tq2_d = [], [], []
    s_covid_tq3_a, s_covid_tq3_u, s_covid_tq3_d = [], [], []
    s_covid_tq4_a, s_covid_tq4_u, s_covid_tq4_d = [], [], []
    s_covid_tq5_a, s_covid_tq5_u, s_covid_tq5_d = [], [], []
    for j in range(jj-5):
        s_covid_tc1_a.append(float(out_data[0][j][0]))
        s_covid_tc1_u.append(float(out_data[0][j][1]))
        s_covid_tc1_d.append(float(out_data[0][j][2]))
        s_covid_tc2_a.append(float(out_data[1][j][0]))
        s_covid_tc2_u.append(float(out_data[1][j][1]))
        s_covid_tc2_d.append(float(out_data[1][j][2]))
        s_covid_tq1_a.append(float(out_data[2][j][0]))
        s_covid_tq1_u.append(float(out_data[2][j][1]))
        s_covid_tq1_d.append(float(out_data[2][j][2]))
        s_covid_tq2_a.append(float(out_data[3][j][0]))
        s_covid_tq2_u.append(float(out_data[3][j][1]))
        s_covid_tq2_d.append(float(out_data[3][j][2]))
        s_covid_tq3_a.append(float(out_data[4][j][0]))
        s_covid_tq3_u.append(float(out_data[4][j][1]))
        s_covid_tq3_d.append(float(out_data[4][j][2]))
        s_covid_tq4_a.append(float(out_data[5][j][0]))
        s_covid_tq4_u.append(float(out_data[5][j][1]))
        s_covid_tq4_d.append(float(out_data[5][j][2]))
        s_covid_tq5_a.append(float(out_data[6][j][0]))
        s_covid_tq5_u.append(float(out_data[6][j][1]))
        s_covid_tq5_d.append(float(out_data[6][j][2]))

    # plot
    fig = plt.figure(dpi=300, figsize=(8, 4.5))
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }

    ax = plt.Axes(fig, [.10, .12, .58, 0.8])
    ax.set_title('Wuhan', font1)
    ax.set_xlabel(u'Date/day', font1, verticalalignment='top')
    ax.set_ylabel(u'Cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center')
    ax.set_ylim(0, 120000)
    ax1 = plt.Axes(fig, [.705, .25, .25, 0.4])
    fig.add_axes(ax)
    fig.add_axes(ax1)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))  # 设置时间标签显示格式
    y_major_locator = MultipleLocator(30000)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xticks(pd.date_range('2020-01-07', '2020-03-31', freq='15D'))


    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax.plot(time[0:85], real_covid_data[0:85], '-.', color='teal', linewidth=2.5, label='Official reported cumulative incidence',
            markersize=2)
    ax.plot(time, s_covid_tc1_a, 'o-', color='crimson', linewidth=1, label='Lockdown 1 day delay',
            markersize=2)
    ax.fill_between(time, s_covid_tc1_d, s_covid_tc1_u, color='salmon', alpha=0.8)
    ax.plot(time, s_covid_tc2_a, 'p-', color='crimson', linewidth=1, label='Lockdown 2 day delay',
            markersize=2)
    ax.fill_between(time, s_covid_tc2_d, s_covid_tc2_u, color='salmon', alpha=0.8)
    ax.plot(time, s_covid_tq1_a, '+-', color='forestgreen', linewidth=1, label='Lockdown 1 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq1_d, s_covid_tq1_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq2_a, 's-', color='forestgreen', linewidth=1, label='Lockdown 2 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq2_d, s_covid_tq2_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq3_a, 'H-', color='forestgreen', linewidth=1, label='Lockdown 3 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq3_d, s_covid_tq3_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq4_a, 'x-', color='forestgreen', linewidth=1, label='Lockdown 4 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq4_d, s_covid_tq4_u, color='lightgreen', alpha=0.8)
    ax.plot(time, s_covid_tq5_a, '2-', color='forestgreen', linewidth=1, label='Lockdown 5 day earlier',
            markersize=2)
    ax.fill_between(time, s_covid_tq5_d, s_covid_tq5_u, color='lightgreen', alpha=0.8)

    ax.vlines(['2020-01-23'], 0, 20083.44, linestyles='dashed', colors='black')

    ax.text(datetime.datetime.strptime(('2020/01/04'), '%Y/%m/%d'), 30000, "lockdown measure in \n advance or delay",
            fontsize=7, color="black", style="italic", weight="light", horizontalalignment='left', rotation=0, fontname="Arial")
    ax.annotate('', xy=(datetime.datetime.strptime(('2020/01/23'), '%Y/%m/%d'), 10),
                xytext=(datetime.datetime.strptime(('2020/01/13'), '%Y/%m/%d'), 29800),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=3,
                                headlength=4))
    ax.annotate('in advance', xy=(datetime.datetime.strptime(('2020/02/06'), '%Y/%m/%d'), 40000),
                xytext=(datetime.datetime.strptime(('2020/02/06'), '%Y/%m/%d'), 45000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=4,
                                headlength=2), fontname="Arial", fontsize=7)
    ax.annotate('delay', xy=(datetime.datetime.strptime(('2020/02/06'), '%Y/%m/%d'), 58000),
                xytext=(datetime.datetime.strptime(('2020/02/06'), '%Y/%m/%d'), 51000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=4,
                                headlength=2), fontname="Arial", fontsize=7)
    ax.annotate('real data', xy=(datetime.datetime.strptime(('2020/03/01'), '%Y/%m/%d'), 50500),
                xytext=(datetime.datetime.strptime(('2020/02/25'), '%Y/%m/%d'), 55000),
                arrowprops=dict(color='r', shrink=0.05, width=2, headwidth=4,
                                headlength=2), color="red", fontname="Arial", fontsize=7)
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 8,
             }
    ax.legend(loc='upper left', frameon=False, prop=font2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    tq1_r = np.diff(s_covid_tq1_a)
    tq2_r = np.diff(s_covid_tq2_a)
    tq3_r = np.diff(s_covid_tq3_a)
    tq4_r = np.diff(s_covid_tq4_a)
    tq5_r = np.diff(s_covid_tq5_a)
    time2 = time[1:len(time)]
    ax1.set_ylim(0, 2100)
    ax1.set_xlabel(u'Date/day', font2, verticalalignment='top')
    ax1.set_ylabel(u'Incidence', font2, verticalalignment='baseline', horizontalalignment='center')
    ax1.plot(time2, tq1_r, '+-', color='forestgreen', linewidth=1, label='Lockdown 1 day earlier', markersize=1.5)
    ax1.plot(time2, tq2_r, 's-', color='forestgreen', linewidth=1, label='Lockdown 2 day earlier', markersize=1.5)
    ax1.plot(time2, tq3_r, 'H-', color='forestgreen', linewidth=1, label='Lockdown 3 day earlier', markersize=1.5)
    ax1.plot(time2, tq4_r, 'x-', color='forestgreen', linewidth=1, label='Lockdown 4 day earlier', markersize=1.5)
    ax1.plot(time2, tq5_r, '2-', color='forestgreen', linewidth=1, label='Lockdown 5 day earlier', markersize=1.5)
    ax1.annotate('Advancing the timing of lockdown in \n epicenter reduces incidence and \n contain movements of at-risk population',
        xy=(datetime.datetime.strptime(('2020/01/27'), '%Y/%m/%d'), 380),
        xytext=(datetime.datetime.strptime(('2020/02/15'), '%Y/%m/%d'), 700),
        arrowprops=dict(color='r', shrink=0.05, width=1, headwidth=2,
                        headlength=2), color="red", fontname="Arial", fontsize=6)

    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))
    ax1.set_xticks(pd.date_range('2020-01-08', '2020-03-31', freq='12D'))
    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 6,
             }
    ax1.legend(loc='upper right', frameon=False, prop=font3)
    ax1.tick_params(labelsize=6)
    labels2 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.show()


def sy5_multi_country(mean_c_data, u_c_data, d_c_data, mean_k_data, u_k_data, d_k_data, mean_a_data, u_a_data, d_a_data, mean_e_data, u_e_data, d_e_data, time):
    # recurrent waves
    file_path_rw = 'C:/Users/Admin/Desktop/xinzeng-multi countries/recurrent waves.csv'
    df = pd.read_csv(file_path_rw)
    rw = [float(i) for i in df.iloc[0:len(df), 0]]

    # plot
    fig = plt.figure(dpi=300, figsize=(7, 5))
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }

    ax = plt.Axes(fig, [.12, .14, .85, 0.8])
    # ax.set_title('Wuhan', font1)
    ax.set_xlabel(u'Date/day', font1, verticalalignment='top')
    ax.set_ylabel(u'Cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center')
    ax.set_ylim(0, 200000)
    ax1 = plt.Axes(fig, [.27, .47, .22, 0.30])
    fig.add_axes(ax)
    fig.add_axes(ax1)
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    y_major_locator = MultipleLocator(40000)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.set_xticks(pd.date_range('2020-01-07', '2020-03-31', freq='15D'))

    ax.hlines(60000, ['2020-02-10'], ['2020-03-31'], linestyles=':', linewidth=1.5, color='k')
    ax.vlines(['2020-02-10'], 0, 60000, linestyles='dashed', linewidth=1.5, colors='darkorange')
    ax.vlines(['2020-03-03'], 0, 60000, linestyles='dashed', linewidth=1.5, colors='darkcyan')
    ax.hlines(182000, ['2020-03-21'], ['2020-03-31'], linestyles=':', linewidth=1.5, color='k')
    ax.vlines(['2020-03-21'], 0, 182000, linestyles='dashed', linewidth=1.5, colors='gray')

    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    t_series = list(range(85))
    ax.plot(time, mean_c_data, 'o-', color='mediumseagreen', linewidth=1, label='CHN',
            markersize=2)
    ax.fill_between(time, d_c_data, u_c_data, color='aquamarine', alpha=0.6)
    ax.plot(time, mean_k_data, 'p-', color='steelblue', linewidth=1, label='KOR',
            markersize=2)
    ax.fill_between(time, d_k_data, u_k_data, color='lightskyblue', alpha=0.6)
    ax.plot(time, mean_a_data, '+-', color='hotpink', linewidth=1, label='USA',
            markersize=2)
    ax.fill_between(time, d_a_data, u_a_data, color='pink', alpha=0.6)
    ax.plot(time, mean_e_data, 's-', color='red', linewidth=1, label='GBR',
            markersize=2)
    ax.fill_between(time, d_e_data, u_e_data, color='salmon', alpha=0.6)

    ax.text(datetime.datetime.strptime(('2020/03/12'), '%Y/%m/%d'), 115000, "Loose mode: \n less containment but \n more infection",
            fontsize=8, color="black", style="italic", weight="light", horizontalalignment='center', rotation=0,
            fontname="Arial")

    ax.text(datetime.datetime.strptime(('2020/03/12'), '%Y/%m/%d'), 25000, "Strict mode: \n more containment but \n less infection",
            fontsize=8, color="black", style="italic", weight="light", horizontalalignment='center', rotation=0,
            fontname="Arial")
    ax.annotate(
        'Whereafter, it would face huge recurrent \n waves in loose mode strategy',
        xy=(datetime.datetime.strptime(('2020/03/19'), '%Y/%m/%d'), 180000),
        xytext=(datetime.datetime.strptime(('2020/02/11'), '%Y/%m/%d'), 178000),
        arrowprops=dict(color='k', shrink=0.05, width=1, headwidth=4,
                        headlength=2), color="k", fontname="Arial", fontsize=8)
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 8,
             }
    ax.legend(loc='upper left', frameon=False, prop=font2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mc_r = np.diff(mean_c_data)
    mk_r = np.diff(mean_k_data)
    ma_r = np.diff(mean_a_data)
    me_r = np.diff(mean_e_data)
    time2 = time[1:len(time)]
    ax1.set_ylim(0, 8000)
    ax1.set_xlabel(u'Date/day', font2, verticalalignment='top')
    ax1.set_ylabel(u'Incidence', font2, verticalalignment='baseline', horizontalalignment='center')
    ax1.plot(time2, mc_r, 'o-', color='mediumseagreen', linewidth=1, label='CHN', markersize=1.5)
    ax1.plot(time2, mk_r, 'p-', color='steelblue', linewidth=1, label='KOR', markersize=1.5)
    ax1.plot(time2, ma_r, '+-', color='hotpink', linewidth=1, label='USA', markersize=1.5)
    ax1.plot(time2, me_r, 's-', color='red', linewidth=1, label='GBR', markersize=1.5)
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%m-%d'))  # 设置时间标签显示格式
    y_major_locator1 = MultipleLocator(2000)
    ax1.yaxis.set_major_locator(y_major_locator1)
    ax1.set_xticks(pd.date_range('2020-01-08', '2020-03-31', freq='15D'))
    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 7,
             }
    ax1.legend(loc='upper left', frameon=False, prop=font3)
    ax1.tick_params(labelsize=6)
    labels2 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ax2 = plt.Axes(fig, [.40, .65, .2, .2])
    # fig.add_axes(ax2)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # t_series = list(range(len(df)))
    # ax2.plot(t_series, rw, 'h-', color='grey', linewidth=1, label='CHN',
    #         markersize=1)

    plt.show()


def sy5_country_ec():
# plot barchart of economic costs
    x_pos = np.arange(4)
    gdp_loss = [176.67, 153.19, 131.69, 113.44]
    md_loss = [43.53, 51.25, 151.83, 157.28]
    total_loss = [220.2, 204.44, 283.52, 270.72]
    gdp_loss, md_loss, total_loss = np.divide(gdp_loss, 6.391), np.divide(md_loss, 6.391), np.divide(total_loss, 6.391)
    gdp_err = [25.16, 21.08, 19.98, 17.78]
    md_err = [3.68, 4.89, 17.16, 18.99]
    total_err = [28.84, 25.97, 37.14, 36.77]
    gdp_err, md_err, total_err = np.divide(gdp_err, 6.391), np.divide(md_err, 6.391), np.divide(total_err, 6.391)

    fig = plt.figure(dpi=300, figsize=(7, 5))
    # ax = plt.Axes(fig, [.12, .18, .38, 0.7])
    # ax = plt.Axes(fig, [.12, .14, .83, 0.75])
    # fig.add_axes(ax)
    # ax1 = plt.Axes(fig, [.6, .18, .35, 0.6])
    ax1 = plt.Axes(fig, [.12, .14, .83, 0.65])
    fig.add_axes(ax1)
    # ax.axvspan(xmin=-0.5, xmax=1.8, facecolor='bisque', alpha=0.5)
    # ax.axvspan(1.8, 4.5, facecolor='aquamarine', alpha=0.35)
    # ax.axhline(y=0, ls="-.", c="black", linewidth=1)
    # ax.bar(x_pos, gdp_loss, yerr=gdp_err, width=0.3, align='center', color='deepskyblue',
    #        label='GDP loss')
    # bar_width = 0.3
    # ax.bar(x_pos + bar_width, md_loss, yerr=md_err, width=bar_width, align='center', color='mediumspringgreen',
    #        label='Medical expenditure')
    # ax.bar(x_pos + 2*bar_width, total_loss, yerr=total_err, width=bar_width, align='center', color='lightpink',
    #    label='Total economic costs')
    # ax.set_xticks([])
    # ax.set_ylim(-15.6, 55)
    # ax.set_yticks([0, 55])
    # ax.set_xlabel('Intervention strategies adopted by various countries', fontname="Arial", fontsize=12, labelpad=20)
    # ax.set_ylabel('Economic costs of intervention strategies \n (billion USD $)', fontname="Arial", fontsize=12, labelpad=2)
    #
    # ax.set_xlim(min(x_pos) - 0.5, max(x_pos) + 1)
    # ax.tick_params(labelsize=10)
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax.text(0.33, -3.13, "CHN strategy", fontsize=8, color="black", weight="light",
    #          horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(1.33, -3.13, "KOR strategy", fontsize=8, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(2.33, -3.13, "USA strategy", fontsize=8, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(3.33, -3.13, "GBR strategy", fontsize=8, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    #
    # ax.text(0.8, -11.73, "Strict mode: \n less economic costs",
    #         fontsize=10, color="black", style="italic", weight="light", horizontalalignment='center', rotation=0,
    #         fontname="Arial")
    #
    # ax.text(2.8, -11.73, "Loose mode: \n more economic costs",
    #         fontsize=10, color="black", style="italic", weight="light", horizontalalignment='center', rotation=0,
    #         fontname="Arial")
    #
    # font2 = {'family': 'Arial',
    #          'weight': 'normal',
    #          'size': 8,
    #          }
    # ax.legend(loc='upper left', frameon=False, prop=font2)

    ax1.set_xlim(0, 30)
    x_major_locator = MultipleLocator(6)
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.set_ylim(0, 30)
    ax1.set_xlabel('Medical expenditure (billion USD $)', fontname="Arial", fontsize=12, verticalalignment='top', labelpad=10)
    ax1.set_ylabel('GDP loss (billion USD $)', fontname="Arial", fontsize=12, verticalalignment='baseline', horizontalalignment='center', labelpad=10)
    area = np.pi * 2 ** 2
    size = [1.6, 1.1, 2.4, 1.9]
    size = np.array(size)
    marker_size = area * size
    ax1.scatter(md_loss, gdp_loss, s=marker_size, c='#DC143C', marker=u'o', alpha=0.5)
    x_axis_bounds = np.array(ax1.get_xlim())
    ax1.plot(x_axis_bounds, 4.7 * x_axis_bounds, 'k--', linewidth=1)
    ax1.plot(x_axis_bounds, 0.65 * x_axis_bounds, 'k--', linewidth=1)
    ax1.annotate(
        'strict mode \n strategy',
        xy=(50/6.391, 178/6.391), xytext=(50/6.391, 205/6.391), arrowprops=dict(color='k', shrink=0.05, width=1, headwidth=4,
                        headlength=2), color="k", fontname="Arial", horizontalalignment='center', fontsize=8)
    ax1.annotate(
        'loose mode \n strategy',
        xy=(159/6.391, 134/6.391), xytext=(159/6.391, 160/6.391), arrowprops=dict(color='k', shrink=0.05, width=1, headwidth=4,
                        headlength=2), color="k", fontname="Arial", horizontalalignment='center', fontsize=8)

    elll = Ellipse(xy = (100/6.391, 145/6.391), width=30/6.391, height=110/6.391, angle=60, facecolor='y', alpha=0.3)
    ax1.add_patch(elll)
    ax1.annotate(
        'middle ground: \n offers governors chances to \n adjust their interventions',
        xy=(105/6.391, 165/6.391), xytext=(105/6.391, 205/6.391), arrowprops=dict(color='k', shrink=0.05, width=1, headwidth=4,
                        headlength=2), color="darkred", fontname="Arial", horizontalalignment='center', fontsize=8)
    ax1.text(43.5/6.391, 167/6.391, "CHN", fontsize=8, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(51.2/6.391, 145/6.391, "KOR", fontsize=8, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(157.8/6.391, 124/6.391, "USA", fontsize=8, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(152.2/6.391, 105/6.391, "GBR", fontsize=8, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.tick_params(labelsize=10)
    labels1 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Arial') for label in labels1]
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.show()


def sy6_multisize():
    ic_1 = [120.29, 134.93, 173.42, 207.34, 250.44, 256.98]
    ic_2 = [126.18, 147.67, 174.72, 206.50, 246.50, 271.17]
    ic_3 = [122.84, 141.64, 168.05, 203.99, 243.94, 264.33]
    sd_1 = [144.93, 204.17, 347.75, 540.25, 806.75]
    sd_2 = [147.67, 206.23, 332.14, 543.51, 827.19]
    sd_3 = [141.64, 228.65, 352.93, 547.33, 819.93]
    dd_1 = [223.12, 210.05, 204.17, 186.61, 168.31, 165.31, 152.37]
    dd_2 = [232.35, 208.21, 202.12, 184.77, 166.54, 165.43, 154.62]
    dd_3 = [229.96, 207.28, 200.13, 190.48, 168.47, 165.11, 159.75]
    x_pos1 = np.arange(6)
    y_pos1 = [1, 1, 1, 1, 1, 1]
    y_pos2 = [2, 2, 2, 2, 2, 2]
    y_pos3 = [3, 3, 3, 3, 3, 3]
    fig = plt.figure(dpi=300, figsize=(4, 6))
    ax = plt.Axes(fig, [.1, .40, .8, 0.25])
    fig.add_axes(ax)
    ax1 = plt.Axes(fig, [.1, .10, .8, 0.25])
    fig.add_axes(ax1)
    ax2 = plt.Axes(fig, [.1, .71, .8, 0.25])
    fig.add_axes(ax2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 4)
    colors = 'lightgreen'
    area = np.pi * 3.5 ** 3.5
    size_ic1 = np.array(ic_1) / 271.17
    marker_size_ic1 = area * size_ic1
    ax.scatter(x_pos1, y_pos1, s=marker_size_ic1, c=colors,  alpha=0.5)
    size_ic2 = np.array(ic_2) / 271.17
    marker_size_ic2 = area * size_ic2
    ax.scatter(x_pos1, y_pos2, s=marker_size_ic2, c=colors, alpha=0.5)
    size_ic3 = np.array(ic_3) / 271.17
    marker_size_ic3 = area * size_ic3
    ax.scatter(x_pos1, y_pos3, s=marker_size_ic3, c=colors, alpha=0.5)
    ax.text(0, -0.4, "4 cases", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(1, -0.4, "5 cases", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(2, -0.4, "6 cases", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(3, -0.4, "7 cases", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(4, -0.4, "8 cases", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(5, -0.4, "9 cases", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(-0.4, 1, "1", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(-0.4, 2, "5", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(-0.4, 3, "10", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(2.5, 3.6, "Initial number of infected cases", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax.set_ylabel('Population sizes of a virtual city (million)', fontname="Arial", fontsize=10, labelpad=10)

    ax2.set_xticks([])
    ax2.set_yticks([])
    x_pos2 = np.arange(5)
    y_pos1 = [1, 1, 1, 1, 1]
    y_pos2 = [2, 2, 2, 2, 2]
    y_pos3 = [3, 3, 3, 3, 3]
    ax2.set_ylim(0, 4)
    colors = 'turquoise'
    area = np.pi * 3.5 ** 3.5
    size_sd1 = np.array(sd_1) / 827.19
    marker_size_sd1 = area * size_sd1
    ax2.scatter(x_pos2, y_pos1, s=marker_size_sd1, c=colors, alpha=0.5)
    size_sd2 = np.array(sd_2) / 827.19
    marker_size_sd2 = area * size_sd2
    ax2.scatter(x_pos2, y_pos2, s=marker_size_sd2, c=colors, alpha=0.5)
    size_sd3 = np.array(sd_3) / 827.19
    marker_size_sd3 = area * size_sd3
    ax2.scatter(x_pos2, y_pos3, s=marker_size_sd3, c=colors, alpha=0.5)
    ax2.text(0, -0.7, "the 7th \n day", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(1, -0.7, "the 8th \n day", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(2, -0.7, "the 9th \n day", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(3, -0.7, "the 10th \n day", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(4, -0.7, "the 11th \n day", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(-0.3, 1, "1", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(-0.3, 2, "5", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(-0.3, 3, "10", fontsize=7, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax2.text(2, 3.6, "The starting time of lockdowns", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.set_xticks([])
    ax1.set_yticks([])
    x_pos3 = np.arange(7)
    y_pos1 = [1, 1, 1, 1, 1, 1, 1]
    y_pos2 = [2, 2, 2, 2, 2, 2, 2]
    y_pos3 = [3, 3, 3, 3, 3, 3, 3]
    ax1.set_ylim(0, 4)
    colors = 'crimson'
    area = np.pi * 3.5 ** 3.5
    size_dd1 = np.array(dd_1) / 232.25
    marker_size_dd1 = area * size_dd1
    ax1.scatter(x_pos3, y_pos1, s=marker_size_dd1, c=colors, alpha=0.4)
    size_dd2 = np.array(dd_2) / 232.25
    marker_size_dd2 = area * size_dd2
    ax1.scatter(x_pos3, y_pos2, s=marker_size_dd2, c=colors, alpha=0.4)
    size_dd3 = np.array(dd_3) / 232.25
    marker_size_dd3 = area * size_dd3
    ax1.scatter(x_pos3, y_pos3, s=marker_size_dd3, c=colors, alpha=0.4)
    ax1.text(0, -0.5, "7 days", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(1, -0.5, "8 days", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(2, -0.5, "9 days", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(3, -0.5, "10 days", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(4, -0.5, "11 days", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(5, -0.5, "12 days", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(6, -0.5, "13 days", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(-0.5, 1, "1", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(-0.5, 2, "5", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(-0.5, 3, "10", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.text(3, 3.6, "The duration time of lockdowns", fontsize=7, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.set_xlabel('Epidemic spread result', fontname="Arial", fontsize=10, labelpad=15)
    plt.show()


def c_cluster():
    # filepath = "C:/Users/Admin/Desktop/xinzeng-multi countries/多国/costs.csv"
    filepath = "C:/Users/Admin/Desktop/xinzeng-multi countries/多国/costs-new.csv"
    cost_data = pd.read_csv(filepath)
    country = cost_data["country"][:].tolist()
    short_name = cost_data["short_n"].tolist()
    GDP = cost_data["Quarterly variation"][:].tolist()
    # continent = cost_data["continent"][:].tolist()
    perGDP = cost_data["PERGDP_loss"][:].tolist()
    GDP_variation = cost_data["PERGDP_loss"][:].tolist()
    MD = cost_data["cost_total"][:].tolist()
    perMD = cost_data["per_cost_total"][:].tolist()
    c_incidence = cost_data["c_incidence"][:].tolist()
    c_incidence_rate = cost_data["c_incidence_rate"][:].tolist()
    # log_MD = [math.log(i, 10) for i in MD]
    # log_perGDP = [math.log(i, 10) for i in perGDP]
    data = np.vstack((MD, perGDP))
    data = np.transpose(data)
    plt.scatter(data[:, 0], data[:, 1], c="red", marker='o', label='see')
    plt.show()

    # estimator = KMeans(n_clusters=3)
    # estimator.fit(data)
    # label_pred = estimator.labels_

    # x0 = data[label_pred == 0]
    # x1 = data[label_pred == 1]
    # x2 = data[label_pred == 2]
    # plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
    # plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
    # plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')

    clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)
    res = clustering.fit(data)
    d0 = data[clustering.labels_ == 0]
    plt.plot(d0[:, 0], d0[:, 1], 'r.')
    d1 = data[clustering.labels_ == 1]
    plt.plot(d1[:, 0], d1[:, 1], 'go')
    d2 = data[clustering.labels_ == 2]
    plt.plot(d2[:, 0], d2[:, 1], 'b*')

    for i in range(0, len(country)):
        plt.text(MD[i], perGDP[i], country[i], fontsize=10, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    plt.xlabel('Governmental expenses')
    plt.ylabel('Per capital GDP loss')
    plt.legend(loc=2)
    plt.show()


def multi_country():
    # filepath = "C:/Users/Admin/Desktop/xinzeng-multi countries/多国/costs.csv"
    filepath = "C:/Users/Admin/Desktop/xinzeng-multi countries/多国/costs-new.csv"
    cost_data = pd.read_csv(filepath)
    country = cost_data["country"][:].tolist()
    short_name = cost_data["short_n"].tolist()
    GDP = cost_data["Quarterly variation"][:].tolist()
    perGDP = cost_data["PERGDP_loss"][:].tolist()
    GDP_variation = cost_data["GDP_variation"][:].tolist()
    MD = cost_data["cost_total"][:].tolist()
    # perMD = cost_data["per_cost_total"][:].tolist()
    c_incidence = cost_data["c_incidence"][:].tolist()
    c_incidence_rate = cost_data["c_incidence_rate"][:].tolist()
    total = [GDP[i] + MD[i] for i in range(min(len(GDP), len(MD)))]
    # per_total = [perGDP[i] + perMD[i] for i in range(min(len(perGDP), len(perMD)))]
    # log_MD = [math.log(i, 10) for i in MD]
    # log_perGDP = [math.log(i, 10) for i in perGDP]
    x_pos = np.arange(68).tolist()
    fig = plt.figure(dpi=300, figsize=(8, 6))
    ax = plt.Axes(fig, [.1, .12, .25, 0.35])
    fig.add_axes(ax)
    colors = '#DC143C'
    area = np.pi * 2 ** 2
    ax.scatter(GDP, MD, s=area, c=colors, marker=u'o', alpha=0.5)
    for i in range(0, len(country)):
        ax.text(GDP[i], MD[i], country[i], fontsize=6, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }
    ax.set_xlabel(u'GDP loss', font1, verticalalignment='top')
    ax.set_ylabel(u'Medical expenditure', font1, verticalalignment='baseline', horizontalalignment='center')
    # ax.set_ylim(0, 200000)
    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax1 = plt.Axes(fig, [.4, .12, .25, 0.35])
    fig.add_axes(ax1)
    colors = '#DC143C'
    area = np.pi * 2 ** 2
    ax1.scatter(perGDP, MD, s=area, c=colors, marker=u'o', alpha=0.5)
    # ax1.scatter(log_perGDP, log_MD, s=area, c=colors, marker=u'o', alpha=0.5)
    for i in range(0, len(country)):
        ax1.text(perGDP[i], MD[i], country[i], fontsize=6, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.set_xlabel(u'Per capital GDP loss', font1, verticalalignment='top')
    ax1.set_ylabel(u'Governmental expenses', font1, verticalalignment='baseline', horizontalalignment='center')
    # ax.set_ylim(0, 200000)
    plt.tick_params(labelsize=10)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax2 = plt.Axes(fig, [.75, .12, .22, 0.35])
    fig.add_axes(ax2)
    colors = '#DC143C'
    area = np.pi * 2 ** 2
    # ax2.scatter(x_pos, total, s=area, c=colors, marker=u'o', alpha=0.5)
    ax2.bar(x_pos, total, width=0.3, align='center', color='deepskyblue', label='Econommic loss')
    for i in range(0, len(country)):
        ax2.text(x_pos[i], total[i], country[i], fontsize=6, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")

    ax2.set_ylabel(u'Economic loss', font1, verticalalignment='baseline', horizontalalignment='center')
    ax2.set_xticks([])
    ax2.set_xlabel(u'Country', font1, verticalalignment='baseline', horizontalalignment='center', labelpad=15)
    plt.tick_params(labelsize=10)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax3 = plt.Axes(fig, [.1, .60, .25, 0.35])
    fig.add_axes(ax3)
    colors = '#DC143C'
    area = np.pi * 2 ** 2
    ax3.scatter( GDP_variation, c_incidence_rate, s=area, c=colors, marker=u'o', alpha=0.5)
    for i in range(0, len(country)):
        ax3.text( GDP_variation[i], c_incidence_rate[i], short_name[i], fontsize=6, color="black", weight="light",
                 horizontalalignment='center', rotation=0, fontname="Arial")
    ax3.set_xlabel(u'Per capital GDP loss', font1, verticalalignment='top')
    ax3.set_ylabel(u'Per capital governmental expenses', font1, verticalalignment='baseline', horizontalalignment='center')
    plt.tick_params(labelsize=10)
    labels = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax4 = plt.Axes(fig, [.4, .60, .25, 0.35])
    fig.add_axes(ax4)
    colors = '#DC143C'
    area = np.pi * 2 ** 2
    ax4.scatter(perGDP, MD, s=area, c=colors, marker=u'o', alpha=0.5)
    for i in range(0, len(country)):
        ax4.text(perGDP[i], MD[i], country[i], fontsize=6, color="black", weight="light",
             horizontalalignment='center', rotation=0, fontname="Arial")
    ax4.set_xlabel(u'Per capital GDP loss', font1, verticalalignment='top')
    ax4.set_ylabel(u'Governmental expenses', font1, verticalalignment='baseline', horizontalalignment='center')
    plt.tick_params(labelsize=10)
    labels = ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    # ax5 = plt.Axes(fig, [.75, .60, .22, 0.35])
    # fig.add_axes(ax5)
    # ax5.bar(x_pos, per_total, width=0.3, align='center', color='deepskyblue', label='Econommic loss')
    # for i in range(0, len(country)):
    #     ax5.text(x_pos[i], per_total[i], country[i], fontsize=6, color="black", weight="light",
    #              horizontalalignment='center', rotation=0, fontname="Arial")
    # ax5.set_ylabel(u'Per capital economic loss', font1, verticalalignment='baseline', horizontalalignment='center')
    # ax5.set_xticks([])
    # ax5.set_xlabel(u'Country', font1, verticalalignment='baseline', horizontalalignment='center', labelpad=15)
    # plt.tick_params(labelsize=10)
    # labels = ax5.get_xticklabels() + ax5.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    plt.show()

def to_percent(temp, position):
    return '%1.0f'%(10*temp) + '%'


def multi_country2():
    filepath = "C:/Users/Admin/Desktop/xinzeng-multi countries/多国/costs-new.csv"
    cost_data = pd.read_csv(filepath)
    country = cost_data["country"][:].tolist()
    short_name = cost_data["short_n"].tolist()
    GDP = cost_data["Quarterly variation"][:].tolist()
    perGDP = cost_data["PERGDP_loss"][:].tolist()
    GDP_variation = cost_data["GDP_variation"][:].tolist()
    MD = cost_data["cost_total"][:].tolist()
    perMD = cost_data["per_cost_total"][:].tolist()
    c_incidence = cost_data["c_incidence"][:].tolist()
    c_incidence_rate = cost_data["c_incidence_rate"][:].tolist()
    total = [GDP[i] + MD[i] for i in range(min(len(GDP), len(MD)))]
    per_total = [perGDP[i] + perMD[i] for i in range(min(len(perGDP), len(perMD)))]
    total_name = zip(total, short_name)
    sorted_total_name = sorted(total_name, key=lambda x: x[0])
    result = zip(*sorted_total_name)
    sorted_total, sorted_name = [list(x) for x in result]
    # log_MD = [math.log(i, 10) for i in MD]
    # log_perGDP = [math.log(i, 10) for i in perGDP]
    x_pos = np.arange(62).tolist()

    fig = plt.figure(dpi=300, figsize=(8, 6))
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }
    plt.bar(x_pos, sorted_total, width=0.95, align='center', color='c', alpha=0.3, label='Econommic costs')
    for i in range(0, len(country)):
        plt.text(x_pos[i], 1.05 * sorted_total[i], sorted_name[i], fontsize=6, color="black", weight="light",
                 horizontalalignment='center', rotation=90, fontname="Arial")
    plt.ylabel(u'Total economic costs', font1, verticalalignment='baseline', horizontalalignment='center')
    plt.xticks([])
    plt.xlabel(u'Country', font1, verticalalignment='baseline', horizontalalignment='center', labelpad=15)
    plt.tick_params(labelsize=10)
    ax = plt.subplot(111)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    # labels = plt.xticklabels() + plt.yticklabels()
    # [label.set_fontname('Arial') for label in labels]

    # fig = plt.figure(dpi=300, figsize=(8, 6))
    # ax = plt.Axes(fig, [.1, .12, .25, 0.2])
    # fig.add_axes(ax)
    # colors = '#DC143C'
    # area = np.pi * 2 ** 2
    # ax.scatter(GDP, c_incidence, s=area, c=colors, marker=u'o', alpha=0.5)
    # for i in range(0, len(country)):
    #     ax.text(GDP[i], c_incidence[i], short_name[i], fontsize=6, color="black", weight="light",
    #             horizontalalignment='center', rotation=0, fontname="Arial")
    # font1 = {'family': 'Arial',
    #          'weight': 'normal',
    #          'size': 6,
    #          }
    # ax.set_xlabel(u'GDP loss', font1, verticalalignment='top')
    # ax.set_ylabel(u'Cumulative incidence', font1, verticalalignment='baseline', horizontalalignment='center')
    # plt.tick_params(labelsize=6)
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax1 = plt.Axes(fig, [.43, .12, .25, 0.2])
    # fig.add_axes(ax1)
    # colors = '#DC143C'
    # area = np.pi * 2 ** 2
    # ax1.scatter(GDP, c_incidence_rate, s=area, c=colors, marker=u'o', alpha=0.5)
    # for i in range(0, len(country)):
    #     ax1.text(GDP[i], c_incidence_rate[i], short_name[i], fontsize=6, color="black", weight="light",
    #              horizontalalignment='center', rotation=0, fontname="Arial")
    # ax1.set_xlabel(u'GDP loss', font1, verticalalignment='top')
    # ax1.set_ylabel(u'Cumulative incidence rate', font1, verticalalignment='baseline', horizontalalignment='center')
    # ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    # plt.tick_params(labelsize=6)
    # labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax2 = plt.Axes(fig, [.75, .12, .22, 0.2])
    # fig.add_axes(ax2)
    # colors = '#DC143C'
    # area = np.pi * 2 ** 2
    # ax2.scatter(GDP, MD, s=area, c=colors, marker=u'o', alpha=0.5)
    # for i in range(0, len(country)):
    #     ax2.text(GDP[i], MD[i], short_name[i], fontsize=6, color="black", weight="light",
    #              horizontalalignment='center', rotation=0, fontname="Arial")
    # ax2.set_ylabel(u'Medical expenditure', font1, verticalalignment='baseline', horizontalalignment='center')
    # ax2.set_xlabel(u'GDP loss', font1, verticalalignment='baseline', horizontalalignment='center', labelpad=15)
    # plt.tick_params(labelsize=6)
    # labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax3 = plt.Axes(fig, [.1, .42, .25, 0.2])
    # fig.add_axes(ax3)
    # colors = '#DC143C'
    # area = np.pi * 2 ** 2
    # ax3.scatter(GDP_variation, c_incidence, s=area, c=colors, marker=u'o', alpha=0.5)
    # for i in range(0, len(country)):
    #     ax3.text(GDP_variation[i], c_incidence[i], short_name[i], fontsize=6, color="black", weight="light",
    #              horizontalalignment='center', rotation=0, fontname="Arial")
    # ax3.set_xlabel(u'GDP variation', font1, verticalalignment='top')
    # ax3.set_ylabel(u'Cumulative incidence', font1, verticalalignment='baseline',
    #                horizontalalignment='center')
    # ax3.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    # plt.tick_params(labelsize=6)
    # labels = ax3.get_xticklabels() + ax3.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax4 = plt.Axes(fig, [.43, .42, .25, 0.2])
    # fig.add_axes(ax4)
    # colors = '#DC143C'
    # area = np.pi * 2 ** 2
    # ax4.scatter(GDP_variation, c_incidence_rate, s=area, c=colors, marker=u'o', alpha=0.5)
    # for i in range(0, len(country)):
    #     ax4.text(GDP_variation[i], c_incidence_rate[i], short_name[i], fontsize=6, color="black", weight="light",
    #              horizontalalignment='center', rotation=0, fontname="Arial")
    # ax4.set_xlabel(u'GDP variation', font1, verticalalignment='top')
    # ax4.set_ylabel(u'Cumulative incidence rate', font1, verticalalignment='baseline', horizontalalignment='center')
    # ax4.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    # ax4.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    # plt.tick_params(labelsize=6)
    # labels = ax4.get_xticklabels() + ax4.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax5 = plt.Axes(fig, [.75, .42, .22, 0.2])
    # fig.add_axes(ax5)
    # colors = '#DC143C'
    # area = np.pi * 2 ** 2
    # ax5.scatter(GDP_variation, MD, s=area, c=colors, marker=u'o', alpha=0.5)
    # for i in range(0, len(country)):
    #     ax5.text(GDP_variation[i], MD[i], short_name[i], fontsize=6, color="black", weight="light",
    #              horizontalalignment='center', rotation=0, fontname="Arial")
    # ax5.set_xlabel(u'GDP variation', font1, verticalalignment='top')
    # ax5.set_ylabel(u'Medical expenditure', font1, verticalalignment='baseline', horizontalalignment='center')
    # ax5.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    # plt.tick_params(labelsize=6)
    # labels = ax5.get_xticklabels() + ax5.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax6 = plt.Axes(fig, [.1, .72, .25, 0.2])
    # fig.add_axes(ax6)
    # colors = '#DC143C'
    # area = np.pi * 2 ** 2
    # ax6.scatter(perGDP, c_incidence, s=area, c=colors, marker=u'o', alpha=0.5)
    # for i in range(0, len(country)):
    #     ax6.text(perGDP[i], c_incidence[i], short_name[i], fontsize=6, color="black", weight="light",
    #              horizontalalignment='center', rotation=0, fontname="Arial")
    # ax6.set_xlabel(u'per capital GDP loss', font1, verticalalignment='top')
    # ax6.set_ylabel(u'Cumulative incidence', font1, verticalalignment='baseline',
    #                horizontalalignment='center')
    # plt.tick_params(labelsize=6)
    # labels = ax6.get_xticklabels() + ax6.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax7 = plt.Axes(fig, [.43, .72, .25, 0.2])
    # fig.add_axes(ax7)
    # colors = '#DC143C'
    # area = np.pi * 2 ** 2
    # ax7.scatter(perGDP, c_incidence_rate, s=area, c=colors, marker=u'o', alpha=0.5)
    # for i in range(0, len(country)):
    #     ax7.text(perGDP[i], c_incidence_rate[i], short_name[i], fontsize=6, color="black", weight="light",
    #              horizontalalignment='center', rotation=0, fontname="Arial")
    # ax7.set_xlabel(u'per capital GDP loss', font1, verticalalignment='top')
    # ax7.set_ylabel(u'Cumulative incidence rate', font1, verticalalignment='baseline', horizontalalignment='center')
    # ax7.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    # plt.tick_params(labelsize=6)
    # labels = ax7.get_xticklabels() + ax7.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax8 = plt.Axes(fig, [.75, .72, .22, 0.2])
    # fig.add_axes(ax8)
    # colors = '#DC143C'
    # area = np.pi * 2 ** 2
    # ax8.scatter(perGDP, MD, s=area, c=colors, marker=u'o', alpha=0.5)
    # for i in range(0, len(country)):
    #     ax8.text(perGDP[i], MD[i], short_name[i], fontsize=6, color="black", weight="light",
    #              horizontalalignment='center', rotation=0, fontname="Arial")
    # ax8.set_xlabel(u'per capital GDP loss', font1, verticalalignment='top')
    # ax8.set_ylabel(u'Medical expenditure', font1, verticalalignment='baseline', horizontalalignment='center')
    # plt.tick_params(labelsize=6)
    # labels = ax8.get_xticklabels() + ax8.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]

    plt.show()



def plot_sensi_results():
    filepath_gr = "C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/sensitivity/gr.csv"
    filepath_r0 = "C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/sensitivity/R0.csv"
    filepath_tgen = "C:/Users/Admin/Desktop/实验一-校验实验结果/wuhan/sensitivity/Tgen.csv"

    gr_data = pd.read_csv(filepath_gr)
    R0_data = pd.read_csv(filepath_r0)
    Tgen_data = pd.read_csv(filepath_tgen)
    gr = gr_data["GR"][:].tolist()
    R0 = R0_data["R0"][:].tolist()
    Tgen = Tgen_data["T_gen"][:].tolist()
    x1 = gr_data["x1"][:].tolist()
    x2 = gr_data["x2"][:].tolist()
    x3 = gr_data["x3"][:].tolist()
    x4 = gr_data["x4"][:].tolist()
    x5 = gr_data["x5"][:].tolist()

    colors = '#DC143C'
    area = np.pi * 2 ** 2
    fig = plt.figure(dpi=300, figsize=(8, 4.5))
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 12,
             }

    ax = plt.Axes(fig, [.11, .15, .12, 0.8])
    ax.set_xlim(0.05, 0.055)
    fig.add_axes(ax)
    # ax.scatter(x1, gr, s=area, c=colors, marker=u'o', alpha=0.5)
    # ax.set_ylabel(r'$\dot{C}$', font1)
    ax.scatter(x1, R0, s=area, c=colors, marker=u'o', alpha=0.5)
    ax.set_ylabel(r'${R}_{0}$', font1)
    # ax.scatter(x1, Tgen, s=area, c=colors, marker=u'o', alpha=0.5)
    # ax.set_ylabel(r'${T}_{gen}$', font1)
    ax.set_xlabel(r'${ix}_{1}$', font1)
    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax1 = plt.Axes(fig, [.29, .15, .12, 0.8])
    ax1.set_xlim(0.45, 1)
    fig.add_axes(ax1)
    # ax1.scatter(x2, gr, s=area, c=colors, marker=u'o', alpha=0.5)
    ax1.scatter(x2, R0, s=area, c=colors, marker=u'o', alpha=0.5)
    # ax1.scatter(x2, Tgen, s=area, c=colors, marker=u'o', alpha=0.5)
    ax1.set_xlabel(r'${ix}_{2}$', font1)
    plt.tick_params(labelsize=10)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax2 = plt.Axes(fig, [.47, .15, .12, 0.8])
    ax2.set_xlim(0.945, 1.0)
    fig.add_axes(ax2)
    # ax2.scatter(x3, gr, s=area, c=colors, marker=u'o', alpha=0.5)
    ax2.scatter(x3, R0, s=area, c=colors, marker=u'o', alpha=0.5)
    # ax2.scatter(x3, Tgen, s=area, c=colors, marker=u'o', alpha=0.5)
    ax2.set_xlabel(r'${ix}_{3}$', font1)
    plt.tick_params(labelsize=10)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax3 = plt.Axes(fig, [.65, .15, .12, 0.8])
    ax3.set_xlim(2.5, 7.5)
    fig.add_axes(ax3)
    # ax3.scatter(x4, gr, s=area, c=colors, marker=u'o', alpha=0.5)
    ax3.scatter(x4, R0, s=area, c=colors, marker=u'o', alpha=0.5)
    # ax3.scatter(x4, Tgen, s=area, c=colors, marker=u'o', alpha=0.5)
    ax3.set_xlabel(r'${ix}_{4}$', font1)
    plt.tick_params(labelsize=10)
    labels = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax4 = plt.Axes(fig, [.83, .15, .12, 0.8])
    ax4.set_xlim(5.0, 16.0)
    fig.add_axes(ax4)
    # ax4.scatter(x5, gr, s=area, c=colors, marker=u'o', alpha=0.5)
    ax4.scatter(x5, R0, s=area, c=colors, marker=u'o', alpha=0.5)
    # ax4.scatter(x5, Tgen, s=area, c=colors, marker=u'o', alpha=0.5)
    ax4.set_xlabel(r'${ix}_{5}$', font1)
    plt.tick_params(labelsize=10)
    labels = ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    plt.show()


def economic_costs():
#     # Plot barchart-wuhan
#     x_pos = np.arange(7)
#     relative_loss = [-1.249, -2.529, -2.633, -2.638, -2.937, 2.208, 2.997]
#     relative_loss = np.array(relative_loss)
#     md_loss = [-2.068, -3.98, -5.67, -6.86, -7.85, 5.7, 11.47]
#     rl_err = [0.391, 0.815, 0.833, 0.889, 0.921, 0.771, 1.098]
#     md_err = [0.815, 0.838, 0.746, 0.826, 0.951, 0.995, 1.098]
#     gdp_err = 25.16/6.391
#
#     fig, ax = plt.subplots(dpi=300, figsize=(5, 3))
#     ax.axhline(y=0, ls="-.", c="black", linewidth=1)
#     ax.bar(x_pos, relative_loss, yerr=rl_err, width=0.3, align='center', color='deepskyblue',
#            label='Relative GDP loss variation')
#     bar_width = 0.3
#     ax.bar(x_pos + bar_width, md_loss, yerr=md_err, width=bar_width, align='center', color='mediumspringgreen',
#            label='Relative medical expenditure variation')
#     ax.set_xticks([])
#     ax.set_ylim(-9, 13)
#     plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
#     ax.set_xlabel('Adjusting the timing of lockdown measures', fontname="Arial", fontsize=10)
#     ax.set_ylabel('Relative economic variation owing to \n adjustment of the timing of lockdown', fontname="Arial", fontsize=10)
#
#     ax.set_xlim(min(x_pos) - 1, max(x_pos) + 4)
#     ax.tick_params(labelsize=8)
#     labels = ax.get_xticklabels() + ax.get_yticklabels()
#     [label.set_fontname('Arial') for label in labels]
#     ax.axvspan(xmin=-1, xmax=4.5, facecolor='bisque', alpha=0.5)
#     ax.text(0.1, 0.2, "1 day \n earlier", fontsize=5, color="black", weight="light",
#              horizontalalignment='center', rotation=0, fontname="Arial")
#     ax.text(1.1, 0.2, "2 day \n earlier", fontsize=5, color="black", weight="light",
#             horizontalalignment='center', rotation=0, fontname="Arial")
#     ax.text(2.1, 0.2, "3 day \n earlier", fontsize=5, color="black", weight="light",
#             horizontalalignment='center', rotation=0, fontname="Arial")
#     ax.text(3.1, 0.2, "4 day \n earlier", fontsize=5, color="black", weight="light",
#             horizontalalignment='center', rotation=0, fontname="Arial")
#     ax.text(4.1, 0.2, "5 day \n earlier", fontsize=5, color="black", weight="light",
#             horizontalalignment='center', rotation=0, fontname="Arial")
#     ax.text(0.8, 7.3, "Earlier-than-\n actual control", fontsize=8, color="black", weight="heavy", style="italic",
#             horizontalalignment='center', rotation=0, fontname="Arial")
#
#     ax.axvspan(4.5, 7.5, facecolor='aquamarine', alpha=0.35)
#     ax.text(5.1, -1.5, "1 day \n delay", fontsize=5, color="black", weight="light",
#             horizontalalignment='center', rotation=0, fontname="Arial")
#     ax.text(6.1, -1.5, "2 day \n delay", fontsize=5, color="black", weight="light",
#             horizontalalignment='center', rotation=0, fontname="Arial")
#     ax.text(5.8, -7.7, "Later-than-\n actual control", fontsize=8, color="black", weight="heavy", style="italic",
#             horizontalalignment='center', rotation=0, fontname="Arial")
#     font2 = {'family': 'Arial',
#              'weight': 'normal',
#              'size': 6,
#              }
#     ax.legend(loc='upper left', frameon=False, prop=font2)
#
#     ax1 = ax.twinx()
#     ax1.set_ylim(-145/6.391, 210/6.391)
#     ax1.set_yticks([0, 210/6.391])
#     ax1.bar(8, 162.562/6.391, width=0.8, align='center', color='pink')
#     ax1.bar(9, 176.666/6.391, yerr=gdp_err, width=0.8, align='center', color='pink')
#     ax1.set_ylabel('Estimated economic loss values \n (billion USD $)', fontname="Arial",
#                   fontsize=10)
#     ax1.text(8, 30/6.391, "Direct GDP \n loss", fontsize=6, color="black", weight="light",
#              horizontalalignment='center', rotation=90, fontname="Arial")
#     ax1.text(9, 30/6.391, "Our model  \n prediction loss", fontsize=6, color="black", weight="light",
#              horizontalalignment='center', rotation=90, fontname="Arial")
#     ax1.text(8.5, -4.5, "Estimated \n GDP loss", fontsize=6, color="black", weight="light",
#             horizontalalignment='center', rotation=0, fontname="Arial")
#     ax1.tick_params(labelsize=8)
#     labels2 = ax1.get_xticklabels() + ax1.get_yticklabels()
#     [label.set_fontname('Arial') for label in labels2]
#     ax1.text(9, -125/6.391, "Wuhan", fontsize=6, color="black", weight="heavy",
#              horizontalalignment='center', rotation=0, fontname="Arial")
#     plt.show()

    # # Plot barchart-guangzhou
    # x_pos = np.arange(7)
    # relative_loss = [-0.692, -1.945, -2.702, -3.794, -4.794, 0.587, 1.491]
    # relative_loss = np.array(relative_loss)
    # md_loss = [-2.39, -4.78, -5.62, -6.33, -7.84, 2.46, 7.01]
    # rl_err = [0.254, 0.763, 0.988, 1.337, 1.617, 0.237, 0.514]
    # md_err = [1.04, 1.01, 1.03, 0.985, 0.926, 1.067, 1.118]
    # gdp_err = 10.23/6.391
    #
    # fig, ax = plt.subplots(dpi=300, figsize=(5, 3))
    # ax.axhline(y=0, ls="-.", c="black", linewidth=1)
    # ax.bar(x_pos, relative_loss, yerr=rl_err, width=0.3, align='center', color='deepskyblue', label='Relative GDP loss variation')
    # bar_width = 0.3
    # ax.bar(x_pos + bar_width, md_loss, yerr=md_err, width=bar_width, align='center', color='mediumspringgreen',
    #        label='Relative medical expenditure variation')
    # ax.set_xticks([])
    # ax.set_ylim(-9.0, 9.0)
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # ax.set_xlabel('Adjusting the timing of lockdown measures', fontname="Arial", fontsize=10)
    # ax.set_ylabel('Relative economic variation owing to \n adjustment of the timing of lockdown', fontname="Arial",
    #               fontsize=10)
    #
    # ax.set_xlim(min(x_pos) - 1, max(x_pos) + 4)
    # ax.tick_params(labelsize=8)
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax.axvspan(xmin=-1, xmax=4.5, facecolor='bisque', alpha=0.5)
    # ax.text(0.1, 0.2, "1 day \n earlier", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(1.1, 0.2, "2 day \n earlier", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(2.1, 0.2, "3 day \n earlier", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(3.1, 0.2, "4 day \n earlier", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(4.1, 0.2, "5 day \n earlier", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(0.8, 4.2, "Earlier-than-\n actual control", fontsize=8, color="black", weight="heavy", style="italic",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    #
    # ax.axvspan(4.5, 7.5, facecolor='aquamarine', alpha=0.35)
    # ax.text(5.1, -1.1, "1 day \n delay", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(6.1, -1.1, "2 day \n delay", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(5.8, -7.7, "Later-than-\n actual control", fontsize=8, color="black", weight="heavy", style="italic",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # font2 = {'family': 'Arial',
    #          'weight': 'normal',
    #          'size': 6,
    #          }
    # ax.legend(loc='upper left', frameon=False, prop=font2)
    #
    # ax1 = ax.twinx()
    # ax1.set_ylim(-130/6.391, 130/6.391)
    # ax1.set_yticks([0, 130/6.391])
    # ax1.bar(8, 111.342/6.391, width=0.8, align='center', color='pink')
    # ax1.bar(9, 111.903/6.391, yerr=gdp_err, width=0.8, align='center', color='pink')
    # ax1.set_ylabel('Estimated GDP loss values \n (billion USD $)', fontname="Arial",
    #                fontsize=10)
    # ax1.text(8, 10/6.391, "Direct GDP \n loss", fontsize=6, color="black", weight="light",
    #          horizontalalignment='center', rotation=90, fontname="Arial")
    # ax1.text(9, 10/6.391, "Our model  \n prediction loss", fontsize=6, color="black", weight="light",
    #          horizontalalignment='center', rotation=90, fontname="Arial")
    # ax1.text(8.5, -4.5, "Estimated \n GDP loss", fontsize=6, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax1.tick_params(labelsize=8)
    # labels2 = ax1.get_xticklabels() + ax1.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels2]
    # ax1.text(9, -115/6.391, "Guangzhou", fontsize=6, color="black", weight="heavy",
    #          horizontalalignment='center', rotation=0, fontname="Arial")
    # plt.show()

    # Plot barchart-wenzhou
    x_pos = np.arange(7)
    relative_loss = [-0.915, -1.949, -2.159, -2.434, -2.709, 0.284, 0.705]
    relative_loss = np.array(relative_loss)
    md_loss = [-3.47, -5.16, -7.16, -8.19, -8.84, 7.51, 20.49]
    rl_err = [0.342, 0.576, 0.753, 0.863, 0.913, 0.105, 0.283]
    md_err = [0.904, 0.841, 0.839, 0.967, 0.831, 1.2, 1.187]
    gdp_err = 1.185/6.391

    fig, ax = plt.subplots(dpi=300, figsize=(5.2, 3.2))
    ax.axhline(y=0, ls="-.", c="black", linewidth=1)
    ax.bar(x_pos, relative_loss, yerr=rl_err, width=0.3, align='center', color='deepskyblue', label='Relative GDP loss variation')
    bar_width = 0.3
    ax.bar(x_pos + bar_width, md_loss, yerr=md_err, width=bar_width, align='center', color='mediumspringgreen',
           label='Relative medical expenditure variation')
    ax.set_xticks([])
    ax.set_ylim(-10, 22)
    ax.set_xlabel('Adjusting the timing of lockdown measures', fontname="Arial", fontsize=12, labelpad=10)
    ax.set_ylabel('Relative economic variation owing to \n adjustment of the timing of lockdown', fontname="Arial",
                  fontsize=12)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.set_xlim(min(x_pos) - 1, max(x_pos) + 4)
    ax.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    ax.axvspan(xmin=-1, xmax=4.5, facecolor='bisque', alpha=0.5)
    ax.text(0.1, 0.2, "1 day \n earlier", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(1.1, 0.2, "2 day \n earlier", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(2.1, 0.2, "3 day \n earlier", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(3.1, 0.2, "4 day \n earlier", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(4.1, 0.2, "5 day \n earlier", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(0.8, 13, "Earlier-than-\n actual control", fontsize=8, color="black", weight="heavy", style="italic",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.axvspan(4.5, 7.5, facecolor='aquamarine', alpha=0.35)
    ax.text(5.1, -1.9, "1 day \n delay", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(6.1, -1.9, "2 day \n delay", fontsize=5, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax.text(5.8, -7.5, "Later-than-\n actual control", fontsize=8, color="black", weight="heavy", style="italic",
            horizontalalignment='center', rotation=0, fontname="Arial")
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 6,
             }
    ax.legend(loc='upper left', frameon=False, prop=font2)

    ax1 = ax.twinx()
    ax1.set_ylim(-6.8/6.391, 15/6.391)
    ax1.set_yticks([0, 15/6.391])
    ax1.bar(8, 12.601/6.391, width=0.8, align='center', color='pink')
    ax1.bar(9, 12.858/6.391, yerr=gdp_err, width=0.8, align='center', color='pink')
    ax1.set_ylabel('Estimated GDP loss values \n (billion USD $)', fontname="Arial",
                   fontsize=10)
    ax1.text(8, 1/6.391, "Direct GDP \n loss", fontsize=6, color="black", weight="light",
             horizontalalignment='center', rotation=90, fontname="Arial")
    ax1.text(9, 1/6.391, "Our model  \n prediction loss", fontsize=6, color="black", weight="light",
             horizontalalignment='center', rotation=90, fontname="Arial")
    ax1.text(8.5, -1.9/6.391, "Estimated \n GDP loss", fontsize=6, color="black", weight="light",
            horizontalalignment='center', rotation=0, fontname="Arial")
    ax1.tick_params(labelsize=8)
    labels2 = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Arial') for label in labels2]
    ax1.text(9, -5.8/6.391, "Wenzhou", fontsize=6, color="black", weight="heavy",
             horizontalalignment='center', rotation=0, fontname="Arial")
    plt.show()

    # # Plot barchart-beijing
    # x_pos = np.arange(7)
    # relative_loss = [-0.054, -0.185, -1.096, -1.375, -1.391, 1.07, 2.02]
    # relative_loss = np.array(relative_loss)
    # md_loss = [-1.989, -3.547, -4.333, -5.222, -5.897, 1.097, 4.912]
    # rl_err = [0.021, 0.069, 0.376, 0.418, 0.424, 0.483, 0.708]
    # md_err = [0.794, 0.999, 0.841, 0.98, 0.918, 0.781, 0.801]
    # gdp_err = 21.37/6.391
    #
    # fig, ax = plt.subplots(dpi=300, figsize=(5, 3))
    # ax.axhline(y=0, ls="-.", c="black", linewidth=1)
    # # ax.bar(x_pos[0, len(x_pos)-1], relative_loss, xerr=mu_star_conf_sorted, align='center', ecolor='black',)
    # ax.bar(x_pos, relative_loss, yerr=rl_err, width=0.3, align='center', color='deepskyblue', label='Relative GDP loss variation')
    # bar_width = 0.3
    # ax.bar(x_pos+bar_width, md_loss, yerr=md_err, width=bar_width, align='center', color='mediumspringgreen', label='Relative medical expenditure variation')
    # ax.set_xticks([])
    # ax.set_ylim(-7, 6)
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    # ax.set_xlabel('Adjusting the timing of lockdown measures', fontname="Arial", fontsize=10)
    # ax.set_ylabel('Relative economic variation owing to \n adjustment of the timing of lockdown', fontname="Arial",
    #               fontsize=10)
    #
    # ax.set_xlim(min(x_pos) - 1, max(x_pos) + 4)
    # ax.tick_params(labelsize=8)
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels]
    #
    # ax.axvspan(xmin=-1, xmax=4.5, facecolor='bisque', alpha=0.5)
    # ax.text(0.1, 0.2, "1 day \n earlier", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(1.1, 0.2, "2 day \n earlier", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(2.1, 0.2, "3 day \n earlier", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(3.1, 0.2, "4 day \n earlier", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(4.1, 0.2, "5 day \n earlier", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(0.8, 3, "Earlier-than-\n actual control", fontsize=8, color="black", weight="heavy", style="italic",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    #
    # ax.axvspan(4.5, 7.5, facecolor='aquamarine', alpha=0.35)
    # ax.text(5.1, -0.7, "1 day \n delay", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(6.1, -0.7, "2 day \n delay", fontsize=5, color="black", weight="light",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # ax.text(5.8, -6, "Later-than-\n actual control", fontsize=8, color="black", weight="heavy", style="italic",
    #         horizontalalignment='center', rotation=0, fontname="Arial")
    # font2 = {'family': 'Arial',
    #          'weight': 'normal',
    #          'size': 6,
    #          }
    # ax.legend(loc='upper left', frameon=False, prop=font2)
    #
    # ax1 = ax.twinx()
    # ax1.set_ylim(-233/6.391, 200/6.391)
    # ax1.set_yticks([0, 200/6.391])
    # ax1.bar(8, 165.602/6.391, width=0.8, align='center', color='pink')
    # ax1.bar(9, 160.799/6.391, yerr=gdp_err, width=0.8, align='center', color='pink')
    # ax1.set_ylabel('Estimated GDP loss values \n (billion USD $)', fontname="Arial",
    #                fontsize=10)
    # ax1.text(8, 10/6.391, "Direct GDP \n loss", fontsize=6, color="black", weight="light",
    #          horizontalalignment='center', rotation=90, fontname="Arial")
    # ax1.text(9, 10/6.391, "Our model  \n prediction loss", fontsize=6, color="black", weight="light",
    #          horizontalalignment='center', rotation=90, fontname="Arial")
    # ax1.text(8.5, -32.5 / 6.391, "Estimated \n GDP loss", fontsize=6, color="black", weight="light",
    #      horizontalalignment='center', rotation=0, fontname="Arial")
    # ax1.tick_params(labelsize=10)
    # labels2 = ax1.get_xticklabels() + ax1.get_yticklabels()
    # [label.set_fontname('Arial') for label in labels2]
    # ax1.text(9, -210/6.391, "Beijing", fontsize=6, color="black", weight="heavy",
    #          horizontalalignment='center', rotation=0, fontname="Arial")
    # plt.show()






