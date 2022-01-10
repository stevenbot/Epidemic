# encoding:utf-8
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from scipy import stats
import math
from netCDF4 import Dataset
from scipy.io import loadmat
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd


def GetCO2(path_f):

    CO2_data=np.zeros((20,12))

    with open(path_f, "r") as file:
        line=file.readline()
        row = 0
        while(line != ""):
            line = file.readline()

            col = 0
            for num_f in line.split(",")[1::]:
                CO2_data[row][col]=float(num_f)
                col+=1
            row+=1

    return CO2_data


def GetGDP(path_f):
    global city_name
    df = pd.read_excel(path_f, sheet_name=city_name)
    data = df.values
    GDP_data=data[0:15,1:5]
    GDP_data=GDP_data[:,::-1]
    print(GDP_data)
    return GDP_data


def GetRelationCO2toGDP(CO2_data,GDP_data):
    global a1_GDP,b1_GDP,a2_GDP,b2_GDP,years,city_name
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            # 'color': 'black',
            'size': 14
            }

    GDP_years=GDP_data.shape[0]

    # GDP_16_19_1 = GDP_data[GDP_years - years:GDP_years, 0]
    # GDP_16_19_2 = GDP_data[GDP_years - years:GDP_years, 1]


    GDP_16_19_1_temp =GDP_data[GDP_years-years:GDP_years,0]/10
    GDP_16_19_2 = GDP_data[GDP_years-years:GDP_years,1]/910
    GDP_16_19_1=GDP_16_19_1_temp
    for i in range(years):
        if (i+2020-years)%4 == 0:
            print('特殊年份：',i+2020-years)
            GDP_16_19_1[i]=GDP_16_19_1_temp[i]/91
        else:
            GDP_16_19_1[i] = GDP_16_19_1_temp[i] / 90

    CO2_16_19_1 = np.mean(CO2_data[20-years:20, 0:3], 1)
    CO2_16_19_2 = np.mean(CO2_data[20-years:20, 3:6], 1)
    y_max=max(max(GDP_16_19_1),max(GDP_16_19_2))
    y_min=min(min(GDP_16_19_1),min(GDP_16_19_2))
    x_max=max(max(CO2_16_19_1),max(CO2_16_19_2))
    x_min=min(min(CO2_16_19_1),min(CO2_16_19_2))
    x1=[]
    y1=[]
    for i in range(len(GDP_16_19_1)):
        x1.append([CO2_16_19_1[i]])
        y1.append(GDP_16_19_1[i])
    regr1 = linear_model.LinearRegression()
    regr1.fit(x1, y1)
    slope, intercept, r_value, p_value1, std_err = stats.linregress(GDP_16_19_1, CO2_16_19_1)

    print('以下为GDP和二氧化碳的第一季度拟合结果：GDP=',regr1.coef_[0],"*CO2+",regr1.intercept_,"\t\t\tR^2=",regr1.score(x1, y1),"\t\t\tP值：",p_value1)
    print("Residual sum of squares (残差平方和): \t %.8f" % np.mean(
        (regr1.predict(x1) - y1) ** 2))

    x2 = []
    y2 = []
    for i in range(len(GDP_16_19_2)):
        x2.append([CO2_16_19_2[i]])
        y2.append(GDP_16_19_2[i])
    regr2 = linear_model.LinearRegression()
    regr2.fit(x2, y2)
    slope, intercept, r_value, p_value2, std_err = stats.linregress(GDP_16_19_2, CO2_16_19_2)

    print('以下为GDP和二氧化碳的第二季度拟合结果：GDP=', regr2.coef_[0], "*CO2+", regr2.intercept_, "\t\t\tR^2=", regr2.score(x2, y2),"\t\t\tP值：",p_value2)
    print("Residual sum of squares (残差平方和): \t %.8f" % np.mean(
        (regr2.predict(x2) - y2) ** 2))

    #font_titile = font.copy()
    #font_titile['family'] = "Times New Roman"
    #plt.title(city_name + ":" + str(years) + "年", fontdict=font_titile)
    figure, ax = plt.subplots(1,1,figsize=(8,6))
    ax.set_xlabel(" $\mathregular{CO_2}$ emission (gram carbon/$\mathregular{m^2}$/day) ", fontdict=font)
    ax.set_ylabel("GDP (billion RMB/day)", fontdict=font)

    ax.plot(x1, regr1.predict(x1), color='tab:red', linewidth=3)
    ax.scatter(x1, y1, color='tab:red', label="The first quarter")

    if regr1.intercept_<=0:
        equation = "$\mathregular{G_{first}}$=%.2f$\mathregular{E_{first}}$-%.2f" % (regr1.coef_[0], abs(regr1.intercept_))
    else:
        equation = "$\mathregular{G_{first}}$=%.2f$\mathregular{E_{first}}$+%.2f" % (regr1.coef_[0], regr1.intercept_)
    font['color'] = 'tab:red'
    ax.text(x=x_min, y=y_min + (y_max - y_min) * 0.7, s=equation, ha='left', va='baseline', fontdict=font)
    R_2 = "$\mathregular{R^2}$=%.2f" % (regr1.score(x1, y1))
    ax.text(x=x_min, y=y_min + (y_max - y_min) * 0.6, s=R_2, ha='left', va='baseline', fontdict=font)
    P_value = "P=%f" % (p_value1)
    ax.text(x=x_min, y=y_min + (y_max - y_min) * 0.5, s=P_value, ha='left',
            va='baseline', fontdict=font)

    ax.plot(x2, regr2.predict(x2), color='tab:blue', linewidth=3)
    ax.scatter(x2, y2,color='tab:blue',label="The second quarter")

    if regr2.intercept_<=0:
        equation = "$\mathregular{G_{second}}$=%.2f$\mathregular{E_{second}}$-%.2f" % (regr2.coef_[0], abs(regr2.intercept_))
    else:
        equation = "$\mathregular{G_{second}}$=%.2f$\mathregular{E_{second}}$+%.2f" % (regr2.coef_[0], regr2.intercept_)
    font['color'] = 'tab:blue'


    ax.text(x=x_min+(x_max - x_min) *0.7,
            y=(y_max - y_min) * 0.2 + y_min, s=equation, ha='left',
            va='baseline', fontdict=font)
    R_2 = "$\mathregular{R^2}$=%.2f" % (regr2.score(x2, y2))
    ax.text(x=x_min+(x_max - x_min) *0.7,
            y=(y_max - y_min) * 0.1 + y_min, s=R_2, ha='left',
            va='baseline', fontdict=font)
    P_value = "P=%f" % (p_value2)
    ax.text(x=x_min+(x_max - x_min) *0.7,
            y=(y_max - y_min) * 0 + y_min, s=P_value, ha='left',
            va='baseline', fontdict=font)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.tick_params(labelsize=10)


    font_legend= {'family': 'Times New Roman',
            'weight': 'normal',
            # 'color': 'black',
            'size': 14
            }

    ax.legend(prop=font_legend)
    print("刻度为：", ax.get_yticks()[-1])
    print(len(plt.yticks()))
    plt.subplots_adjust(left=0.11, bottom=0.12, right=0.95, top=0.95,
                        wspace=0.2, hspace=0.3)

    plt.savefig('D:\\桌面\\NC\\figure\\CO2 and GDP 拟合图_' + city_name + '——' + str(years) + '年_end.png', dpi=600)
    plt.savefig('D:\\桌面\\NC\\figure\\CO2 and GDP 拟合图_' + city_name + '——' + str(years) + '年_end.pdf')
    plt.savefig('D:\\桌面\\NC\\figure\\CO2 and GDP 拟合图_' + city_name + '——' + str(years) + '年_end.eps',format='eps',dpi=600)
    #plt.show()

    a1_GDP = regr1.coef_[0]
    b1_GDP = regr1.intercept_
    a2_GDP = regr2.coef_[0]
    b2_GDP = regr2.intercept_



def GetRelationGDPtotime(GDP_f):
    global a1_GDP, b1_GDP, a2_GDP, b2_GDP, years, city_name
    font = {'family': 'SimHei',
            'weight': 'normal',
            # 'color': 'black',
            'size': 16
            }
    GDP_years = GDP_data.shape[0]

    GDP_16_19_1 = GDP_data[GDP_years - years:GDP_years, 0]
    GDP_16_19_2 = GDP_data[GDP_years - years:GDP_years, 1]
    CO2_16_19_1 = np.mean(CO2_data[20 - years:20, 0:3], 1)
    CO2_16_19_2 = np.mean(CO2_data[20 - years:20, 3:6], 1)

    time_f = np.array(range(2020 - years, 2020))
    regr1 = linear_model.LinearRegression()
    x1=time_f.reshape(len(time_f),1)
    y1=np.array(GDP_16_19_1).reshape(len(GDP_16_19_1),1)
    regr1.fit(x1, y1)
    slope, intercept, r_value, p_value1, std_err = stats.linregress(x1.reshape(x1.shape[0]), y1.reshape(y1.shape[0]))

    regr2 = linear_model.LinearRegression()
    x2 = time_f.reshape(len(time_f), 1)
    y2 = np.array(GDP_16_19_2).reshape(len(GDP_16_19_2), 1)
    regr2.fit(x2, y2)
    slope, intercept, r_value, p_value2, std_err = stats.linregress(x2.reshape(x2.shape[0]), y2.reshape(y2.shape[0]))

    font_titile = font.copy()
    font_titile['family'] = "SimHei"
    plt.title(city_name + ":" + str(years) + "年", fontdict=font_titile)

    plt.plot(x1, regr1.predict(x1), color='tab:red', linewidth=3)
    plt.scatter(x1, y1, color='tab:red', label="第一季度")

    equation = "$\mathregular{W_j}$=%.2f$\mathregular{\Omega_j}$+%.2f" % (regr1.coef_[0], regr1.intercept_)
    font['color'] = 'tab:red'
    plt.text(x=(plt.xticks()[0][1] + plt.xticks()[0][-1]) / 2,
             y=(plt.yticks()[0][-1] - plt.yticks()[0][0]) / 2 * 0.4 + plt.yticks()[0][0], s=equation, ha='left',
             va='baseline', fontdict=font)
    R_2 = "$\mathregular{R^2}$=%.2f" % (regr1.score(x1, y1))
    plt.text(x=(plt.xticks()[0][1] + plt.xticks()[0][-1]) / 2,
             y=(plt.yticks()[0][-1] - plt.yticks()[0][0]) / 2 * 0.2 + plt.yticks()[0][0], s=R_2, ha='left',
             va='baseline', fontdict=font)
    P_value = "P=%f" % (p_value1)
    plt.text(x=(plt.xticks()[0][1] + plt.xticks()[0][-1]) / 2,
             y=(plt.yticks()[0][-1] - plt.yticks()[0][0]) / 2 * 0.0 + plt.yticks()[0][0], s=P_value, ha='left',
             va='baseline', fontdict=font)

    plt.plot(x2, regr2.predict(x2), color='tab:blue', linewidth=3)
    plt.scatter(x2, y2, color='tab:blue', label="第二季度")
    plt.xlabel(" time(year) ")
    plt.ylabel("GDP(quarter)")

    equation = "$\mathregular{W_j}$=%.2f$\mathregular{\Omega_j}$+%.2f" % (regr2.coef_[0], regr2.intercept_)
    font['color'] = 'tab:blue'
    plt.text(x=plt.xticks()[0][1], y=(plt.yticks()[0][-1] * 1.5 + plt.yticks()[0][0]) / 2, s=equation, ha='left',
             va='baseline', fontdict=font)
    R_2 = "$\mathregular{R^2}$=%.2f" % (regr2.score(x2, y2))
    plt.text(x=plt.xticks()[0][1], y=(plt.yticks()[0][-1] * 1.3 + plt.yticks()[0][0]) / 2, s=R_2, ha='left',
             va='baseline', fontdict=font)
    P_value = "P=%f" % (p_value2)
    plt.text(x=plt.xticks()[0][1], y=(plt.yticks()[0][-1] * 1.1 + plt.yticks()[0][0]) / 2, s=P_value, ha='left',
             va='baseline', fontdict=font)

    font_legend = {'family': 'SimHei',
                   'weight': 'normal',
                   # 'color': 'black',
                   'size': 16
                   }

    plt.legend(prop=font_legend)

    plt.savefig('D:\\桌面\\NC\\figure\\GDP and time 拟合图_'+city_name+'——'+str(years)+'年2c.jpg', dpi=600)
    plt.show()


def Detrended_and_predict_CO2(co2):
    global years
    co2_16_19=co2[20-years:20,:]
    x=[]
    for i in range(years):
        num_f=2019-i
        x.append([num_f])
    x.reverse()
    slope=[]
    co2_20=[]
    for i in range(12):
        y=co2_16_19[:,i]
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        slope.append(regr.coef_[0])
        co2_20.append(regr.predict([[2020]]))

        # print('2016-2019CO2浓度拟合(斜率): \t', regr.coef_[0])
        # print('2016-2019CO2浓度拟合(拟合优度): \t', regr.score(x, y))

    detrendedco2=np.zeros(co2.shape)
    for i in range(co2.shape[0]):
        for j in range(co2.shape[1]):
            detrendedco2[i][j]=co2[i][j]+slope[j]*(20-i)

    return detrendedco2,co2_20

def GetCCFAdd():
    global city_name
    path = 'D:\\桌面\\NC\\data\\不同城市及省份CCF.xlsx'
    CCF={}
    data_all=[]

    df = pd.read_excel(path, sheet_name=city_name)

    CCF_value = df.values[::, 2:4]
    for i in range(CCF_value.shape[0]):
        if ~np.isnan(CCF_value[i,1]) :
            data_all.append((int(CCF_value[i,0]),float(CCF_value[i,1])))

    day= data_all[0][0]
    last_temp=0
    for temp in data_all:
        if (temp[0]==day):
            CCF[temp[0]]=temp[1]
            day+=1
        else:
            while day <= temp[0]:
                CCF[day]=last_temp[1]+(day-last_temp[0])*(temp[1]-last_temp[1])/(temp[0]-last_temp[0])
                day+=1
        last_temp=temp
    return CCF

def GetDetaCO2(detrendedco2,CCF):
    const1 = np.mean(detrendedco2[16:20, 1]) * detrendedco2[19, 11] / np.mean(detrendedco2[15:19, 11])  # -23到6为1月
    const2 = np.mean(detrendedco2[16:20, 2]) * detrendedco2[19, 11] / np.mean(detrendedco2[15:19, 11])  # 7到35为2月
    const3 = np.mean(detrendedco2[16:20, 3]) * detrendedco2[19, 11] / np.mean(detrendedco2[15:19, 11])  # 36到66为3月
    const4 = np.mean(detrendedco2[16:20, 4]) * detrendedco2[19, 11] / np.mean(detrendedco2[15:19, 11])  # 67到96为4月
    const5 = np.mean(detrendedco2[16:20, 5]) * detrendedco2[19, 11] / np.mean(detrendedco2[15:19, 11])  # 97到120为5月

    # const1 = detrendedco2[19 ,11] / np.mean(detrendedco2[15:19, 11])  #-23到6为1月
    # const2 = detrendedco2[19, 11] / np.mean(detrendedco2[15:19, 11])  #7到35为2月
    # const3 = detrendedco2[19, 11] / np.mean(detrendedco2[15:19, 11])  #36到66为3月
    # const4 = detrendedco2[19, 11] / np.mean(detrendedco2[15:19, 11])  #67到96为4月
    # const5 = detrendedco2[19, 11] / np.mean(detrendedco2[15:19, 11])  #97到120为5月
    detaco2={}

    for day in CCF.keys():
        if day<=6:
            detaco2[day] = const1 * (CCF[day]-1)
        elif day <=35:
            detaco2[day] = const2 * (CCF[day] - 1)
        elif day <=66:
            detaco2[day] = const3 * (CCF[day] - 1)
        elif day <=96:
            detaco2[day] = const4 * (CCF[day] - 1)
        elif day <=120:
            detaco2[day] = const5 * (CCF[day] - 1)
        else:
            print('获取二氧化碳释放差异时出错！！')
            exit(-1)

    return detaco2

def GetRelationCO2toIll(CO2,Ill):
    global a_CO2_redu,b_CO2_redu,total_121, years, city_name,begin_day
    font = {'family': 'Arial',
            'weight': 'normal',
            'color': 'black',
            'size': 16
            }

    print("第一季度二氧化碳",20-years,"到19：",np.mean(CO2[20-years:20,0:3],1))
    print("第二季度二氧化碳",20-years,"到19：", np.mean(CO2[20-years:20, 3:6], 1))
    detrendedco2, co2_20 = Detrended_and_predict_CO2(CO2)
    total_121 = co2_20[0] * 31 + co2_20[1] * 29 + co2_20[2] * 31 + co2_20[3] * 30

    CCF = GetCCFAdd()
    detaco2 = GetDetaCO2(detrendedco2, CCF)

    temp_sum = 0
    detaco2_new = {}
    for temp in detaco2.keys():
        temp_sum += detaco2[temp]
        detaco2_new[temp] = temp_sum / total_121
    detaco2 = detaco2_new




    print("真实数据病人处理后：", Ill)
    print("真实二氧化碳处理后：", CCF)
    x = []
    y = []
    x2 = []
    y2 = []

    for i in range(Ill.shape[1]):
        key=i+begin_day
        if key >120or key<-24:
            break
        xishu=1
        if city_name =="温州":
            xishu=2
            if  (key not in detaco2) or  Ill[0,i]==0 or Ill[0,i]>-2 or Ill[0,i]<-3.80: # or key==104):  # Ill[i] == 0 or i<-11 or i>55):

                continue
        else:
            if  (key not in detaco2) or  Ill[0,i]==0 or Ill[0,i]>-2.6: # or key==104):  # Ill[i] == 0 or i<-11 or i>55):

                continue

        x.append(-detaco2[key]*xishu)
        y.append(Ill[0,i])

        x2.append([Ill[0,i]])
        y2.append(-detaco2[key]*xishu)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(x).reshape(len(x)), np.array(y).reshape(len(y)))
    regr2 = linear_model.LinearRegression()
    regr2.fit(x2, y2)
    a_CO2_redu = regr2.coef_[0]
    b_CO2_redu = regr2.intercept_

    print('以下为感染病例和二氧化碳的拟合结果：')
    print('Coefficients                 (斜率): \t', regr.coef_[0],"slope:",slope)
    print("Intercept                    (截距): \t", regr.intercept_, "intercept:", intercept)
    print('Score                     (拟合优度): \t', regr.score(x, y), "r^2:",r_value)
    print('P                             (P值): \t', p_value)
    # The mean square error
    print("Residual sum of squares (残差平方和): \t %.8f" % np.mean(
        (regr.predict(x) - y) ** 2))
    x_min=min(x)
    y_min=min(y)
    x_max=max(x)
    y_max=max(y)

    figure, ax = plt.subplots(1,1,figsize=(8,6))
    ax.scatter(list(np.log10(x)), list(np.log10(y)), marker='^', color="", edgecolors="blue", s=64)
    ax.plot(x, regr.predict(x), color='red', linewidth=3)
    sns.regplot(x, y, color='tab:red', ci=95, scatter=False, ax=ax)
    ax.scatter(x, y, edgecolors='black', color='')
    ax.set_xlabel("Indicator of $\mathregular{CO_2}$ emission ($\mathregular{\it{Z}_{j,4}}$)",
                  fontdict=font)
    ax.set_ylabel("Indicator of new case in COVID-19 ($\mathregular{\it{W}_{j,4}}$)", fontdict=font)
    # plt.axis([-0.01, 0.25, -5, -2.5])
    if regr.intercept_ >= 0:
        equation = "$\mathregular{\it{W}_{j,4}}$= %.2f·$\mathregular{\it{Z}_{j,4}}$ + %.2f" % (regr.coef_[0], regr.intercept_)
    else:
        equation = "$\mathregular{\it{W}_{j,4}}$= %.2f·$\mathregular{\it{Z}_{j,4}}$ - %.2f" % (
        regr.coef_[0], abs(regr.intercept_))

    para_x = 0.05
    para_y1 = 0.12
    para_y2 = 0.02
    para_x_cityname=0.46
    text_city_name=""
    if city_name == "武汉":
        text_city_name="Wuhan"
    elif city_name == "广州":
        text_city_name = "Guangzhou"
        para_x_cityname = 0.43
    elif city_name == "温州":
        text_city_name = "Wenzhou"
    elif city_name == "北京":
        text_city_name = "Beijing"


    ax.text(x=x_min + (x_max - x_min) * para_x_cityname, y=y_min + (y_max - y_min) * 0.93, s=text_city_name, ha='left', va='baseline', fontdict=font,fontweight='bold',)

    ax.text(x=x_min+(x_max-x_min)*para_x, y=y_min+(y_max-y_min)*para_y1, s=equation, ha='left', va='baseline', fontdict=font)
    R_2 = "$\mathregular{\it{R}^2}$= %.2f" % (regr.score(x, y))
    ax.text(x=x_min+(x_max-x_min)*para_x, y=y_min+(y_max-y_min)*para_y2, s=R_2, ha='left', va='baseline', fontdict=font)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    ax.tick_params(labelsize=14)

    plt.subplots_adjust(left=0.14, bottom=0.13, right=0.95, top=0.95,
                        wspace=0.2, hspace=0.3)

    plt.savefig('D:\\桌面\\NC\\图\\CO2 and covid-19 拟合图_' + city_name + '_end.png', dpi=300)
    plt.savefig('D:\\桌面\\NC\\图\\CO2 and covid-19 拟合图_' + city_name + '_end.ps',dpi=300)
    plt.savefig('D:\\桌面\\NC\\图\\CO2 and covid-19 拟合图_' + city_name + '_end.svg',dpi=300)
    plt.show()


def GetIllDataForEvaluation(path):
    global city_name
    data_clear = []
    flage=False

    df = pd.read_excel(path, sheet_name=city_name)
    Ill = df.values[::, 7]


    for i in range(Ill.shape[0]):
        if ~np.isnan(Ill[i] ):
            data_clear.append(Ill[i])

    temp = np.zeros([1, len(data_clear)])
    temp[0, :] = np.array(data_clear)
    return temp

def Move7WindowLog(increase):
    global gama
    gama=1

    col=increase.shape[1]

    for i in range(1):
        increase2 = np.zeros([1,col + 7])
        increase2[0,4:col + 4] = increase
        increase2[0,0:4]=np.ones([1,4])*increase[0,0]
        increase2[0,col + 4::]=np.ones([1,3])*increase[0,-1]
        move7 = []

        for i in range(3, col + 4):
            move7.append(np.mean(increase2[0,i - 3:i + 4]))
        move7 = np.array(move7).reshape([1,col+1])
        increase = move7[0,1:col+1].reshape([1,col])

    increase=move7
    move7log=np.zeros([1,col])


    for num in range(col):
        if (increase[0, num+1]==0 and increase[0, num] == 0 ):
            move7log[0, num] =0
        else:
            move7log[0,num] = math.log((increase[0, num+1]+gama) / ( (num+1)*(increase[0, num]+gama)))
    return move7log



if __name__=="__main__":
    global years, begin_day, city_name
    city_name = "广州"
    years = 4


    if city_name=="北京":
        begin_day = 4
        path_co2 = "D:\\桌面\\NC\\data\\beijing-co2-2c.csv"
    elif city_name=="天津":
        begin_day=-1
        path_co2="D:\\桌面\\NC\\data\\tianjin-co2-2c.csv"
    elif city_name=="重庆":
        begin_day=-3
        path_co2="D:\\桌面\\NC\\data\\chongqing-co2-concentration.csv"
    elif city_name == "广州":
        begin_day = 4
        path_co2 = "D:\\桌面\\NC\\data\\guanggzhou-co2.csv"
    elif city_name =="温州":
        begin_day = 12
        path_co2 = "D:\\桌面\\NC\\data\\wenzhou-co2-4.csv"
    elif city_name=='武汉':
        begin_day=-4
        path_co2= "D:\\桌面\\NC\\data\\wuhan-co2-concentration.csv"
    else:
        print("城市选择错误！")
        exit(0)




    path_GDP="D:\\桌面\\NC\\data\\GDP-new.xlsx"
    path_Ill="D:\\桌面\\NC\\data\\疫情-new.xlsx"

    CO2_data=GetCO2(path_co2)
    GDP_data=GetGDP(path_GDP)

    GetRelationCO2toGDP(CO2_data,GDP_data)
    Ill_1 = GetIllDataForEvaluation(path_Ill)
    Ill = Move7WindowLog(Ill_1)

    GetRelationCO2toIll(CO2_data,Ill)