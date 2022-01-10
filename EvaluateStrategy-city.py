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
    # print(GDP_data)
    return GDP_data

def GetRelationCO2toGDP(CO2_data,GDP_data):
    global a1_GDP,b1_GDP,a2_GDP,b2_GDP,years,city_name
    font = {'family': 'Times new roman',
            'weight': 'normal',
            # 'color': 'black',
            'size': 16
            }
    GDP_years=GDP_data.shape[0]

    GDP_16_19_1 = GDP_data[11:15, 0] / 90
    GDP_16_19_1[0] = GDP_data[11, 0] / 91
    GDP_16_19_2 = GDP_data[11:15, 1] / 91  # [32.68, 36.92, 41.64, 45.29]

    CO2_16_19_1 = np.mean(CO2_data[20-years:20, 0:3], 1)
    CO2_16_19_2 = np.mean(CO2_data[20-years:20, 3:6], 1)

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

    plt.figure(6)
    font_titile = font.copy()
    font_titile['family'] = "SimHei"
    plt.title(city_name + ":" + str(years) + "年", fontdict=font_titile)


    x_min=min(min(x1))
    x_max=max(max(x1))
    y_min=min(y1)
    y_max=max(y1)

    plt.plot(x1, regr1.predict(x1), color='tab:red', linewidth=3)
    plt.scatter(x1, y1, color='tab:red', label="第一季度")

    equation = "$\mathregular{W_j}$=%.2f$\mathregular{\Omega_j}$+%.2f" % (regr1.coef_[0], regr1.intercept_)
    font['color'] = 'tab:red'


    plt.text(x= x_min,
            y=(y_max - y_min) * 0.7 + y_min, s=equation, ha='left',
            va='baseline', fontdict=font)
    R_2 = "$\mathregular{R^2}$=%.2f" % (regr2.score(x2, y2))
    plt.text(x= x_min,
            y=(y_max - y_min) * 0.6 + y_min, s=R_2, ha='left',
            va='baseline', fontdict=font)
    P_value = "P=%f" % (p_value2)
    plt.text(x= x_min,
            y=(y_max - y_min) * 0.5 + y_min, s=P_value, ha='left',
            va='baseline', fontdict=font)

    x_min = min(min(x2))
    x_max = max(max(x2))
    y_min = min(y2)
    y_max = max(y2)

    plt.plot(x2, regr2.predict(x2), color='tab:blue', linewidth=3)
    plt.scatter(x2, y2,color='tab:blue',label="第二季度")
    plt.xlabel(" $\mathregular{CO_2}$ emission ")
    plt.ylabel("GDP everyday")

    equation = "$\mathregular{W_j}$=%.2f$\mathregular{\Omega_j}$+%.2f" % (regr2.coef_[0], regr2.intercept_)
    font['color'] = 'tab:blue'
    plt.text(x=(x_max + x_min) * 0.499,
             y=(y_max - y_min) * 0.2 + y_min, s=equation, ha='left',
             va='baseline', fontdict=font)
    R_2 = "$\mathregular{R^2}$=%.2f" % (regr2.score(x2, y2))
    plt.text(x=(x_max + x_min) * 0.499,
             y=(y_max - y_min) * 0.1 + y_min, s=R_2, ha='left',
             va='baseline', fontdict=font)
    P_value = "P=%f" % (p_value2)
    plt.text(x=(x_max + x_min) * 0.499,
             y=(y_max - y_min) * 0 + y_min, s=P_value, ha='left',
             va='baseline', fontdict=font)


    font_legend= {'family': 'SimHei',
            'weight': 'normal',
            # 'color': 'black',
            'size': 16
            }



    plt.legend(prop=font_legend)
    a1_GDP = regr1.coef_[0]
    b1_GDP = regr1.intercept_
    a2_GDP = regr2.coef_[0]
    b2_GDP = regr2.intercept_




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

    detrendedco2=np.zeros(co2.shape)
    for i in range(co2.shape[0]):
        for j in range(co2.shape[1]):
            detrendedco2[i][j]=co2[i][j]+slope[j]*(20-i)

    return detrendedco2,co2_20


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


def GetIllDataForEvaluation(path):
    global city_name
    data_clear = []
    flage=False

    df = pd.read_excel(path, sheet_name=city_name)
    Ill = df.values[::, 7]
    #Ill.reshape(1,Ill.shape[])

    for i in range(Ill.shape[0]):
        if ~np.isnan(Ill[i] ):
            data_clear.append(Ill[i])

    temp = np.zeros([1, len(data_clear)])
    temp[0, :] = np.array(data_clear)
    return temp


def GetSimulationIllData(path,row):
    global city_name
    data_clear = []
    flage=False

    df = pd.read_excel(path, sheet_name="Sheet1")
    Ill = df.values[::, row]
    #Ill.reshape(1,Ill.shape[])

    for i in range(Ill.shape[0]):
        if ~np.isnan(Ill[i]):
            data_clear.append(Ill[i])

    data_clear_new=[]
    for j in range(len(data_clear)):
        if j == 0:
            data_clear_new.append(data_clear[j])
        else:
            data_clear_new.append(data_clear[j]-data_clear[j-1])

    temp = np.zeros([1, len(data_clear_new)])
    temp[0, :] = np.array(data_clear_new)
    return temp

def Move7WindowLog(increase):
    global gama
    gama=1
    col=increase.shape[1]
    move7 = []
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




def GetTreatCost(infect,city):
    global Tr_draw

    infect_acc=[]
    infect_acc.append(infect[0,0])
    for i in range(infect.shape[1]):
        if i >0:
            infect_acc.append(infect_acc[-1]+infect[0,i])
    infect_acc=np.array(infect_acc).reshape(infect.shape)
    mortality_rate = 0.0201
    infect_acc=np.mean(infect_acc,0)
    death_acc=infect_acc*mortality_rate

    infect_num=max(infect_acc)
    death_num=max(death_acc)



    cost_cured=(0.9*14.7+1.2*14.7+8.1*17+17*17+19.2*20.6+22.4*20.6+19.2*19.2+8.8*19.2+3.2*15.4)*6.5478*1000/100
    cost_fatal=(0.1*15.3+0.7*16.1+1.8*15.8+3.7*13.8+12.7*10.3+30.2*6.7+30.5*3.7+20.3*1.5)*6.5478*1000000/100

    Tr_draw[city]=[]
    for i in range(len(death_acc)):
        cost=-(cost_fatal*death_acc[i]+cost_cured*(infect_acc[i]-death_acc[i]))/100000000
        Tr_draw[city].append(cost)

    cost_all=(death_num*cost_fatal+(infect_num-death_num)*cost_cured)/100000000
    return cost_all

def GetDataIllMove7Rate():
    path='D:\\桌面\\NC\\数据\\真实病例数据\\use_this_武汉新冠医学院数据.csv'
    data=[]
    data_clear=[]
    springdate=0
    order = 0
    with open(path,'r') as file:
        lines=file.readlines()
        for line in lines[1::]:
            line_split=line.split(',')
            if int(line_split[8])==0:
                data.append((line_split[1], 0))
                data_clear.append(0)
            else:
                data.append((line_split[1],int(line_split[8])))
                data_clear.append((int(line_split[8])))

            if(line_split[1]=='2020/1/25'):
                springdate=order

            order+=1
    data_clear=np.array(data_clear)
    data_clear2=np.zeros([data_clear.shape[0]+6])
    data_clear2[3:data_clear.shape[0]+3]=data_clear
    data_clear=data_clear2
    data_part1= {}
    for i in range(3,data_clear.shape[0]-3):
            data_part1[i - springdate-3] = np.mean(data_clear[i - 3:i + 4])

    return data_part1



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



def GetCCFAdd():
    global city_name
    path = 'D:\\桌面\\NC\\data\\不同城市及省份CCF.xlsx'
    CCF = {}
    data_all = []

    df = pd.read_excel(path, sheet_name=city_name)

    CCF_value = df.values[::, 2:4]
    for i in range(CCF_value.shape[0]):
        if ~np.isnan(CCF_value[i, 1]):
            data_all.append((int(CCF_value[i, 0]), float(CCF_value[i, 1])))

    day = data_all[0][0]
    last_temp = 0
    for temp in data_all:
        if (temp[0] == day):
            CCF[temp[0]] = temp[1]
            day += 1
        else:
            while day <= temp[0]:
                CCF[day] = last_temp[1] + (day - last_temp[0]) * (temp[1] - last_temp[1]) / (temp[0] - last_temp[0])
                day += 1
        last_temp = temp
    return CCF




def GetRelationIlltoCO2(Ill,CO2):
    global a_CO2_redu,b_CO2_redu,total_121, years, city_name,begin_day
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'color': 'black',
            'size': 16
            }

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

    x = []
    y = []
    x2 = []
    y2 = []

    for i in range(Ill.shape[1]):
        key=i+begin_day
        if key >120or key<-24:
            break

        if  (key not in detaco2) or  Ill[0,i]==0 or Ill[0,i]>-2: # or key==104):  # Ill[i] == 0 or i<-11 or i>55):
            continue

        x.append(-detaco2[key])
        y.append(Ill[0,i])

        x2.append([Ill[0,i]])
        y2.append(-detaco2[key])

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(x).reshape(len(x)), np.array(y).reshape(len(y)))
    regr2 = linear_model.LinearRegression()
    regr2.fit(x2, y2)
    a_CO2_redu = regr2.coef_[0]
    b_CO2_redu = regr2.intercept_

    x_min=min(min(x))
    x_max=max(max(x))
    y_min=min(y)
    y_max=max(y)

    print('以下为感染病例和二氧化碳的拟合结果：')
    print('Coefficients                 (斜率): \t', regr.coef_[0],"slope:",slope)
    print("Intercept                    (截距): \t", regr.intercept_, "intercept:", intercept)
    print('Score                     (拟合优度): \t', regr.score(x, y), "r^2:",r_value)
    print('P                             (P值): \t', p_value)

    print("Residual sum of squares (残差平方和): \t %.8f" % np.mean(
        (regr.predict(x) - y) ** 2))

    plt.figure(66)

    plt.plot(x, regr.predict(x), color='red', linewidth=3)
    sns.regplot(x,y,color='tab:red',ci=95,scatter=False)
    plt.scatter(x, y,edgecolors='black',color='')
    font_titile=font.copy()
    font_titile['family']="SimHei"
    plt.title(city_name,fontdict=font_titile)
    plt.xlabel("Indicator of $\mathregular{CO_2}$ emission reduction ($\mathregular{\Omega_j}$)", fontdict=font)
    plt.ylabel("Indicator of new case in COVID-19 ($\mathregular{W_j}$)", fontdict=font)


    if regr.intercept_ >= 0:
        equation = "$\mathregular{W_j}$= %.2f $\mathregular{\Omega_j}$ + %.2f" % (regr.coef_[0], regr.intercept_)
    else:
        equation = "$\mathregular{W_j}$= %.2f $\mathregular{\Omega_j}$ - %.2f" % (
        regr.coef_[0], abs(regr.intercept_))

    # equation=r'$s(t) = \mathcal{A}\mathrm{sin}(2 \omega t)$'
    font['color']='tab:red'
    plt.text(x=x_min,y=y_min+(y_max-y_min)*0.2,s=equation,ha='left',va='baseline',fontdict=font)
    R_2="$\mathregular{R^2}$=%.2f"%(regr.score(x,y))
    plt.text(x=x_min,y=y_min+(y_max-y_min)*0.1,s=R_2,ha='left',va='baseline',fontdict=font)
    plt.tight_layout()
    #plt.savefig('D:\\桌面\\NC\\figure\\CO2 and covid-19 拟合图_'+city_name+'2c.jpg', dpi=600)
    #plt.show()



def GetCO2ReductionEmission(move7log):
    global a_CO2_redu,b_CO2_redu
    co2_v = [0]
    for i in range(move7log.shape[0]):
        if move7log[i] == 0:
            co2_v.append(co2_v[-1])
            continue
        co2_v.append(-(move7log[i] * a_CO2_redu + b_CO2_redu))
    co2_v = np.array(co2_v)
    return co2_v[1::]






def GetGDPLoss(infection_all,city,evaluation_day,eva_len=5):
    global a_CO2_redu,b_CO2_redu,a1_GDP,b1_GDP,a2_GDP,b2_GDP,total_121,G_draw
    len=infection_all.shape[1]
    move7log = Move7WindowLog(np.mean(infection_all, 0).reshape([1,len]))

    co2_v=GetCO2ReductionEmission(move7log[0,:])

    sum_deta_co2_v=co2_v*total_121
    data_co2_v=[]
    data_co2_v.append(sum_deta_co2_v[0])
    for i in range(1,sum_deta_co2_v.shape[0]):
        data_co2_v.append(sum_deta_co2_v[i]-sum_deta_co2_v[i-1])

    GDP_loss = 0
    order=1
    G_draw[city]=[]

    for deta_temp in data_co2_v:
        if deta_temp == 0 or deta_temp>200 :#or deta_temp<-9.5 :
            G_draw[city].append(float(GDP_loss))
            continue
        if order<=91:
            GDP_loss+=deta_temp*a1_GDP
        else:
            GDP_loss += deta_temp * a2_GDP
        G_draw[city].append(float(GDP_loss))

    GDP_loss=sum(data_co2_v[evaluation_day-eva_len:evaluation_day+eva_len])* a2_GDP/(2*eva_len)


    return -1*GDP_loss[0]

def Predict2020GDP( GDP_data):

    GDP_16_19_1=GDP_data[11:15,0]/90
    GDP_16_19_1[0]=GDP_data[11,0]/91
    GDP_16_19_2 = GDP_data[11:15,1]/91 #[32.68, 36.92, 41.64, 45.29]

    time=[[2016],[2017],[2018],[2019]]
    regr1 = linear_model.LinearRegression()
    regr1.fit(time, GDP_16_19_1)

    print('以下为年份和GDP的第一季度拟合结果：GDP=', regr1.coef_[0], "*t+", regr1.intercept_, "\t\t\t拟合优度：", regr1.score(time, GDP_16_19_1))
    print("Residual sum of squares (残差平方和): \t %.8f" % np.mean(
        (regr1.predict(time) - GDP_16_19_1) ** 2))

    regr2 = linear_model.LinearRegression()
    regr2.fit(time, GDP_16_19_2)
    print('以下为年份和GDP的第二季度拟合结果：GDP=', regr2.coef_[0], "*t+", regr2.intercept_, "\t\t\t拟合优度：",
          regr2.score(time, GDP_16_19_2))
    print("Residual sum of squares (残差平方和): \t %.8f" % np.mean(
        (regr2.predict(time) - GDP_16_19_2) ** 2))

    GDP_2020_1to4=regr1.predict([[2020]])*91+regr2.predict([[2020]])*30
    return GDP_2020_1to4

def GetReal2020GDP(city_name):
    real_2020_f=0
    if city_name == "武汉":
        first_quarter=2257.30
        April=float(4038)/91*30
        real_2020_f=first_quarter+April
    elif city_name == "广州":
        first_quarter=5228.8
        April=float(5739.49)/91*30
        real_2020_f=first_quarter+April
    elif city_name == "北京":
        first_quarter=7462.2
        April=float(8743.4)/91*30
        real_2020_f=first_quarter+April
    elif city_name == "温州":
        first_quarter=1352.66
        April=float(1724.34)/91*30
        real_2020_f=first_quarter+April
    else:
        print("GetReal2020GDP: city name wrong!")
        exit(0)
    return real_2020_f

def GetGDPLossAll(infection_all,country,city_name  ):
    global a_CO2_redu,b_CO2_redu,a1_GDP,b1_GDP,a2_GDP,b2_GDP,total_121,G_draw
    len=infection_all.shape[1]
    move7log = Move7WindowLog(np.mean(infection_all, 0).reshape([1,len]))

    co2_v=GetCO2ReductionEmission(move7log[0,:])

    sum_deta_co2_v=co2_v*total_121
    data_co2_v=[]
    data_co2_v.append(sum_deta_co2_v[0])
    for i in range(1,sum_deta_co2_v.shape[0]):
        data_co2_v.append(sum_deta_co2_v[i]-sum_deta_co2_v[i-1])

    GDP_loss = 0
    order=1
    G_draw[country]=[]
    if city_name == "武汉":
        for deta_temp in data_co2_v:
            if deta_temp == 0 or deta_temp>1.8 or deta_temp<-7 :
                G_draw[country].append(float(GDP_loss))
                continue
            if order<=91:
                GDP_loss+=deta_temp*a1_GDP
                order+=1
            else:
                GDP_loss += deta_temp * a2_GDP
                order+=1
            G_draw[country].append(float(GDP_loss))
    elif city_name == "广州":
        for deta_temp in data_co2_v:
            if deta_temp == 0 or deta_temp > 1.8 or deta_temp < -7:
                G_draw[country].append(float(GDP_loss))
                continue
            if order <= 91:
                GDP_loss += deta_temp * a1_GDP
                order += 1
            else:
                GDP_loss += deta_temp * a2_GDP
                order += 1
            G_draw[country].append(float(GDP_loss))
    elif city_name == "北京":
        for deta_temp in data_co2_v:
            if deta_temp == 0 or deta_temp > 3 or deta_temp < -4.38:
                G_draw[country].append(float(GDP_loss))
                continue
            if order <= 91:
                GDP_loss += deta_temp * a1_GDP
                order += 1
            else:
                GDP_loss += deta_temp * a2_GDP
                order += 1
            G_draw[country].append(float(GDP_loss))
    elif city_name == "温州":
        for deta_temp in data_co2_v:
            if deta_temp == 0 or deta_temp < -2.2:
                G_draw[country].append(float(GDP_loss))
                continue
            if order <= 91:
                GDP_loss += deta_temp * a1_GDP
                order += 1
            else:
                GDP_loss += deta_temp * a2_GDP
                order += 1
            G_draw[country].append(float(GDP_loss))
    else:
        print("GetGDPLossAll: city name wrong!")
        exit(0)

    return -1*GDP_loss


def EvaluateCity(city,start_day,co2_path):
    global years, begin_day, city_name, G_draw, Tr_draw, country_4kind_data, path_co2, path_GDP, path_Ill
    city_name = city
    years = 4
    print("以下为"+city+"市评估结果：")
    begin_day = start_day
    path_co2 = co2_path
    path_GDP = "D:\\桌面\\NC\\data\\GDP-new.xlsx"
    path_Ill = "D:\\桌面\\NC\\data\\疫情-new.xlsx"
    illdatapath = "D:\\桌面\\NC\\data\\实验四数据\\仿真的平均结果-实验四-"+city+".xlsx"

    G_draw = {}
    Tr_draw = {}
    country_4kind_data = {}

    CO2_data = GetCO2(path_co2)
    GDP_data = GetGDP(path_GDP)

    Ill_1=GetSimulationIllData(illdatapath, 1)

    Ill = Move7WindowLog(Ill_1)

    GetRelationIlltoCO2(Ill, CO2_data)
    GetRelationCO2toGDP(CO2_data, GDP_data)

    pre_2020 = Predict2020GDP(GDP_data)
    real_2020 = GetReal2020GDP(city_name)
    data_GDP = pre_2020 - real_2020
    print("通过GDP变化趋势预测2020年前四月的GDP得到的",city_name,"防疫GDP下降值为（亿元）：", data_GDP[0])
    #increase_real = GetIllDataForEvaluation(path_Ill)
    increase_real = GetSimulationIllData(illdatapath, 1)
    # print("武汉2020年前四月病例情况：",increase_real)
    GDP_loss = GetGDPLossAll(increase_real, 'real_data',city_name)
    print("通过真实疫情数据和评估模型得到的2020年前四月",city_name,"防疫GDP下降值为（亿元）：", GDP_loss[0],"\t和预测差异为(%)：",(GDP_loss[0]-data_GDP[0])/data_GDP[0]*100)

    temp, CO2_2020=Detrended_and_predict_CO2(CO2_data)
    first_season_co2_emission_predict_2020=np.mean(CO2_2020[0:3])
    if city_name == "武汉":          #3 18
        eva_len = 3
        evaluate_day=9
        GDP_loss_std = GetGDPLoss(increase_real, "无提前", evaluate_day+12,eva_len=eva_len)
        evaluate_day += 1

    elif city_name == "广州":#5 16   #9 17
        eva_len = 1
        evaluate_day=14
        GDP_loss_std = GetGDPLoss(increase_real, "无提前", evaluate_day, eva_len=eva_len)
    elif city_name == "北京":#7 21
        eva_len = 6
        evaluate_day =21
        GDP_loss_std = GetGDPLoss(increase_real, "无提前", evaluate_day+5, eva_len=eva_len)
    elif city_name == "温州":       #7 14
        eva_len = 4
        evaluate_day =9
        GDP_loss_std = GetGDPLoss(increase_real, "无提前", evaluate_day+0, eva_len=eva_len)
    else :
        print("city name wrong!")
        exit(0)
    Treat_cost = GetTreatCost(increase_real, "无提前")
    print(city+":\t真实数据医疗消耗：",Treat_cost)

    # increase_real = GetSimulationIllData(illdatapath, 3)
    # GDP_loss = GetGDPLoss(increase_real, "无提前95CI", evaluate_day, eva_len=eva_len)
    # Treat_cost = GetTreatCost(increase_real, "无提前95CI")
    # print(city + ":\t 无提前95CI: 二氧化碳释放量：(单位：gram/m2/day)：",first_season_co2_emission_predict_2020 * (1 + (GDP_loss_std - GDP_loss) / GDP_loss_std),"\tGDP 相对策略不移动损失(单位：%)：", (GDP_loss - GDP_loss_std) / GDP_loss_std * 100, "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 4)
    # GDP_loss= GetGDPLoss(increase_real, "提前1天",evaluate_day,eva_len=eva_len)
    # Treat_cost = GetTreatCost(increase_real,  "提前1天")
    # print(city+":\t 提前1天: 二氧化碳释放量：(单位：gram/m2/day)：",first_season_co2_emission_predict_2020*(1+(GDP_loss_std-GDP_loss )/GDP_loss_std), "\tGDP 相对策略不移动损失(单位：%)：", (GDP_loss-GDP_loss_std)/GDP_loss_std*100, "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 5)
    # Treat_cost = GetTreatCost(increase_real, "提前1天95CI")
    # print(city + ":\t 提前1天95CI: ","\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 6)
    # GDP_loss= GetGDPLoss(increase_real,  "提前2天",evaluate_day,eva_len=eva_len)
    # Treat_cost = GetTreatCost(increase_real,  "提前2天")
    # print(city+":\t 提前2天: 二氧化碳释放量：(单位：gram/m2/day)：",first_season_co2_emission_predict_2020*(1+(GDP_loss_std-GDP_loss )/GDP_loss_std), "\tGDP 相对策略不移动损失(单位：%)：", (GDP_loss-GDP_loss_std)/GDP_loss_std*100, "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 7)
    # Treat_cost = GetTreatCost(increase_real, "提前2天95CI")
    # print(city + ":\t 提前2天95CI: ", "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 8)
    # GDP_loss= GetGDPLoss(increase_real,  "提前3天",evaluate_day,eva_len=eva_len)
    # Treat_cost = GetTreatCost(increase_real,  "提前3天")
    # print(city+":\t 提前3天: 二氧化碳释放量：(单位：gram/m2/day)：",first_season_co2_emission_predict_2020*(1+(GDP_loss_std-GDP_loss )/GDP_loss_std), "\tGDP 相对策略不移动损失(单位：%)：", (GDP_loss-GDP_loss_std)/GDP_loss_std*100, "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 9)
    # Treat_cost = GetTreatCost(increase_real, "提前3天95CI")
    # print(city + ":\t 提前3天95CI: ", "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 10)
    # GDP_loss = GetGDPLoss(increase_real,  "提前4天",evaluate_day,eva_len=eva_len)
    # Treat_cost = GetTreatCost(increase_real,  "提前4天")
    # print(city+":\t 提前4天: 二氧化碳释放量：(单位：gram/m2/day)：",first_season_co2_emission_predict_2020*(1+(GDP_loss_std-GDP_loss )/GDP_loss_std), "\tGDP 相对策略不移动损失(单位：%)：", (GDP_loss-GDP_loss_std)/GDP_loss_std*100, "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 11)
    # Treat_cost = GetTreatCost(increase_real, "提前4天95CI")
    # print(city + ":\t 提前4天95CI: ", "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 12)
    # GDP_loss= GetGDPLoss(increase_real,  "提前5天",evaluate_day,eva_len=eva_len)
    # Treat_cost = GetTreatCost(increase_real,  "提前5天")
    # print(city+":\t 提前5天: 二氧化碳释放量：(单位：gram/m2/day)：",first_season_co2_emission_predict_2020*(1+(GDP_loss_std-GDP_loss )/GDP_loss_std), "\tGDP 相对策略不移动损失(单位：%)：", (GDP_loss-GDP_loss_std)/GDP_loss_std*100, "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 13)
    # Treat_cost = GetTreatCost(increase_real, "提前5天95CI")
    # print(city + ":\t 提前5天95CI: ", "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 14)
    # GDP_loss= GetGDPLoss(increase_real,  "推迟1天",evaluate_day,eva_len=eva_len)
    # Treat_cost = GetTreatCost(increase_real,  "推迟1天")
    # print(city+":\t 推迟1天: 二氧化碳释放量：(单位：gram/m2/day)：",first_season_co2_emission_predict_2020*(1+(GDP_loss_std-GDP_loss )/GDP_loss_std), "\tGDP 相对策略不移动损失(单位：%)：", (GDP_loss-GDP_loss_std)/GDP_loss_std*100, "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 15)
    # Treat_cost = GetTreatCost(increase_real, "推迟1天95CI")
    # print(city + ":\t 推迟1天95CI: ", "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 16)
    # GDP_loss = GetGDPLoss(increase_real,  "推迟2天",evaluate_day,eva_len=eva_len)
    # Treat_cost = GetTreatCost(increase_real,  "推迟2天")
    # print(city+":\t 推迟2天: 二氧化碳释放量：(单位：gram/m2/day)：",first_season_co2_emission_predict_2020*(1+(GDP_loss_std-GDP_loss )/GDP_loss_std), "\tGDP 相对策略不移动损失(单位：%)：", (GDP_loss-GDP_loss_std)/GDP_loss_std*100, "\t医疗消耗(单位：亿元)：", Treat_cost)
    #
    # increase_real = GetSimulationIllData(illdatapath, 17)
    # Treat_cost = GetTreatCost(increase_real, "推迟2天95CI")
    # print(city + ":\t 推迟2天95CI: ", "\t医疗消耗(单位：亿元)：", Treat_cost)

    deta_day=0
    # if city_name == "武汉":
    #     deta_day = 12
    # elif city_name == "广州":
    #     deta_day = 8
    # elif city_name == "北京":
    #     deta_day = 6
    # elif city_name == "温州":
    #     deta_day = 6
    # else:
    #     print("city name wrong!")
    if city_name == "北京":
        deta_day =14

    increase_real = GetSimulationIllData(illdatapath, 18)
    GDP_loss = GetGDPLoss(increase_real, "无措施", evaluate_day-deta_day, eva_len=eva_len)
    Treat_cost = GetTreatCost(increase_real, "无措施")
    print(city + ":\t 无措施: 二氧化碳释放量：(单位：gram/m2/day)：",first_season_co2_emission_predict_2020 * (1 + (GDP_loss_std - GDP_loss) / GDP_loss_std), "\tGDP 相对策略不移动损失(单位：%)：", (GDP_loss - GDP_loss_std) / GDP_loss_std * 100, "\t医疗消耗(单位：亿元)：", Treat_cost)


    # DrawFigure()

def SumTrandG(G_list,Tr_list):
    #print('两个的大小：',len(G_list),'\t',len(Tr_list))
    sum_list=[]
    for i in range(len(G_list)):
        sum_list.append(G_list[i]+Tr_list[i])
    return sum_list


def DrawFigure():
    global Tr_draw, G_draw,cityname
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'color': 'black',
            'size': 16
            }

    figure, ax = plt.subplots(1, 3, num=666, figsize=(12, 5))
    temp_ax = ax[0]
    temp_ax.plot(range(len(Tr_draw['提前1天'])), np.array(Tr_draw['提前1天'])/10 , '--', color='tab:red', label='Tq 1 day')
    temp_ax.plot(range(len(Tr_draw['提前2天'])), np.array(Tr_draw['提前2天']) / 10, '--', color='tab:green', label='Tq 2 day')
    temp_ax.plot(range(len(Tr_draw['提前3天'])), np.array(Tr_draw['提前3天']) / 10, '--', color='tab:blue', label='Tq 3 day')
    temp_ax.plot(range(len(Tr_draw['提前4天'])), np.array(Tr_draw['提前4天']) / 10, '--', color='y', label='Tq 4 day')
    temp_ax.plot(range(len(Tr_draw['提前5天'])), np.array(Tr_draw['提前5天']) / 10, '--', color='c', label='Tq 5 day')
    temp_ax.plot(range(len(Tr_draw['推迟1天'])), np.array(Tr_draw['推迟1天']) / 10, '--', color='k', label='Tc 1 day')
    temp_ax.plot(range(len(Tr_draw['推迟2天'])), np.array(Tr_draw['推迟2天']) / 10, '--', color='r', label='Tc 2 day')

    labels = temp_ax.get_xticklabels() + temp_ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    temp_ax.tick_params(labelsize=16)
    temp_ax.legend(prop={'family': 'Times New Roman',
                         'weight': 'normal',
                         'size': 14
                         })

    temp_ax.set_ylabel("Treat cost (billion ¥)", fontdict=font)
    temp_ax.set_xlabel("Time (day)\n(a)", fontdict=font)

    temp_ax = ax[1]
    temp_ax.plot(range(len(G_draw['提前1天'])), np.array(G_draw['提前1天']) / 10, '--', color='tab:red', label='Tq 1 day')
    temp_ax.plot(range(len(G_draw['提前2天'])), np.array(G_draw['提前2天']) / 10, '--', color='tab:green', label='Tq 2 day')
    temp_ax.plot(range(len(G_draw['提前3天'])), np.array(G_draw['提前3天']) / 10, '--', color='tab:blue', label='Tq 3 day')
    temp_ax.plot(range(len(G_draw['提前4天'])), np.array(G_draw['提前4天']) / 10, '--', color='y', label='Tq 4 day')
    temp_ax.plot(range(len(G_draw['提前5天'])), np.array(G_draw['提前5天']) / 10, '--', color='c', label='Tq 5 day')
    temp_ax.plot(range(len(G_draw['推迟1天'])), np.array(G_draw['推迟1天']) / 10, '--', color='k', label='Tc 1 day')
    temp_ax.plot(range(len(G_draw['推迟2天'])), np.array(G_draw['推迟2天']) / 10, '--', color='r', label='Tc 2 day')

    labels = temp_ax.get_xticklabels() + temp_ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    temp_ax.tick_params(labelsize=16)
    temp_ax.legend(prop={'family': 'Times New Roman',
                         'weight': 'normal',
                         'size': 14
                         })

    temp_ax.set_ylabel("GDP loss (billion ¥)", fontdict=font)
    temp_ax.set_xlabel("Time (day)\n(b)", fontdict=font)

    temp_ax = ax[2]
    temp_ax.plot(range(len(Tr_draw['提前1天'])), np.array(SumTrandG(G_draw['提前1天'],Tr_draw['提前1天'])) / 10, '--', color='tab:red', label='Tq 1 day')
    temp_ax.plot(range(len(Tr_draw['提前2天'])), np.array(SumTrandG(G_draw['提前2天'],Tr_draw['提前2天'])) / 10, '--', color='tab:green', label='Tq 2 day')
    temp_ax.plot(range(len(Tr_draw['提前3天'])), np.array(SumTrandG(G_draw['提前3天'],Tr_draw['提前3天'])) / 10, '--', color='tab:blue', label='Tq 3 day')
    temp_ax.plot(range(len(Tr_draw['提前4天'])), np.array(SumTrandG(G_draw['提前4天'],Tr_draw['提前4天'])) / 10, '--', color='y', label='Tq 4 day')
    temp_ax.plot(range(len(Tr_draw['提前5天'])), np.array(SumTrandG(G_draw['提前5天'],Tr_draw['提前5天'])) / 10, '--', color='c', label='Tq 5 day')
    temp_ax.plot(range(len(Tr_draw['推迟1天'])), np.array(SumTrandG(G_draw['推迟1天'],Tr_draw['推迟1天'])) / 10, '--', color='k', label='Tc 1 day')
    temp_ax.plot(range(len(Tr_draw['推迟2天'])), np.array(SumTrandG(G_draw['推迟2天'],Tr_draw['推迟2天'])) / 10, '--', color='r', label='Tc 2 day')

    labels = temp_ax.get_xticklabels() + temp_ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    temp_ax.tick_params(labelsize=16)
    temp_ax.legend(prop={'family': 'Times New Roman',
                         'weight': 'normal',
                         'size': 14
                         })

    temp_ax.set_ylabel("GDP cost (billion ¥)", fontdict=font)
    temp_ax.set_xlabel("Time (day)\n(c)", fontdict=font)
    #temp_ax.set_xticks(range(0, 230, 70))



    plt.tight_layout()
    # plt.plot(range(len(Tr_draw['nostrategy'])), Tr_draw['nostrategy'],'.')
    plt.savefig('D:\\桌面\\NC\\图\\'+city_name+'_经济效应图.jpg', dpi=900)
    #plt.show()
    # print(G_draw['china'])
    return 0
if __name__=="__main__":
    # EvaluateCity("武汉",-4,"D:\\桌面\\NC\\data\\wuhan-co2-concentration.csv")
    EvaluateCity("广州", 4, "D:\\桌面\\NC\\data\\guanggzhou-co2.csv")
    # EvaluateCity("北京", 4, "D:\\桌面\\NC\\data\\beijing-co2-2c.csv")
    # EvaluateCity("温州", 12, "D:\\桌面\\NC\\data\\wenzhou-co2-4.csv")













