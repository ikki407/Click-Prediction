"""
実行する際に変更する点としまして、ファイル入出力のPATHを変更していただきたいと思います。
お手数おかけしますがよろしくお願いします。

ニックネーム：きゃずき
本名：田中一樹
"""


import pandas as pd
from pandas import DataFrame,Series
import numpy as np

#配布されたtrain.csvをtrain-2.csvと改名してしまっているが、コードをtrain.csvとすれば良い
train = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/train-2.csv',header=None,sep=',')
test = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/test.csv',header=None,sep=',')
Label = train.iloc[:,1]

train.columns = ['Id','Label','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14']
test.columns = ['Id','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14']


#まず、配布されたtrain,testデータにおいて連続するログデータの数を各列に作成する。
#新たに列を追加したデータをtrain_n_3.csv,test_n_3.csvとして保存する。
#
#
#例）全く連続していないもの->0
#    ２個連続しているもの->1
#    10個連続しているもの->9
#
#

#新たに値を追加する列を追加（直接代入すると時間がかかるので小さなDataFrameを作成したほうが時間は1/4程度に減少できたが使用したコードを載せます）
train['I15'] = DataFrame(np.zeros(len(train)))
test['I15'] = DataFrame(np.zeros(len(test)))
#欠損値を比較できないので、-100を代入しておく
train = train.fillna(-100)
test = test.fillna(-100)

#trainの数
for i in xrange(len(train)):
    c = 0
    decision = True
    j_plus=1
    j_minus=-1

    while decision:
        try:
            if list(train.iloc[i,2:22])==list(train.iloc[(j_plus+i),2:22]):
                c+=1
                j_plus+=1
            if list(train.iloc[i,2:22])==list(train.iloc[(j_minus+i),2:22]):
                c+=1
                j_minus+= -1

            if list(train.iloc[i,2:22])!=list(train.iloc[(j_plus+i),2:22]) and list(train.iloc[i,2:22])!=list(train.iloc[(j_minus+i),2:22]):
                decision = False

        except:
            pass
            decision = False

    train.iloc[i,22] = c
    print c,i

train_n = train['I15'].copy()
train = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/train-2.csv',header=None,sep=',')
train.columns = ['Id','Label','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14']
train['I15'] = DataFrame(train_n)
#保存
train.to_csv('/Users/IkkiTanaka/Documents/OPT2/train_n_3.csv',index=None,header=None)

#test
for i in xrange(len(test)):
    c = 0
    decision = True
    j_plus=1
    j_minus=-1

    while decision:
        try:
            if list(test.iloc[i,1:21])==list(test.iloc[(j_plus+i),1:21]):
                c+=1
                j_plus+=1
            if list(test.iloc[i,1:21])==list(test.iloc[(j_minus+i),1:21]):
                c+=1
                j_minus+= -1

            if list(test.iloc[i,1:21])!=list(test.iloc[(j_plus+i),1:21]) and list(test.iloc[i,1:21])!=list(test.iloc[(j_minus+i),1:21]):
                decision = False

        except:
            pass
            decision = False

    test.iloc[i,21] = c
    print c,i
test_n = test['I15'].copy()
test = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/test.csv',header=None,sep=',')
test.columns = ['Id','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14']
test['I15'] = DataFrame(test_n)
#保存
test.to_csv('/Users/IkkiTanaka/Documents/OPT2/test_n_3.csv',index=None,header=None)



#quad3_4_cionly.csvを作成
#
# 1)train_n_3.csv,test_n_3.csvを読み込む
# 2)vowpal wabbit(VW)用にデータを変換(csv_to_vwモジュール使用)
# 3)subprocessモジュールを使用してpythonからterminalでVWを実行
#   VWコマンド
#   train: vw train_vw3.vw -k -c -f train.model3.vw --loss_function logistic --passes 20 -l 0.5 -b 30 --nn 50 --holdout_period 5 -q ci
#   test: vw test_vw3.vw -t -i train.model3.vw -p test3.txt
# 4)ロジスティックシグモイド関数を用いて予測値に変換
#
#

train = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/train_n_3.csv',header=None,sep=',')
test = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/test_n_3.csv',header=None,sep=',')

train.columns = ['Id','Label','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15']
test.columns = ['Id','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15']

train.to_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/train3.csv')
test.to_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/test3.csv')

#VW用に変換
import csv_to_vw
csv_to_vw.csv_to_vw('/Users/IkkiTanaka/Documents/OPT2/VWdata/train3.csv','/Users/IkkiTanaka/Documents/OPT2/VWdata/train_vw3.vw',train=True)
csv_to_vw.csv_to_vw('/Users/IkkiTanaka/Documents/OPT2/VWdata/test3.csv','/Users/IkkiTanaka/Documents/OPT2/VWdata/test_vw3.vw',train=False)

#subprocessモジュールを使用してterminalでVWを実行
import subprocess 
command_train = 'vw train_vw3.vw -k -c -f train.model3.vw --loss_function logistic --passes 20 -l 0.5 -b 30 --nn 50 --holdout_period 5 -q ci'
command_test = 'vw test_vw3.vw -t -i train.model3.vw -p test3.txt'

subprocess.check_output(command_train, shell=True)
subprocess.check_output(command_test, shell=True)
#最終出力ファイルのPATH(quad3_4_cionly.csv)
outputfile = "/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_cionly.csv"

#ロジスティックシグモイド関数を用いて予測値に変換
import math
def zygmoid(x):
    return 1 / (1 + math.exp(-x))

with open(outputfile,"wb") as outfile:
    #outfile.write("Id,Predicted\n")
    for line in open("/Users/IkkiTanaka/Documents/OPT2/VWdata/test3.txt"):
        row = line.strip().split(" ")
        outfile.write("%s,%f\n"%(row[1],zygmoid(float(row[0]))))


#scikit-learnのGradientBoostingClassifierを用いて予測
# 1)木の数30個、深さ2~10の合計9つのGradientBoostingClassifierを作成
# 2)train_n_3.csvで学習し,test_n_3.csvで1(クリックされる)になる確率を予測し新たに列に追加
#   同時に、train_n_3.csvでも予測し列を追加
# 3)元のtrain,testデータのNumericalデータ('I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15')を２乗し対数変換
#   new_x = math.log(x*x)
# 4)新たに作成したtrainデータを復元抽出(シードは木の深さ+100)
# 5)VW用にデータを変換
# 6)subprocessモジュールを使用してpythonからterminalでVWを実行し予測値に変換
#   VWコマンド
#   train: 'vw train_vw_%s_%s_3.vw -k -c -f train.model5_3.vw --loss_function logistic --passes 15 -l 0.28 -b 30 --nn 50 --holdout_period 3 -q ci -q ii' % (n_estimate,max_depth)
#   test: 'vw test_vw_%s_%s_3.vw -t -i train.model5_3.vw -p test7_3.txt' % (n_estimate,max_depth)
#
#


from sklearn.ensemble import GradientBoostingClassifier
import math
for n_estimate in [30,]:
    for max_depth in [2,3,4,5,6,7,8,9,10]:

        train = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/train_n_3.csv',header=None,sep=',')
        test = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/test_n_3.csv',header=None,sep=',')
        Label = train.iloc[:,1]
        train.columns = ['Id','Label','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15']
        test.columns = ['Id','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15']

        #GBDT
        clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.25, n_estimators=n_estimate, subsample=1.0, min_samples_split=5, min_samples_leaf=1, max_depth=max_depth, init=None, random_state=407, max_features=None, verbose=2, max_leaf_nodes=None, warm_start=False)

        #trainデータで学習(欠損値は1で補間)
        tt = clf.fit(train.iloc[:,2:].fillna(1),Label)
        #train,test両方で確率を予測
        tr_gbdt = tt.predict_proba(train.iloc[:,2:].fillna(1))
        ts_gbdt = tt.predict_proba(test.iloc[:,1:].fillna(1))

        tr_gbdt = DataFrame(DataFrame(tr_gbdt)[1])
        ts_gbdt = DataFrame(DataFrame(ts_gbdt)[1])
        #train,test共に元のデータに予測値を新たな列に追加
        train_gbdt = pd.concat([train,tr_gbdt],axis=1)
        test_gbdt = pd.concat([test,ts_gbdt],axis=1)

        for i in ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15']:
            train_gbdt.loc[(train_gbdt[i]>=2).values,i] = map(lambda x:math.log(x*x),train_gbdt.iloc[(train_gbdt[i]>=2).values,:][i].values)

            test_gbdt.loc[(test_gbdt[i]>=2).values,i] = map(lambda x:math.log(x*x),test_gbdt.iloc[(test_gbdt[i]>=2).values,:][i].values)


        train_gbdt.columns = ['Id','Label','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15','I16']
        test_gbdt.columns = ['Id','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15','I16']
        
        #再利用可能のためにデータを保存
        train_gbdt.to_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/train3_n_gbdt_Ilog_3.csv',header=None,index=False)
        test_gbdt.to_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/test3_n_gbdt_Ilog_3.csv',header=None,index=False)
        #保存したデータを読み込む
        train = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/train3_n_gbdt_Ilog_3.csv',header=None,sep=',')
        test = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/test3_n_gbdt_Ilog_3.csv',header=None,sep=',')
        Label = train.iloc[:,1]
        train.columns = ['Id','Label','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15','I16']
        test.columns = ['Id','I1','I2','C1','C2','C3','C4','C5','C6','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15','I16']
        
        #復元抽出
        np.random.seed(max_depth+100)
        train = train.loc[np.random.choice(train.index, len(train), replace=True)]#重複あり
        

        train.to_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/train3_3.csv')
        test.to_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/test3_3.csv')


        #VW用に変換
        import csv_to_vw
        path_train = '/Users/IkkiTanaka/Documents/OPT2/VWdata/train_vw_%s_%s_3.vw' % (n_estimate,max_depth)
        path_test = '/Users/IkkiTanaka/Documents/OPT2/VWdata/test_vw_%s_%s_3.vw' % (n_estimate,max_depth)

        csv_to_vw.csv_to_vw('/Users/IkkiTanaka/Documents/OPT2/VWdata/train3_3.csv',path_train,train=True)
        csv_to_vw.csv_to_vw('/Users/IkkiTanaka/Documents/OPT2/VWdata/test3_3.csv',path_test,train=False)
        print '%s_%s is done!' % (n_estimate,max_depth)

#
#それぞれ実行,変換
#

import subprocess 
for n_estimate in [30,]:
    for max_depth in [2,3,4,5,6,7,8,9,10]:
        command_train = 'vw train_vw_%s_%s_3.vw -k -c -f train.model5_3.vw --loss_function logistic --passes 15 -l 0.28 -b 30 --nn 50 --holdout_period 3 -q ci -q ii' % (n_estimate,max_depth)
        command_test = 'vw test_vw_%s_%s_3.vw -t -i train.model5_3.vw -p test7_3.txt' % (n_estimate,max_depth)

        subprocess.check_output(command_train, shell=True)
        subprocess.check_output(command_test, shell=True)
        outputfile = "/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_ci_gbdt_est%sdep%s_ran_3.csv" % (n_estimate,max_depth) 
        
        import math

        def zygmoid(x):
        	#I know it's a common Sigmoid feature, but that's why I probably found
        	#it on FastML too: https://github.com/zygmuntz/kaggle-stackoverflow/blob/master/sigmoid_mc.py
        	return 1 / (1 + math.exp(-x))

        with open(outputfile,"wb") as outfile:
        	#outfile.write("Id,Predicted\n")
        	for line in open("/Users/IkkiTanaka/Documents/OPT2/VWdata/test7_3.txt"):
        		row = line.strip().split(" ")
        		outfile.write("%s,%f\n"%(row[1],zygmoid(float(row[0])))
        
                )

#ensemble model
# 1)ここまで作成した10個の予測値の平均を取る
#   quad3_4_cionly.csv,quad3_4_ci_gbdt_est%sdep%s_ran_3.csv(9個)
# 2)予測値が0.9999以上のものを0.9999にする
# 3)最終提出物last_submit1.csvを保存する
#

#各予測値を読み込む
label1 = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_ci_gbdt_est30dep2_ran_3.csv',header=None,sep=',')
label2 = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_ci_gbdt_est30dep3_ran_3.csv',header=None,sep=',')
label3 = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_ci_gbdt_est30dep4_ran_3.csv',header=None,sep=',')
label4 = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_ci_gbdt_est30dep5_ran_3.csv',header=None,sep=',')
label5 = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_ci_gbdt_est30dep6_ran_3.csv',header=None,sep=',')
label6 = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_ci_gbdt_est30dep7_ran_3.csv',header=None,sep=',')
label7 = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_ci_gbdt_est30dep8_ran_3.csv',header=None,sep=',')
label8 = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_ci_gbdt_est30dep9_ran_3.csv',header=None,sep=',')
label9 = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_ci_gbdt_est30dep10_ran_3.csv',header=None,sep=',')

label10 = pd.read_csv('/Users/IkkiTanaka/Documents/OPT2/VWdata/quad3_4_cionly.csv',header=None,sep=',')

all_label = label1[1]+label2[1]+label3[1]+label4[1]+label5[1]+label6[1]+label7[1]+label8[1]+label9[1]+label10[1]
#平均を取る
all_label = all_label/10.0
all_label = pd.concat([label1[0],all_label],axis=1)
#0.9999以上を0.9999にする
all_label.loc[(all_label[1]>=0.9999).values,1] = 0.9999
#保存
all_label.to_csv("/Users/IkkiTanaka/Documents/OPT2/VWdata/last_submit1.csv",header=None,index=False)

