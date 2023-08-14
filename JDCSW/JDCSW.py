import time
from datetime import datetime
import spams ##
import math
import numpy as np
import random as rd
import random as rnd
import scipy.io
from scipy import stats
from scipy.stats import bernoulli
from numpy import matlib
import sklearn.preprocessing as sk
from sklearn.model_selection import StratifiedKFold
from scipy import *
def display_iteration_detail_to_screen(datasetsName,algosettings,accuracy, trainTime, trainRateTime, testRateTime, initDictSize,finalDictSize,smean):
    print(datasetsName)
    print(algosettings)
    print(f"[Recognition rate: {(accuracy * 100):6.2f} %]")
    print(f"[Mean recognition rate: {(smean):6.2f} %]")
    print(f"[Training time: {(trainTime/60):6.2f} mins.]")
    print(f"[Training time per example: {(1000*trainRateTime):6.2f} milli-sec.]")
    print(f"[Classification time per sample: {(1000 * testRateTime):6.2f} milli-sec.]")
    print(f"[Initial dictionary size: {initDictSize:5d} atoms]")
    print(f"[Final dictionary size: {finalDictSize:5d} atoms]")
def write_iteration_detail_to_file(ff,strn,algosettings,accuracy, trainTime, trainRateTime, testRateTime,  initDictSize, finalDictSize,smean):
    ff.write(strn)
    ff.write(algosettings + "\n")
    ff.write(f"[Recognition rate: {(accuracy * 100):6.2f} %]\n")
    ff.write(f"[Mean recognition rate: {(smean):6.2f} %]\n")
    ff.write(f"[Training time: {(trainTime/60):6.2f} mins.]\n")
    ff.write(f"[Training time per example: {(1000*trainRateTime):6.2f} milli-sec.]\n")
    ff.write(f"[Classification time per sample: {(1000 * testRateTime):6.2f} milli-sec.]\n")
    ff.write(f"[Initial dictionary size: {initDictSize:5d} atoms]\n")
    ff.write(f"[Final dictionary size: {finalDictSize:5d} atoms]\n")
def write_summary_to_file(f, strn, algosettings,iter1,mean,sdev,meand, averageTrainTime,averageTrainRateTime, averageTestRateTime):
    f.write(strn)
    f.write(algosettings + "\n")
    f.write(f"Average iterations = {iter1:4.0f} Nos\n")
    f.write(f"Average accuracy = {mean:6.2f} %\n")
    f.write(f"Standard Deviation = {sdev:5.2f}\n")
    f.write(f"Average dictionary size = {meand:5.0f}\n")
    f.write(f"Average training time = {averageTrainTime/60:8.2f} mins.\n")
    f.write(f"Average training time per example = {1000*averageTrainRateTime:8.2f} milli-sec.\n")
    f.write(f"Averge classification time per sample = {1000*averageTestRateTime:8.2f} milli-sec.\n")
def display_summary_to_screen(strn,algosettings, iter1,mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime):
    print(strn)
    print(algosettings + "\n")
    print(f"Average iterations = {iter1:4.0f} Nos\n")
    print(f"Average accuracy = {mean:6.2f} %\n")
    print(f"Standard Deviation = {sdev:5.2f}\n")
    print(f"Average dictionary size = {meand:5.0f}\n")
    print(f"Average training time = {averageTrainTime/60:8.2f} mins.\n")
    print(f"Average training time per example = {1000*averageTrainRateTime:8.2f} milli-sec.\n")
    print(f"Averge classification time per sample = {1000*averageTestRateTime:8.2f} ms\n")
def extended_yaleB(features,labels,counter):
    classes=labels.shape[0]
    trainColIndex=list()
    b = np.arange(0, labels.shape[1])
    for c in range(classes):
        classIndex=[x0 for x0, val in enumerate(b) if labels[c,val]!=0]
        trainColIndex = trainColIndex + rd.sample(classIndex, 15)
    testColIndex=[x0 for x0, val in enumerate(b) if x0 not in trainColIndex]
    training_samples=features[:,trainColIndex]
    train_label=labels[:,trainColIndex]
    test=features[:,testColIndex]
    test_label=labels[:,testColIndex]
    return training_samples, train_label, test, test_label
def fifteen_scene(features,labels,counter):
    classes=labels.shape[0]
    trainColIndex=list()
    b = np.arange(0, labels.shape[1])
    for c in range(classes):
        classIndex=[x0 for x0, val in enumerate(b) if labels[c,val]!=0]
        trainColIndex = trainColIndex + rd.sample(classIndex, 50)
    testColIndex=[x0 for x0, val in enumerate(b) if x0 not in trainColIndex]
    training_samples=features[:,trainColIndex]
    train_label=labels[:,trainColIndex]
    test=features[:,testColIndex]
    test_label=labels[:,testColIndex]
    return training_samples, train_label, test, test_label
def caltech_data(features,labels,counter):
    classes=labels.shape[0]
    trainColIndex=list()
    b = np.arange(0, labels.shape[1])
    trainSamples=counter*5
    for c in range(classes):
        classIndex=[x0 for x0, val in enumerate(b) if labels[c,val]!=0]  #[x0 for x0, val in enumerate(labels[c, :]) if val != 0]
        trainColIndex = trainColIndex + rd.sample(classIndex, trainSamples)
    testColIndex=[x0 for x0, val in enumerate(b) if x0 not in trainColIndex]
    training_samples=features[:,trainColIndex]
    train_label=labels[:,trainColIndex]
    test=features[:,testColIndex]
    test_label=labels[:,testColIndex]
    return training_samples, train_label, test, test_label
def AR_datset(features,labels,counter):
    classes=labels.shape[0]
    trainColIndex=list()
    b = np.arange(0, labels.shape[1])
    for c in range(classes):
        classIndex=[x0 for x0, val in enumerate(b) if labels[c,val]!=0]  #[x0 for x0, val in enumerate(labels[c, :]) if val != 0]
        trainColIndex = trainColIndex + rd.sample(classIndex, 7)
    testColIndex=[x0 for x0, val in enumerate(b) if x0 not in trainColIndex]
    training_samples=features[:,trainColIndex]
    train_label=labels[:,trainColIndex]
    test=features[:,testColIndex]
    test_label=labels[:,testColIndex]
    return training_samples, train_label, test, test_label
def ucf_datset2(DB):
    abc = np.load(DB)
    features = abc['features']
    labels = abc['labels']
    lstrain=list()
    lstest=list()
    X=np.zeros(features.shape[1])
    labs=np.zeros(features.shape[1])

    for c in range (labels.shape[0]):
        labs[labels[c,:]!=0]=c
    skf=StratifiedKFold(5)
    for train_index, test_index in skf.split(X,labs):
        for ii in range(1):
            lstrain.append(train_index)
            lstest.append(test_index)
    return lstrain,lstest, features,labels
def JDCSW_main(named,iiii,dicSize,DB,Code_DPath,resultFile,DBdisc,gibi,k0,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw,stp,batchsize,foperation="w+"):
    f = open(Code_DPath+resultFile, foperation)
    if named==1:
        mat = scipy.io.loadmat(DB)
        DataSet = mat['DataBase']
        training_samples = DataSet[0][0][0]
        test = DataSet[0][0][1]
        train_label = DataSet[0][0][2]
        test_label = DataSet[0][0][3]
        features=np.hstack((training_samples,test))
        labels=np.hstack((train_label,test_label))
        rcounter=0
        counter=0
        meand=0.0
        averageTrainTime=0.0
        averageTrainRateTime=0.0
        averageTestRateTime=0.0
        mean=0.0
        iter1=0
        sl=[]
        fh=0
        while counter!=10:
            f = open(Code_DPath+resultFile, foperation)
            strn = ""
            training_samples, train_label, test, test_label  =extended_yaleB(features, labels, counter)
            algosettings, iter,accuracy, trainTime, trainRateTime, testRateTime, initDictSize, finalDictSize = JDCSW_training(stp,batchsize,DBdisc,dicSize,iiii,training_samples, train_label, test,test_label,k0,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw)
            mean=mean+(accuracy*100)
            sl = sl + [accuracy * 100]
            iter1 = iter1 + iter
            counter = counter + 1
            rcounter = rcounter + 1
            smean=mean/rcounter
            meand = meand + finalDictSize
            averageTrainTime = averageTrainTime + trainTime
            averageTrainRateTime = averageTrainRateTime + trainRateTime
            averageTestRateTime=averageTestRateTime+testRateTime
            if fh==0:
                f.write("\n\n\n###Experiment for YaleB dataset for face recogition ("+str(datetime.now())+")###\n")
                fh=1
            strn=f"\n======================Experiment No. {rcounter:3d} of YaleB dataset for face recognition===============\n"
            write_iteration_detail_to_file(f, strn, algosettings,accuracy, trainTime, trainRateTime, testRateTime, initDictSize, finalDictSize, smean)
            display_iteration_detail_to_screen(strn,algosettings, accuracy, trainTime, trainRateTime, testRateTime, initDictSize, finalDictSize,smean)
            f.close()
        mean=mean/rcounter
        sl=np.asarray(sl)
        sdev=sum((sl-mean)**2)
        sdev=np.sqrt(sdev/rcounter)
        meand=meand/rcounter
        averageTrainTime = averageTrainTime/rcounter
        averageTrainRateTime = averageTrainRateTime/rcounter
        averageTestRateTime=averageTestRateTime/rcounter
        iter1=iter1/rcounter
        f = open(Code_DPath+resultFile, foperation)
        strn=f"\nSummary of {rcounter:4d} experiments of YaleB dataset for face recognition\n"
        write_summary_to_file(f, strn, algosettings, iter1, mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)
        display_summary_to_screen(strn, algosettings, iter1,mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)
        f.close()
    if named==2:
        abc=np.load(DB)
        training_samples = abc['training_samples']
        test=abc['test']
        train_label=abc['train_label']
        test_label=abc['test_label']
        features=np.hstack((training_samples,test))
        labels=np.hstack((train_label,test_label))
        rcounter = 0
        counter=0
        meand=0.0
        averageTrainTime=0.0
        averageTrainRateTime=0.0
        averageTestRateTime=0.0
        mean=0.0
        iter1 = 0
        sl=[]
        fh=0
        while counter!=10:
            f = open(Code_DPath+resultFile, foperation)
            strn = ""
            training_samples, train_label, test, test_label  =AR_datset(features,labels,counter)
            algosettings,iter,accuracy, trainTime, trainRateTime, testRateTime, initDictSize, finalDictSize = JDCSW_training(stp,batchsize,DBdisc,dicSize,iiii,training_samples, train_label, test,test_label,k0,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw)
            sl = sl + [accuracy * 100]
            iter1 = iter1 + iter
            counter = counter + 1
            rcounter = rcounter + 1
            mean = mean + (accuracy * 100)
            smean = mean / rcounter
            meand = meand + finalDictSize
            averageTrainTime = averageTrainTime + trainTime
            averageTrainRateTime = averageTrainRateTime + trainRateTime
            averageTestRateTime=averageTestRateTime+testRateTime
            if fh==0:
                f.write("\n\n\n###Experiment of AR face database for face recognition ("+str(datetime.now())+")###\n")
                fh=1
            strn=f"\n======================Experiment No. {rcounter:5d} of AR face database for face recognition ===============\n"
            write_iteration_detail_to_file(f, strn, algosettings,accuracy, trainTime, trainRateTime, testRateTime, initDictSize,finalDictSize, smean)
            display_iteration_detail_to_screen(strn, algosettings,accuracy, trainTime, trainRateTime, testRateTime, initDictSize,finalDictSize, smean)
            f.close()
        mean=mean/rcounter
        sl=np.asarray(sl)
        sdev=sum((sl-mean)**2)
        sdev=np.sqrt(sdev/rcounter)
        meand=meand/rcounter

        averageTrainTime = averageTrainTime/rcounter
        averageTrainRateTime = averageTrainRateTime/rcounter
        averageTestRateTime=averageTestRateTime/rcounter
        iter1 = iter1 / rcounter
        f = open(Code_DPath+resultFile, foperation)
        strn=f"\nSummary of 10 experiments of AR face database for face recognition \n"
        write_summary_to_file(f, strn, algosettings, iter1,mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)

        strn = f"\nSummary of 10 experiments of AR face database for face recognition \n"
        display_summary_to_screen(strn, algosettings, iter1, mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)
        f.close()
    if named==3:

        abc=np.load(DB)
        training_samples = abc['training_samples']
        test=abc['test']
        train_label=abc['train_label']
        test_label=abc['test_label']
        features=np.hstack((training_samples,test))
        labels=np.hstack((train_label,test_label))
        gcounter=0
        while gcounter<6:
            gcounter=gcounter+1
            rcounter = 0
            counter=0
            meand=0.0
            averageTrainTime=0.0
            averageTrainRateTime=0.0
            averageTestRateTime=0.0
            mean=0.0
            iter1 = 0
            sl=[]
            fh=0
            flc=0
            while counter!=1:
                flc=1
                f = open(Code_DPath+resultFile, foperation)
                strn = ""
                training_samples, train_label, test, test_label  =caltech_data(features,labels,gcounter)
                algosettings,iter,accuracy, trainTime, trainRateTime, testRateTime, initDictSize, finalDictSize = JDCSW_training(stp,batchsize,DBdisc,dicSize,iiii,training_samples, train_label, test,test_label,k0,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw)
                sl = sl + [accuracy * 100]
                iter1 = iter1 + iter
                counter = counter + 1
                rcounter = rcounter + 1
                mean = mean + (accuracy * 100)
                smean=mean/rcounter
                meand = meand + finalDictSize
                averageTrainTime = averageTrainTime + trainTime
                averageTrainRateTime = averageTrainRateTime + trainRateTime
                averageTestRateTime=averageTestRateTime+testRateTime
                if fh==0:
                    f.write("\n\n\n###Experiment of Caltech database for object classification ("+str(datetime.now())+")###\n")
                    fh=1
                strn = f"\n===JBDC_p Experiment No. {rcounter:2d} for training samples {((gcounter) * 5):5d} per class of Caltech database for object classification===\n"
                write_iteration_detail_to_file(f, strn, algosettings, accuracy, trainTime, trainRateTime, testRateTime, initDictSize,finalDictSize, smean)
                display_iteration_detail_to_screen(strn, algosettings, accuracy, trainTime, trainRateTime, testRateTime,initDictSize,finalDictSize, smean)
                f.close()
            mean=mean/rcounter
            sl=np.asarray(sl)
            sdev=sum((sl-mean)**2)
            sdev=np.sqrt(sdev/rcounter)
            meand=meand/rcounter
            averageTrainTime = averageTrainTime/rcounter
            averageTrainRateTime = averageTrainRateTime/rcounter
            averageTestRateTime=averageTestRateTime/rcounter
            iter1 = iter1 / rcounter
            f = open(Code_DPath+resultFile, foperation)
            strn = f"\nJBDC_p Summary of {((rcounter)):2d} experiments of {((gcounter) * 5):2d} samples per class from one out of a series of 6 experiments [5,10,15,20,25,30] training samples per class of clatech dataset for object recognition \n"
            write_summary_to_file(f, strn, algosettings, iter1,mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)

            strn = f"\nJBDC_p Summary of {((rcounter)):2d} experiments of {((gcounter) * 5):2d} samples per class from one out of a series of 6 experiments [5,10,15,20,25,30] training samples per class of clatech dataset for object recognition \n"
            display_summary_to_screen(strn, algosettings, iter1, mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)
            f.close()
    if named==4:

        rcounter=0
        counter=0
        meand=0.0
        averageTrainTime=0.0
        averageTrainRateTime=0.0
        averageTestRateTime=0.0
        mean=0.0
        iter1 = 0
        lstrain,lstest, features,labels=ucf_datset2(DB)
        sl=[]
        ff=0
        fh=0
        while counter<len(lstrain):
            f = open(Code_DPath+resultFile, foperation)
            strn = ""
            training_samples=features[:,lstrain[counter]]
            train_label=labels[:,lstrain[counter]]
            test=features[:,lstest[counter]]
            test_label=labels[:,lstest[counter]]
            algosettings,iter,accuracy, trainTime, trainRateTime, testRateTime, initDictSize, finalDictSize = JDCSW_training(stp,batchsize,DBdisc,dicSize,iiii,training_samples, train_label, test,test_label,k0,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw)
            sl = sl + [accuracy * 100]
            iter1 = iter1 + iter
            counter = counter + 1
            rcounter = rcounter + 1
            mean = mean + (accuracy * 100)
            smean=mean/rcounter
            meand = meand + finalDictSize
            averageTrainTime = averageTrainTime + trainTime
            averageTrainRateTime = averageTrainRateTime + trainRateTime
            averageTestRateTime=averageTestRateTime+testRateTime
            if fh==0:
                f.write("\n\n\n###Experiment of ucf sport database for action recognition (5-fold) ("+str(datetime.now())+")###\n")
                fh=1
            strn = f"\n===Experiment No. {rcounter:5d} for 5-fold  of ucf sport database for action recognition===\n"
            write_iteration_detail_to_file(f, strn, algosettings, accuracy, trainTime, trainRateTime, testRateTime, initDictSize,finalDictSize, smean)
            display_iteration_detail_to_screen(strn, algosettings, accuracy, trainTime, trainRateTime, testRateTime, initDictSize,finalDictSize, smean)
            f.close()
        mean=mean/rcounter
        sl=np.asarray(sl)
        sdev=sum((sl-mean)**2)
        sdev=np.sqrt(sdev/rcounter)
        meand=meand/rcounter
        averageTrainTime = averageTrainTime/rcounter
        averageTrainRateTime = averageTrainRateTime/rcounter
        averageTestRateTime=averageTestRateTime/rcounter
        iter1 = iter1 /rcounter
        f = open(Code_DPath+resultFile, foperation)
        strn = f"\nSummary of 5-fold  of ucf sport database for action recognition \n"
        write_summary_to_file(f, strn, algosettings, iter1, mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)

        strn = f"\nSummary of 5-fold  of ucf sport database for action recognition \n"
        display_summary_to_screen(strn, algosettings, iter1, mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)
        f.close()
    if named==5:
        abc = np.load(DB)
        training_samples = abc['training_samples']
        test = abc['test']
        train_label = abc['train_label']
        test_label = abc['test_label']
        features = np.hstack((training_samples, test))
        labels = np.hstack((train_label, test_label))
        rcounter=0
        counter=0
        meand=0.0
        averageTrainTime=0.0
        averageTrainRateTime=0.0
        averageTestRateTime=0.0
        mean=0.0
        iter1=0
        sl=[]
        fh=0
        while counter!=10:
            strn=""
            training_samples, train_label, test, test_label  =fifteen_scene(features, labels, counter)
            algosettings,iter, accuracy, trainTime, trainRateTime, testRateTime, initDictSize, finalDictSize = JDCSW_training(stp,batchsize,DBdisc,dicSize,iiii,training_samples, train_label, test,test_label,k0,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw)
            sl = sl + [accuracy * 100]
            iter1 = iter1 + iter
            counter = counter + 1
            rcounter = rcounter + 1
            mean = mean + (accuracy * 100)
            smean=mean/rcounter
            meand = meand + finalDictSize
            averageTrainTime = averageTrainTime + trainTime
            averageTrainRateTime = averageTrainRateTime + trainRateTime
            averageTestRateTime=averageTestRateTime+testRateTime
            f = open(Code_DPath+resultFile, foperation)
            if fh==0:
                f.write("\n\n\n#####Experiment of fifteen scene database for scene categorization ("+str(datetime.now())+")###\n")
                fh=1
            strn = f"\n===Experiment No. {rcounter:5d} of fifteen scene  for scene categorization===\n"
            write_iteration_detail_to_file(f, strn, algosettings, accuracy, trainTime, trainRateTime, testRateTime, initDictSize,finalDictSize, smean)
            display_iteration_detail_to_screen(strn, algosettings, accuracy, trainTime, trainRateTime, testRateTime, initDictSize,finalDictSize, smean)
            f.close()
        mean=mean/rcounter
        sl=np.asarray(sl)
        sdev=sum((sl-mean)**2)
        sdev=np.sqrt(sdev/rcounter)
        meand=meand/rcounter
        averageTrainTime = averageTrainTime/rcounter
        averageTrainRateTime = averageTrainRateTime/rcounter
        averageTestRateTime=averageTestRateTime/rcounter
        iter1=iter1/rcounter
        f = open(Code_DPath+resultFile, foperation)
        strn = f"\n==============Summary of fifteen scence dataset for scence categorization============\n"
        write_summary_to_file(f, strn, algosettings, iter1, mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)

        strn = f"\n==============Summary of fifteen scence dataset for scence categorization============\n"
        display_summary_to_screen(strn, algosettings, iter1, mean, sdev, meand, averageTrainTime,averageTrainRateTime, averageTestRateTime)
        f.close()
    return 0
def initDSeqSelection(train, label, k):
    m, n = train.shape
    if k < n + 1:
        ichh = list(range(k))
    elif k > n:
        a = list(range(n))
        b = list(range(k - n))
        ichh = a + b
    Dinit = train[:, ichh]
    label = label[:, ichh]
    mn = np.matlib.repmat(np.mean(Dinit, 0), m, 1)
    sb = np.subtract(Dinit, mn)
    Dinit = sk.normalize(sb, norm='l2', axis=0)
    return Dinit, label
def initClassifier(A, label):
    lmbda = 1
    C = A.dot(A.T) + lmbda * np.eye(A.shape[0])
    W = np.linalg.inv(C).dot(A.dot(label.T))
    W = W.T
    W = np.subtract(W, np.matlib.repmat(np.mean(W,0), W.shape[0], 1))
    return W
def ompAlgo(Di,train_s,sparsity):
    np.random.seed(0)
    X = train_s
    X = np.asfortranarray(X, dtype=np.float64)
    D = Di
    D = np.asfortranarray(D / np.tile(np.sqrt((D * D).sum(axis=0)), (D.shape[0], 1)), dtype=np.float64)
    eps = 1.0
    numThreads = -1
    Alpha_init=spams.omp(X, D,L=sparsity,return_reg_path=False).toarray()
    return Alpha_init
def classification(D, W, data, Hlabel, sparsity):
    Dn = sk.normalize(D, norm='l2', axis=0)
    A=ompAlgo(Dn, data, sparsity)
    forAnorms = np.matlib.repmat(np.sqrt(np.sum(D ** 2, axis=0)), A.shape[1], 1)
    A = A / forAnorms.T
    err = []
    prediction = []
    estMat=W.dot(A)
    estIndices=np.argmax(estMat,0)
    refIndices=np.argmax(Hlabel,0)
    matchnum=np.sum(estIndices==refIndices)
    accuracy = (matchnum*1.0) / (data.shape[1]);
    return prediction, accuracy, err
def classification_Ensemble(D, W, data, Hlabel, sparsity):
    print("Inside classificationTemp")
    ii=0
    estIndicesMatrix=0
    f=0
    err = []
    prediction = []
    for i in range(len(D)):
        Dn = sk.normalize(D[i], norm='l2', axis=0)  # normc(D); %To take advantage of Cholesky based OMP
        A=ompAlgo(Dn, data, sparsity)
        forAnorms = np.matlib.repmat(np.sqrt(np.sum(D[i] ** 2, axis=0)), A.shape[1], 1)
        A = A / forAnorms.T  # To adjust the sparse codes according to D's normalization
        
        estMat=W[i].dot(A)
        estIndices=np.argmax(estMat,0)
        if f==0:
            f=1
            estIndicesMatrix=estIndices
        else:
            estIndicesMatrix=np.vstack((estIndicesMatrix, estIndices))
    print(type(estIndicesMatrix))
    print(type(stats.mode(estIndicesMatrix, axis=0)))
    estIndicesOverAll=list(stats.mode(estIndicesMatrix, axis=0)[0])[0]
    print(type(estIndicesOverAll))
    refIndices=np.argmax(Hlabel,0)
    print(refIndices)
    print(estIndicesOverAll)
    print(refIndices)
    matchnum=np.sum(estIndicesOverAll==refIndices)
    print(matchnum)
    print(data.shape[1])
    accuracy = (matchnum*1.0) / (data.shape[1]);
    return prediction, accuracy, err
########################################################################################################
####(Functions for Gibb Sampling)#####
def sample_D(Xd, D, Sd, g_eps_d, Z, Hw, W, g_eps_w, batchsize):
    m = Xd.shape[0]
    c = Hw.shape[0]
    K = D.shape[1]
    print("looping in D ")
    indexAtoms=list()
    if batchsize!=0 and batchsize<=K:
        batchsize=batchsize
    else:
        batchsize=K
    for i in range(0,K, batchsize):
        if (i+batchsize)<=K:
            indexAtoms=list(range(i,i+batchsize))
        else:
            indexAtoms=list(range(i,K))
        Xdd = Xd + D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hwd = Hw + W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Sds=np.sum((Z[:,indexAtoms]*Sd[:,indexAtoms])**2,0)[np.newaxis,:]
        sig_Dk = 1 / (g_eps_d * Sds + m)
        mu_Dk = g_eps_d * sig_Dk * ((Xd.dot(Z[:,indexAtoms]*Sd[:,indexAtoms]))+(D[:,indexAtoms]*Sds))
        D[:,indexAtoms] = (mu_Dk + np.random.randn(D.shape[0],len(indexAtoms)) * np.sqrt(sig_Dk))
        sig_Dk_w = 1 / (g_eps_w * Sds + c)
        mu_Dk_w = g_eps_w * sig_Dk_w * ((Hw.dot(Z[:,indexAtoms]*Sd[:,indexAtoms]))+(W[:,indexAtoms]*Sds))
        W[:,indexAtoms] = (mu_Dk_w + np.random.randn(W.shape[0],len(indexAtoms)) * np.sqrt(sig_Dk_w))
        Xd = Xdd - D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hw = Hwd - W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
    return Xd, D, Hw, W
def sample_ZS(Xd, D, Sd, Z, Pid, g_sd, g_eps_d, Hw, W, g_sw, g_eps_w,batchsize):
    K = D.shape[1] 
    print("looping in ZS " ) 
    indexAtoms=list()
    if (batchsize!=0 and batchsize<=K):
        batchsize=batchsize
    else:
        batchsize=K
    for i in range(0,K, batchsize):
        if (i+batchsize)<=K:
            indexAtoms=list(range(i,i+batchsize))
        else:
            indexAtoms=list(range(i,K))
        Xdd = Xd + D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hwd = Hw + W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        DTD = np.sum(D[:, indexAtoms] ** 2,0)[np.newaxis,:]
        WTW = np.sum(W[:, indexAtoms] ** 2,0)[np.newaxis,:]
        sigS1d = 1. / (g_sd + (g_eps_d*(Z[:, indexAtoms]**2) * DTD+g_eps_w*(Z[:, indexAtoms]**2) * WTW))
        SdM=sigS1d * (g_eps_d * Z[:, indexAtoms]*((Xd.T).dot(D[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*DTD))+g_eps_w * Z[:, indexAtoms]*((Hw.T).dot(W[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*WTW)))
        Sd[:, indexAtoms] = np.random.randn(Sd.shape[0], len(indexAtoms)) * np.sqrt(sigS1d) + SdM
        temp1 = - 0.5 * g_eps_d * ((Sd[:, indexAtoms] ** 2) * DTD - 2 * Sd[:, indexAtoms] * ((Xd.T).dot(D[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*DTD)))
        temp2 = - 0.5 * g_eps_w * ((Sd[:, indexAtoms] ** 2) * WTW - 2 * Sd[:, indexAtoms] * ((Hw.T).dot(W[:, indexAtoms])+((Z[:, indexAtoms]*Sd[:, indexAtoms])*WTW)))
        temp =Pid[:, indexAtoms]*np.exp(temp1+temp2)
        if not np.isnan(np.sum(temp)):
            A=np.random.rand(Z.shape[0],len(indexAtoms))
            B=Z[:, indexAtoms]
            B[A >  ((1 - Pid[:, indexAtoms]) / (temp + 1 - Pid[:, indexAtoms]))] = 1
            B[A <= ((1 - Pid[:, indexAtoms]) / (temp + 1 - Pid[:, indexAtoms]))] = 0 
            Z[:, indexAtoms]=B
        Xd = Xdd - D[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)
        Hw = Hwd - W[:, indexAtoms].dot(Z[:, indexAtoms].T*Sd[:, indexAtoms].T)      
    return Xd, Hw, Sd, Z
def sample_Pi(Z, train_label,Pid,Pi_C, a0, b0,Xdclist):
    sumZ=train_label.dot(Z)
    K = Z.shape[1]
    Pi_C = np.random.beta((sumZ + (a0*1.0 / K)), ((b0 * 1.0*(K - 1) / K) + (np.sum(train_label, 1)[:,np.newaxis] - sumZ)))
    Pid=train_label.T.dot(Pi_C)
    return Pid,Pi_C
def sample_g_s(S,train_label,c0, d0, Z, g_s, Xdclist):
    a1 = c0 + 0.5 * (np.sum(train_label,1))[:,np.newaxis] * Z.shape[1]
    a2 = d0 + 0.5 * (np.sum((train_label.dot(S*S)), 1))[:, np.newaxis]
    d=np.random.gamma(a1, 1. / a2)
    g_s =train_label.T.dot(d)        
    return g_s
def sample_g_eps(X_k, e0, f0):
    e = e0 + 0.5 * X_k.shape[0] * X_k.shape[1]
    f = f0 + 0.5 * np.sum(X_k ** 2)
    g_eps = np.random.gamma(e, 1. / f)
    return g_eps
######################Gibbs Sampling main Function######################################    
def gibb_sampling(stp,batchsize,training_samplesm, train_labelm,test,test_label,dicSize,pars,gibi,k0,iiii,DBdisc,g_eps_dd,g_eps_ww,g_sdd,g_sww):
    f=0
    a0 = pars['a0']
    b0 = pars['b0']
    c0 = pars['c0']
    d0 = pars['d0']
    e0 = pars['e0']
    f0 = pars['f0']
    c=0
    trainssamplescount=0
    gibi2=0
    classes=train_labelm.shape[0]
    gibi2=np.math.floor(training_samplesm.shape[1]/(train_labelm.shape[0]*1.0))# (trainssamplescount*1.0))
    ii=0
    DD=list()
    WW=list()
    #stp=100
    lenl=np.math.floor(train_labelm.shape[1]/(train_labelm.shape[0]*1.0))
    step=stp
    if stp==0:
        lenl=1
        step=1
    for ii in range(0,lenl,step):
        iia=ii
        if ii==0:
            iia=1
        print(str(np.math.ceil((iia)/(1.0*step))) + " off " + str(int(lenl/(1.0*step))))
        trainColIndex=list()
        if stp!=0:
            b = np.arange(0, train_labelm.shape[1])
            for c in range(train_labelm.shape[0]):
                classIndex=[x0 for x0, val in enumerate(b) if train_labelm[c,val]!=0]
                if ii<=(len(classIndex)-step):
                    trainColIndex = trainColIndex + classIndex[ii:(ii+step)]
                elif ii<len(classIndex):
                    trainColIndex = trainColIndex + classIndex[ii:len(classIndex)]+ classIndex[0:step-(len(classIndex)-ii)]
                else:
                    trainColIndex = trainColIndex + classIndex[0:step]
        else:
            trainColIndex=list(range(train_labelm.shape[1]))
        training_samples=np.copy(training_samplesm[:,trainColIndex])
        train_label=train_labelm[:,trainColIndex]
        training_samples = sk.normalize(training_samples, norm='l2', axis=0)
        if dicSize!=0:
            initDictSize=dicSize
        else:
            initDictSize = int(np.floor(1.25 * training_samples.shape[1]))
        Dinit, label_init = initDSeqSelection(training_samples, train_label, initDictSize)
        Alpha_init = ompAlgo(Dinit, training_samples, sparsity)
        Winit = initClassifier(Alpha_init, train_label)
        D = Dinit
        W = Winit
        Sd = Alpha_init
        K=initDictSize
        DLiterations = pars['DLiterations']
        Xd = np.copy(training_samples)
        g_sd = g_sdd*np.ones((Xd.shape[1],1), dtype=np.float64)
        g_sw = g_sww*np.ones((Xd.shape[1],1), dtype=np.float64)
        g_eps_d=g_eps_dd
        g_eps_w=g_eps_ww
        m, N = Xd.shape
        Hw = np.copy(train_label)
        labels=np.copy(train_label)
        Xdclist=list()
        for c in range(labels.shape[0]):
            Xdc=[x0 for x0,val in enumerate(labels[c,:]) if val!=0]
            Xdclist=Xdclist+[Xdc]
        classes, N = Hw.shape
        Pid = 0.5 * np.ones((Xd.shape[1], D.shape[1]), dtype=np.float64)
        Pi = 0.5*np.ones((1,D.shape[1]), dtype=np.float64)
        Pi_C = np.matlib.repmat(Pi, classes, 1)
        Z = np.copy(Sd)
        Z[Z !=0] = 1
        Z = Z.T
        Sd = Sd.T
        Xd = training_samples - D.dot(Z.T*Sd.T)
        Hw = labels - W.dot(Z.T*Sd.T)
        print('\nBayesian Inference using Gibbs sampling........................\n')
        DLiterations=gibi
        diffc=0
        iter=0
        itcounter=0
        while iter<gibi:
            iter=iter+1
            Xd, D, Hw, W = sample_D(Xd, D, Sd, g_eps_d, Z, Hw, W, g_eps_w, batchsize)
            Xd, Hw, Sd,Z = sample_ZS(Xd, D, Sd, Z, Pid, g_sd, g_eps_d, Hw, W,g_sw, g_eps_w,batchsize)
            Pid,Pi_C = sample_Pi(Z,labels,Pid,Pi_C, a0, b0, Xdclist)
            g_eps_dtemp=sample_g_eps(Xd, e0, f0)
            if not np.isnan(g_eps_dtemp):
                g_eps_d = g_eps_dtemp
            g_eps_wtemp = sample_g_eps(Hw, e0, f0)
            if not np.isnan(g_eps_wtemp):
                g_eps_w = g_eps_wtemp
            g_sd = sample_g_s(Sd,labels,c0, d0, Z, g_sd,Xdclist)
            ps=np.sum(Pi_C,axis=0)
            Pidex = [x0 for x0, val in enumerate(ps) if val >k0]
            if (iter >= 0 and iiii!=0 and (D.shape[1]-len(Pidex))>0):
                Pidexr=list(set(range(D.shape[1])) - set(Pidex))
                Xd = Xd + D[:,Pidexr].dot(Z[:,Pidexr].T*Sd[:,Pidexr].T)
                Hw = Hw + W[:,Pidexr].dot(Z[:,Pidexr].T*Sd[:,Pidexr].T)
                D=D[:,Pidex]
                W=W[:,Pidex]
                Z = Z[:, Pidex]
                Pid=Pid[:,Pidex]
                Sd = Sd[:, Pidex]
                Pi_C=Pi_C[:,Pidex]
            print('Gibb sampling iter #:' + str(iter) + '/' +str(gibi) + '   Dict. size:' + str(D.shape[1]))
            if iter%5==0:
                print(DBdisc)
            current=D.shape[1]
        DD=DD+[D]
        WW=WW+[W]
    return DD,WW,gibi,initDictSize
################################Gibbs Sampling calling function#############################################################################
def JDCSW_training(stp,batchsize,DBdisc,dicSize,iiii,training_samples, train_label, test, test_label,k0,gibi,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw):
    test = sk.normalize(test, norm='l2', axis=0)
    starttime = time.time()  # tic
    S = list()
    for i in range(train_label.shape[0]):
        xdumy = train_label[i, :] > 0
        x = train_label[i, :][xdumy == True]
        s = len(x)
        S = np.append(s, S)
    ss = S.min()
    a00 = ss / 4.
    b00 = ss / 4.
    
    if a0!=0:
        aaa=1
    else:
        a0=ss / 4.
    if b0!=0:
        aaa=1
    else:
        b0=ss / 4.
    DLiterations = 10
    pars = {'a0': a0, 'b0': b0, 'c0': c0, 'd0': d0, 'e0': e0, 'f0': f0,'DLiterations': DLiterations}
    starttime = time.time()
    D,W, gibi,initDictSize= gibb_sampling(stp,batchsize,training_samples, train_label,test,test_label,dicSize,pars,gibi,k0,iiii,DBdisc,g_eps_d,g_eps_w,g_sd,g_sw)
    algosettings = "Date and time = " + str(datetime.now()) + ", a0 = " + str(a0) + ", b0 = " + str(b0) + ", c0 = " + str(c0) + ", d0 = " + str(d0) + ", e0 = " + str(e0) + ", f0 = " + str(f0) + ", sparsity = " + str(sparsity) + ", g_eps_d = " + str(g_eps_d) + ", g_eps_w = " + str(g_eps_w) + ", Iterations  = " +  str(gibi)
    summm=0.0
    for ia in range(len(D)):
        summm=summm+D[ia].shape[1]
    finalDictSize = np.math.ceil(summm/len(D))
    endtime = time.time()  # trainTime
    trainTime = endtime - starttime
    print('\nClassification...\n')
    starttime = time.time()  # tic
    if stp==0:
        prediction, accuracy, err = classification(D[0], W[0], test, test_label, sparsity)
    else:
        prediction, accuracy, err = classification_Ensemble(D, W, test, test_label, sparsity)
    endtime = time.time()
    testTime = endtime - starttime  # testTime
    return algosettings,gibi,accuracy,trainTime, trainTime/train_label.shape[1], testTime / test_label.shape[1], initDictSize, finalDictSize

Code_DPath="./Results/"
f=0
while f<5:
    f=f+1
    if f==1:
        DBdisc="\n===================JDCSW (Proposed) training on Extended Yale database for face recognition====================="
        resultFile="JDCSW_YaleB_Result.txt"
        k0=1.e-6
        gibi=500
        sparsity=25
        a0=0
        b0=0
        c0=1.e-6
        d0=1.e-6
        e0=1.e-6
        f0=1.e-6
        g_eps_d = 1.e+9
        g_eps_w = 1.e+9 
        g_sd = 1.0
        g_sw = 1.0
        stp=0
        batchsize=5
        q=JDCSW_main(1,1,0,'./Data/ExtendedYaleB.mat',Code_DPath,resultFile,DBdisc,gibi,k0,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw,stp,batchsize,foperation='a+')
    if f==2:
        DBdisc="\n===================JDCSW (Proposed) training on AR database for face recognition================================="
        resultFile="JDCSW_ARdat_Result.txt"
        k0=1.e-6
        gibi=500
        sparsity=40
        a0=0
        b0=0
        c0=1.e-6
        d0=1.e-6
        e0=1.e-6
        f0=1.e-6
        g_eps_d = 1.e+9
        g_eps_w = 1.e+9
        g_sd = 1.0
        g_sw = 1.0
        stp=0
        batchsize=5
        q=JDCSW_main(2,1,0,'./Data/ARdat.npz',Code_DPath,resultFile,DBdisc,gibi,k0,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw,stp,batchsize,foperation='a+')
    if f==3:
        DBdisc="\n===================JDCSW (Proposed) training on Caltech Database for object classification======================="
        resultFile="JDCSW_Caltech_Result.txt"
        k0=1.e-4
        gibi=500
        sparsity=70
        a0=0 
        b0=0
        c0=1.e-6
        d0=1.e-6
        e0=1.e-6
        f0=1.e-6
        g_eps_d = 1.e+9
        g_eps_w = 1.e+9
        g_sd =1.0
        g_sw =1.0
        stp=0
        batchsize=10
        q=JDCSW_main(3,1,0,'./Data/caltechData.npz',Code_DPath,resultFile,DBdisc,gibi,k0,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw,stp,batchsize,foperation='a+')
    if f==4:
        DBdisc="\n===================JDCSW (Proposed) training on UCF database for action recognition=============================="
        resultFile="JDCSW_ucf_Result.txt"
        k0=1.e-6
        gibi=500
        sparsity=30
        a0=0
        b0=0
        c0=1.e-6
        d0=1.e-6
        e0=1.e-6
        f0=1.e-6
        g_eps_d = 1.e+12
        g_eps_w = 1.e+12
        g_sd = 1.0
        g_sw = 1.0
        stp=0
        batchsize=1
        q=JDCSW_main(4,1,0,'./Data/ucfdata.npz',Code_DPath,resultFile,DBdisc,gibi,k0,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw,stp,batchsize,foperation='a+')
    if f==5:
        DBdisc="\n===================JDCSW (Proposed) training on 15 scence categories database for scence category classification======================"
        resultFile="JDCSW_15SceneCat_Result.txt"
        k0=1.e-6
        gibi=500
        sparsity=30
        a0=0
        b0=0
        c0=1.e-6
        d0=1.e-6
        e0=1.e-6
        f0=1.e-6
        g_eps_d = 1.e+9
        g_eps_w = 1.e+9
        g_sd = 1.0
        g_sw = 1.0
        stp=0
        batchsize=5
        q = JDCSW_main(5, 1, 0,'./Data/spatialpyramidfeatures4scene15.npz',Code_DPath,resultFile,DBdisc,gibi,k0,sparsity,a0,b0,c0,d0,e0,f0,g_eps_d,g_eps_w,g_sd,g_sw,stp,batchsize,foperation='a+')
