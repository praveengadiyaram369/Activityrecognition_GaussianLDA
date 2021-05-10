import os
import collections
from sklearn import svm, metrics
import pandas as pd
import numpy as np
from scipy import stats 
import statistics
import sys
import csv
from sklearn.preprocessing import Normalizer

def load_train_test_data(input_file_path, col_names):

    main_df = pd.read_csv(input_file_path, names=col_names)
    main_df = main_df.astype({'subject_id': int, 'activityID': int})
    

    return main_df

def Extract_features(main_df, window_length, window_overlap):

    max_window_index = len(main_df.index)
    sequence_names = main_df.columns.tolist()
    num_of_subsequences = len(sequence_names)
    sub_sequences = [[] for x in range(num_of_subsequences)]

    window_index = 0

    while window_index <= (max_window_index - window_length):

        activity_sequence = main_df[sequence_names[1]
                                    ][window_index:window_index+window_length].tolist()
        subject_sequence = main_df[sequence_names[0]
                                   ][window_index:window_index+window_length].tolist()
        if len(set(activity_sequence)) == 1:
            sub_sequences[1].append(activity_sequence[0])
            sub_sequences[0].append(subject_sequence[0])

            for idx in range(2, num_of_subsequences):
                sub_sequences[idx].append(
                    main_df[sequence_names[idx]][window_index:window_index+window_length].tolist())

        window_index += window_overlap
    np_sequences = np.asarray(sub_sequences[2:])

    doc_df = pd.DataFrame(columns=col_names[:1])
    doc_df['subject_id'] = sub_sequences[0]
    doc_df['activityID'] = sub_sequences[1]
    df=pd.DataFrame()
    for j in range(0,len(np_sequences)):
        subsequences=np_sequences[j]
        data=[]
        for i in range(0,len(subsequences)):
            mean=sum(subsequences[i])/len(subsequences[i])
            std=statistics.stdev(subsequences[i])
            skewness=stats.skew(subsequences[i])
            q3,q2,q1=np.percentile(subsequences[i],[75 ,50,25])        
            iqr=q3-q1
            slope=subsequences[i][0]-subsequences[i][len(subsequences[i])-1]
            energy=sum(subsequences[i]*subsequences[i])/len(subsequences[i])
            data.append([mean,std,skewness,iqr,slope,energy])
            data1=np.array(data)
        raw_features=zip(data1)
        Id=pd.DataFrame(raw_features,columns=[j])
        df = pd.concat([df,Id],axis=1) 
    doc_df=pd.concat([doc_df,df],axis=1)
    Feature_Vectors=[]
    final_df=pd.DataFrame()

    for subject in doc_df['subject_id'].unique():
    
        for activity in doc_df.loc[(doc_df['subject_id'] == subject)]['activityID'].unique():
       
            a=doc_df.loc[(doc_df['subject_id'] == subject)&(doc_df['activityID']==activity)]
        
            b=a[a.columns[2:8]]
        
        
            arr = b.to_numpy().flatten()
            act=activity
            print(act)
            mean_feature_vector=np.sum(arr)/len(arr)
            data=(act,mean_feature_vector)
            df = pd.DataFrame(data) 
            df=df.T
        
            final_df=pd.concat([final_df,df])
    
    y=final_df[0].tolist()
    X=final_df[1].tolist()

    return X,y

def svm_classification(X_train,y_train,X_test,y_test):
    X_train = Normalizer().fit_transform(X_train)
    X_test = Normalizer().fit_transform(X_test)
    model = svm.SVC(kernel='poly').fit(X_train, y_train)
    preds = model.predict(X_train)
    preds_test=model.predict(X_test)
    svm_f1score_train = metrics.f1_score(y_train, preds, average='macro') * 100
    svm_f1score_test=metrics.f1_score(y_test,preds_test,average='macro')*100

    return svm_f1score_train,svm_f1score_test   


if __name__ == '__main__':

    
    window_length = int(sys.argv[1])
    window_overlap = int(sys.argv[2])
    print('\n\n\n')
    print('loading filepaths....')


    train_file_path = os.getcwd() + f'/../../data/output_csv/processed_data_train.csv'
    test_file_path = os.getcwd() + f'/../../data/output_csv/processed_data_test.csv'
    col_names = ['subject_id', 'activityID',
                'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
    print('\n\n\n')
    print('loading train and test data....')

    train_df = load_train_test_data(train_file_path, col_names)
    test_df = load_train_test_data(test_file_path, col_names)

    print('\n\n\n')
    print('Extracting statisticsl features....')

    X_train,y_train = Extract_features(
        train_df, window_length=window_length, window_overlap=window_overlap)
    X_test,y_test = Extract_features(
        test_df, window_length=window_length, window_overlap=window_overlap)
    
    print('\n\n\n')
    print('training svm...')


    f1_train,f1_test=svm_classification(X_train,y_train,X_test,y_test)

    print('\n\n\n')
    print('writing result....')

    data=[window_length,window_overlap,f1_train,f1_test]
    

    with open(os.getcwd() + f'/../../data/result.csv', "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(data)
    print('Finished writing results')

   

