import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier

from global_settings import *
import csv

def get_f1_score(X_train, y_train, X_test, y_test, model):

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    X_train = Normalizer().fit_transform(X_train)
    X_test = Normalizer().fit_transform(X_test)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1score = metrics.f1_score(y_test, preds, average='macro') * 100

    print(f'f1-score on features: {f1score} \n')

    return f1score


def feature_sum(vec_list):
    vec_list = np.array(vec_list)
    vec_sum = vec_list[0]
    try:
        for idx in range(1, len(vec_list)):
            vec_sum += vec_list[idx]
    except:
        print(vec_list)
        exit()
    return vec_sum.tolist()


def mean_feature_sum(vec_list):
    n = len(vec_list)
    vec_list_sum = np.array(feature_sum(vec_list))
    vec_list_sum = vec_list_sum/n
    return vec_list_sum.tolist()


def get_feature_vector_fromwords(instance_words):

    feature_vector = []
    for line in instance_words:
        for word in line:
            feature_vector.append(words_embedding_dict[word])

    return feature_vector


def get_feature_data(main_df):

    activity_subject_df_train = main_df[['activityID', 'subject_id']].drop_duplicates().values.astype('int32')
    col_names = ['subject_id', 'activityID',
                 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']

    X = []
    y = []

    for instance in activity_subject_df_train:
        activity = instance[0]
        subject = instance[1]
        instance_data = main_df.loc[(main_df['subject_id'] == subject) & (main_df['activityID'] == activity)]
        instance_words = instance_data[col_names[2:]].values
        instance_feature_vector = get_feature_vector_fromwords(instance_words)

        X_avg = mean_feature_sum(instance_feature_vector)
        X.append(np.array(X_avg).reshape(-1, 1))
        y.append(activity)
    
    return np.array(X), np.array(y)


def load_data():

    X_train, y_train = get_feature_data(sensory_words_traindf)
    X_test, y_test = get_feature_data(sensory_words_testdf)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]).astype('float32')
    y_train = y_train.astype('int32')
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]).astype('float32')
    y_test = y_test.astype('int32')

    return X_train, y_train, X_test, y_test


def get_svm_wordembds():

    print('................ svm classfier .........')

    model = svm.SVC(kernel='poly')
    return model


def get_rfc_wordembds():

    print('................ rfc classifer .........')

    model = RandomForestClassifier(n_estimators=200)
    #model = RandomForestClassifier(n_estimators=450,criterion='entropy', max_depth=20, min_samples_split=6, min_samples_leaf=2, max_features='log2', bootstrap=True , n_jobs=-1, random_state=123)

    return model


def perform_classification_on_features(cluster_cnts):

    print('................ after clustering.........')

    X_train, y_train, X_test, y_test = load_data()
    svc_model = get_svm_wordembds()
    svc_f1_score = get_f1_score(X_train, y_train, X_test, y_test, svc_model)
    rfc_model = get_rfc_wordembds()
    rfc_f1_score = get_f1_score(X_train, y_train, X_test, y_test, rfc_model)

    clf_data = [cluster_cnts, svc_f1_score, rfc_f1_score]
    with open("output/clf_performance_data.csv", "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(clf_data)


def perform_classification_on_rawfeatures(X_train, y_train, X_test, y_test):

    print('................ before clustering.........')

    svc_model = get_svm_wordembds()
    get_f1_score(X_train, y_train, X_test, y_test, svc_model)
    rfc_model = get_rfc_wordembds()
    get_f1_score(X_train, y_train, X_test, y_test, rfc_model)