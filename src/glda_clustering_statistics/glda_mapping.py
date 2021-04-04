import pickle
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer


def transform_data(data):

    data = np.transpose(data)
    one_hotencoded_data = np.zeros(data.shape, dtype='int32')
    for idx in range(data.shape[0]):
        max_idx = np.argmax(data[idx])
        one_hotencoded_data[idx][max_idx] = 1

    return np.transpose(one_hotencoded_data)


def topic_doc_mapping(topic_doc_counts_df):

    mapping_dict = {}

    if(topic_doc_counts_df.shape[0] >= topic_doc_counts_df.shape[1]):

        while topic_doc_counts_df.shape[1] != 0:

            max_value = topic_doc_counts_df.values.max()
            max_ind = np.where(topic_doc_counts_df.values == max_value)
            row_ind = max_ind[0][0]
            col_ind = max_ind[1][0]
            docOfMax = topic_doc_counts_df.columns[col_ind]
            topicOfMax = topic_doc_counts_df.index[row_ind]
            mapping_dict[docOfMax] = topicOfMax
            topic_doc_counts_df = topic_doc_counts_df.drop(docOfMax, axis=1)
            topic_doc_counts_df = topic_doc_counts_df.drop(topicOfMax, axis=0)

    else:

        while topic_doc_counts_df.shape[0] != 0:

            max_value = topic_doc_counts_df.values.max()
            max_ind = np.where(topic_doc_counts_df.values == max_value)
            row_ind = max_ind[0][0]
            col_ind = max_ind[1][0]
            docOfMax = topic_doc_counts_df.columns[col_ind]
            topicOfMax = topic_doc_counts_df.index[row_ind]
            mapping_dict[docOfMax] = topicOfMax
            topic_doc_counts_df = topic_doc_counts_df.drop(docOfMax, axis=1)
            topic_doc_counts_df = topic_doc_counts_df.drop(topicOfMax, axis=0)

    print(mapping_dict)

    return mapping_dict


def cluster_gldaoutput(data):

    data = np.transpose(np.array(data))
    data = Normalizer().fit_transform(data)

    model = KMeans(n_clusters=6, random_state=234).fit(data)
    cluster_ids = model.predict(data)
    one_hot_mapping = pd.get_dummies(cluster_ids.astype('str'))

    return np.transpose(one_hot_mapping.values)


def get_activity_topic_mapping(activity_labels, activity_doc_count_index):

    with open("saved_model/table_counts_per_doc.pkl", "rb") as doc:
        data = pickle.load(doc)

    activity_count = []
    # data = cluster_gldaoutput(data)
    data = transform_data(data)

    for activity in activity_labels:
        temp = data[:, activity_doc_count_index[activity]]
        activity_count.append(np.sum(temp, axis=1).tolist())

    activity_count = np.transpose(np.asarray(activity_count))

    topic_index = ['Topic'+str(i) for i in range(len(activity_labels))]

    topic_doc_counts_df = pd.DataFrame(
        activity_count, index=topic_index, columns=activity_labels)
    print(topic_doc_counts_df.T)

    fig, ax = plt.subplots()

    # hide axes
    ax.axis('off')
    ax.axis('tight')

    activity_labels.insert(0, 'mapping')
    topic_doc_counts_df['aa'] = topic_index
    topic_doc_counts_df = topic_doc_counts_df.reindex(
        sorted(topic_doc_counts_df.columns), axis=1)

    ax.table(cellText=topic_doc_counts_df.values,
             colLabels=activity_labels, loc='center')

    fig.tight_layout()

    plt.savefig('output/' +
                'activity_topic_mapping.png', bbox_inches='tight')

    topic_doc_counts_df.drop(['aa'], axis=1, inplace=True)
    activity_labels.pop(0)
    return topic_doc_mapping(topic_doc_counts_df)


if __name__ == "__main__":

    with open("saved_model/table_counts_per_doc.pkl", "rb") as doc:
        data = pickle.load(doc)

    topic_doc_counts_df = pd.DataFrame(data, index=[
                                       'Topic'+str(i) for i in list(range(8))], columns=['Doc'+str(i) for i in list(range(8))])
    print(topic_doc_counts_df)

    topic_doc_mapping(topic_doc_counts_df)
