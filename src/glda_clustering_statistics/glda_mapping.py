import pickle
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


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


def get_activity_topic_mapping(activity_labels, activity_doc_count_index):

    with open("saved_model/table_counts_per_doc.pkl", "rb") as doc:
        data = pickle.load(doc)
    
    activity_count = []

    for activity in activity_labels:
        temp = data[: , activity_doc_count_index[activity]]
        activity_count.append(np.sum(temp, axis = 1).tolist())

    topic_index = ['Topic'+str(i) for i in list(range(len(activity_labels)))]

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
