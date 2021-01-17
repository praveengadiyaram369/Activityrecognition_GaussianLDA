import pickle
import re
import numpy as np
import pandas as pd


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


def get_activity_topic_mapping(activity_labels):

    with open("saved_model/table_counts_per_doc.pkl", "rb") as doc:
        data = pickle.load(doc)

    topic_doc_counts_df = pd.DataFrame(
        data, index=['Topic'+str(i) for i in list(range(len(activity_labels)))], columns=activity_labels)
    print(topic_doc_counts_df.T)

    return topic_doc_mapping(topic_doc_counts_df)


if __name__ == "__main__":

    with open("saved_model/table_counts_per_doc.pkl", "rb") as doc:
        data = pickle.load(doc)

    topic_doc_counts_df = pd.DataFrame(data, index=[
                                       'Topic'+str(i) for i in list(range(8))], columns=['Doc'+str(i) for i in list(range(8))])
    print(topic_doc_counts_df)

    topic_doc_mapping(topic_doc_counts_df)
