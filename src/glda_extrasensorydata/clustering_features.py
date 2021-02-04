import sys
import logging
import logging.config
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from glda_clustering import perform_glda_clustering


def configure_log_settings():

    logging.basicConfig(level=logging.INFO, filename='logs/glda_testing_analysis.log',
                        filemode='a', format='%(name)s - %(levelname)s - %(message)s')
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
    })

    logging.getLogger("GLDA").setLevel(logging.WARNING)
    logging.getLogger("GLDA").propagate = False

    return logging.getLogger('glda_testing_logger')


def generate_words(cluster_cnts):

    prefix = 'W_'
    return [prefix+str(val) for val in range(1, cluster_cnts+1)]


def generate_cluster_words(seq_clusters, cluster_words):

    assigned_words = np.where(
        seq_clusters != 0, seq_clusters, cluster_words[0])
    for idx in range(1, len(cluster_words)):
        assigned_words = np.where(
            seq_clusters != idx, assigned_words, cluster_words[idx])

    return assigned_words


def write_clustering_output(logger, doc_df, features):

    features_filepath = f'clustering_output/featues_from_clustering.txt'
    pd.DataFrame(features).to_csv(features_filepath, index=False, header=False)

    for activity in doc_df['activity_label'].unique():
        output_filepath = f'clustering_output/activity_subseq_' + \
            str(activity) + '.txt'
        doc_df.loc[doc_df['activity_label'] == activity][['element_name']].to_csv(
            output_filepath, sep='\t', index=False, header=False)

    logger.info(
        f'Finished writing features and document files')


def perform_clustering(logger, clustering_algo, cluster_cnts):

    main_df = pd.read_csv('input/sensor_features_lstm_tuned.csv', header=None)
    main_df.iloc[:, 1:] = StandardScaler().fit_transform(
        main_df.iloc[:, 1:].to_numpy())

    activity_data = main_df.iloc[:, 0].values.astype('int32')
    sensor_data = main_df.iloc[:, 1:].values.astype('float32')

    logger.info(
        f'clustering with -- {clustering_algo} algorithm with cluster counts:{cluster_cnts}')

    if clustering_algo == 'gmm':
        gmm = GaussianMixture(n_components=cluster_cnts,
                              random_state=2).fit(sensor_data)
        clusters = gmm.predict(sensor_data)
        features = gmm.means_
    elif clustering_algo == 'kmeans':
        kmeans = KMeans(n_clusters=cluster_cnts,
                        random_state=0).fit(sensor_data)
        clusters = kmeans.labels_
        features = kmeans.cluster_centers_

    logger.info(
        f'Finished clustering')

    cluster_names = generate_words(cluster_cnts)
    doc_df = pd.DataFrame(columns=['activity_label', 'element_name'])
    doc_df['activity_label'] = list(activity_data)
    doc_df['element_name'] = generate_cluster_words(clusters, cluster_names)

    write_clustering_output(logger, doc_df, features)


if __name__ == '__main__':

    logger = configure_log_settings()
    logger.info(
        f'Starting clustering features: usage --> python clustering_features.py _clustering_algo_ _cluster_cnts_')

    clustering_algo = sys.argv[1]
    cluster_cnts = int(sys.argv[2])

    perform_clustering(logger, clustering_algo, cluster_cnts)

    logger.info(
        f'Finished clustering features')

    perform_glda_clustering(logger, clustering_algo, cluster_cnts)
