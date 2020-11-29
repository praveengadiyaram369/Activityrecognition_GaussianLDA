import pandas as pd


def data_preprocessing():

        #path of data file
        path = 'Human_Activity_Recognition/data/sample.dat'
        #Giving column names 
        #considered columns from dataset are
        # timestamp (s)
        # activityID  
        # heart rate (bpm)
        # (IMU hand) 3D-acceleration data (ms-2), scale: ±16g, resolution: 13-bit 
        # (IMU hand) 3D-gyroscope data (rad/s)
        col_names = ['timestamp (s)', 'activityID', 'heart rate (bpm)', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
        #required columns list
        required_cols = [0,1,2,4,5,6,10,11,12]
        #loading data into pandas dataframe
        data = pd.read_csv(path, header=None, names=col_names, sep='\s+', usecols=required_cols, engine='python')
        #printing few records of data
        print(data.head())
        #showing null values count for each column
        print(data.isna().sum())
        #columns list for which we need to perform ouliers operation
        cols = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
        #calculating threshold values to remove outliers which are +3 or -3 std away from mean
        lower = data[cols].quantile(0.01)
        higher  = data[cols].quantile(0.99)
        print(lower,higher)
        # checking and removing outliers
        data = data[((data[cols] < higher) & (data[cols] > lower)).any(axis=1)]
        #printing few records of data
        print(data.head())


if __name__ == '__main__':
        data_preprocessing()