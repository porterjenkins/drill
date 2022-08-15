# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 19:47:47 2022

@author: bryan
"""
# %%
import pandas as pd
import numpy as np
from os.path import exists
import random
# import array as arr
# import datetime

# Get start and end date information
dates = pd.read_csv('Dates.csv')

# Specify Train/Test/Validation Split
train = 0.8
test = 0.1
val = 0.1
test_increment = train+test


master_exists = exists('master_key.csv')
if master_exists:
    # Old Master Key filename used to determine whether a new segment needs to be created
    old_master_key = pd.read_csv('master_key.csv',header=None)

# %%
number_of_dates = 25
dates_to_run = range(number_of_dates)

for z in dates_to_run:
    include_in_training_data = dates.iloc[z,25]
    if include_in_training_data:
        start_date = dates.iloc[z,1]
        end_date   = dates.iloc[z,2]
        
        # open label data excel file
        label_dir = 'C:/Users/bryan/Documents/1 PhD/drill/Raw_Data/Data-2022-'+start_date+'_to_2022-'+end_date+'/label-data-2022-'+start_date+'_to_2022-'+end_date+'.xlsx'
        label_data = pd.read_excel(label_dir,usecols="A:L,O:P,R",skiprows=[1])
        
        
        month = label_data.iloc[:,0].dt.month.values
        day = label_data.iloc[:,0].dt.day.values
        date_info = np.hstack((label_data.iloc[:,2:5].values,label_data.iloc[:,6:9].values))
        num_users = label_data.iloc[:,9].values
        start_times = label_data.iloc[:,10].values
        stop_times = label_data.iloc[:,11].values
        duration = label_data.iloc[:,12].values
        time_between = label_data.iloc[:,13].values
        label = label_data.iloc[:,14].values # same thing as user type
        
        # import all sensor data
        sensor_data_dir = 'C:/Users/bryan/Documents/1 PhD/drill/Raw_Data/Data-2022-'+start_date+'_to_2022-'+end_date+'/data-2022-'+start_date+'_to_2022-'+end_date+'.csv'
        sensor_data = pd.read_csv(sensor_data_dir,header=None)
        sensor_data = sensor_data.values
        
        num_users_total = month.size
        num_users = np.nan_to_num(num_users,nan=1)
    
        # %%
        # get start and stop time indices
        start_times_i = np.zeros(num_users_total,dtype=int)
        stop_times_i = np.zeros(num_users_total,dtype=int)
        n = len(sensor_data)
        a = 0
        b = 0
        for i in range(n):
            if(sensor_data[i,0]>stop_times[b]):
                stop_times_i[b] = i-1;
                b = b + 1
                
            if(a>num_users_total-1):
                break
            
            if(sensor_data[i,0]>start_times[a]):
                start_times_i[a] = i
                a = a + 1
    
        #%%
    
        for j in range(num_users_total):
        # for j in range(1): # DEBUG line
            if label[j]<4 and duration[j]>0 and time_between[j]>0: #filter through bad users
                # Save key to the master user file key csv the...
                # - Col 0: user ID (month, day, hour, minute, second of start time of pumping_user label (M) i.e. 03-01_13-22-58_1
                # - Col 1: label (1, 2, or 3) 
                # - Col 2: pumping month (MM)
                # - Col 3: pumping day (DD)
                # - Col 4: pumping start hour (HH)
                # - Col 5: pumping start minute (MM)
                # - Col 6: pumping start second (SS)
                # - Col 7: pumping duration (Seconds)
                # - Col 8: time between users (Seconds)
                # - Col 9:number of users that pumped during this time segment
                # df.to_csv('master_key.csv',mode='a',index=False,header=False)
                random_splitter = random.uniform(0, 1)
                if random_splitter < train:
                    split_category = 1
                elif random_splitter >= train and random_splitter < test_increment:
                    split_category = 2
                elif random_splitter >= test_increment:
                    split_category = 3
                
                user_ID_string = str(month[j])+'-'+str(day[j])+'_'+str(date_info[j,0])+'-'+str(date_info[j,1])+'-'+str(date_info[j,2])+'_'+str(label[j])
                #                                    C0              C1        C2        C3      C4              C5              C6              C7           C8               C9            C10
                user_info = np.expand_dims(np.array([user_ID_string, label[j], month[j], day[j], date_info[j,0], date_info[j,1], date_info[j,2], duration[j], time_between[j], num_users[j], split_category]),axis=0)
                user_info_df = pd.DataFrame(user_info)
                if j == 0 and z==dates_to_run[0]:
                    user_info_df.to_csv('master_key.csv',index=False,header=False,mode='w')
                else:
                    user_info_df.to_csv('master_key.csv',mode='a',index=False,header=False)
                # extract dt, HE1, HE2, and HE3 for that particular user
                sub_array = []
                sub_array = sensor_data[start_times_i[j]:stop_times_i[j],:]
                user_df = pd.DataFrame(sub_array)    
                
                string = 'Segments/'+user_ID_string+'.csv'
                user_df.to_csv(string,index=False,header=False)

# Create train/test/val keys from master key
master_key = pd.read_csv('master_key.csv',header=None)

key_train = master_key.loc[master_key.iloc[:,10]==1]
key_test = master_key.loc[master_key.iloc[:,10]==2]
key_val = master_key.loc[master_key.iloc[:,10]==3]

key_train.to_csv('key_train.csv',index=False,header=False,mode='w')
key_test.to_csv('key_test.csv',index=False,header=False,mode='w')
key_val.to_csv('key_val.csv',index=False,header=False,mode='w')
            # Can use this below if I don't want to save new segments every time
            # # if the segment doesn't exist already, save the pumping data to a csv in the Segments folder and to the master_key
            # if master_exists:
                
            # if not old_master_key.iloc[:,0].eq(user_ID_string).any():
            #     print('Entered IF')

    
