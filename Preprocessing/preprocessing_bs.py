# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 19:47:47 2022

@author: bryan
"""
# %%
import pandas as pd
import numpy as np
# import datetime

    # declare directory of the time collection period
start_date_list = ['05-31']
end_date_list   = ['06-02']

# start_date_list = ['05-31']
# end_date_list   = ['06-02']

for z in range(1):

# %%

    start_date =start_date_list[z]
    end_date = end_date_list[z]
    
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
    
    num_users = month.size

# %%
    # get start and stop time indices
    start_times_i = np.zeros(num_users,dtype=int)
    stop_times_i = np.zeros(num_users,dtype=int)
    n = len(sensor_data)
    a = 0
    b = 0
    for i in range(n):
        if(sensor_data[i,0]>stop_times[b]):
            stop_times_i[b] = i-1;
            b = b + 1
            
        if(a>num_users-1):
            break
        
        if(sensor_data[i,0]>start_times[a]):
            start_times_i[a] = i
            a = a + 1

#%%

    for j in range(num_users):
    # for j in range(1): # DEBUG line
        if label[j]<4 and duration[j]>0 and time_between[j]>0: #filter through bad users
            # Save key to the master user file key csv the...
            # - Col 1: user ID (month, day, hour, minute, second of start time of pumping_user label (M) i.e. 03-01_13-22-58_1
            # - Col 2: label (1, 2, or 3) 
            # - Col 3: pumping month (MM)
            # - Col 4: pumping day (DD)
            # - Col 5: pumping start hour (HH)
            # - Col 6: pumping start minute (MM)
            # - Col 7: pumping start second (SS)
            # - Col 8: pumping duration (Seconds)
            # - Col 9: number of users that pumped during this time segment
            # df.to_csv('master_key.csv',mode='a',index=False,header=False)
            user_ID_string = str(month[j])+'-'+str(day[j])+'_'+str(date_info[j,0])+'-'+str(date_info[j,1])+'-'+str(date_info[j,2])+'_'+str(label[j])
            user_info = np.expand_dims(np.array([user_ID_string, label[j], month[j], day[j], date_info[j,0],date_info[j,1],date_info[j,2], duration[j], time_between[j]]),axis=0)
            user_info_df = pd.DataFrame(user_info)
            if j == 0 and z==0:
                user_info_df.to_csv('master_key.csv',index=False,header=False)
            else:
                user_info_df.to_csv('master_key.csv',mode='a',index=False,header=False)
            # extract dt, HE1, HE2, and HE3 for that particular user
            sub_array = []
            sub_array = sensor_data[start_times_i[j]:stop_times_i[j],:]
            user_df = pd.DataFrame(sub_array)    
            
            # save the pumping data to a csv in a folder
            string = 'Segments/'+user_ID_string+'.csv'
            user_df.to_csv(string,index=False,header=False)
    
