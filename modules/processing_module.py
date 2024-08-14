#%%
from dataclasses import dataclass
import numpy as np
import random

# %% define data class

@dataclass
class EEGdata:
    label: int
    data: float
    Fs: int

#%% files to load

def data_load(allList,grps,Fs):
    dataset=[] # dataset[group][subject]

    for g,grp in enumerate(grps):
        dlist=allList[g]
        dataList=[]

        if grp=='C':
            label=0
        else:
            label=1

        for i in dlist:
            fname='data/raw/'+grp+str(i).zfill(2)+'.csv'
            print(fname)
            rawdata = np.loadtxt(fname, delimiter=',')
            rawdata=np.transpose(rawdata)
            data=EEGdata(label=label,data=rawdata,Fs=Fs)
            dataList.append(data)
        
        dataset.append(dataList)

    return dataset

# %% segment data into 5s and split in train/val/test

def data_prepare(dataset,segLength,datasplit):
    dataLen=dataset[0][0].data.shape[0]
    Fs=dataset[0][0].Fs
    segLen=Fs*segLength
    segTotal=dataLen // segLen

    # Determine the sizes of train, test, and validation sets
    train_size = int(datasplit[0] * segTotal)  # 60% for training
    test_size = int(datasplit[1] * segTotal)   # 20% for testing
    # The rest for validation



    # Set the random seed for reproducibility
    random.seed(42)

    train_data=[]
    val_data=[]
    test_data=[]

    for grp in range(len(dataset)):
        for s in range(len(dataset[grp])):  
            curr_eegdata=dataset[grp][s]          
            data=curr_eegdata.data
            data_segmented=np.vsplit(data,segTotal)
            
            data_segment_list=[]
            for data_seg in data_segmented:
                data_s=EEGdata(label=curr_eegdata.label,data=data_seg.copy(),Fs=curr_eegdata.Fs)
                data_segment_list.append(data_s)
            
            random.shuffle(data_segment_list)

            # Split the list
            train_set = data_segment_list[:train_size]
            test_set = data_segment_list[train_size:train_size + test_size]
            val_set = data_segment_list[train_size + test_size:]

            train_data.extend(train_set)
            val_data.extend(val_set)
            test_data.extend(test_set)

    return train_data,val_data,test_data

#%% separate each EEG channel as independent data
def data_serealize(dataset):
    dataset_ser=[]
    nChan=dataset[0].data.shape[1]
    for data in dataset:
        for c in range(nChan):
            newdata=EEGdata(label=data.label,data=data.data[:,c].copy(),Fs=data.Fs)
            dataset_ser.append(newdata)
    return dataset_ser