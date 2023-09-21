import os
path1 = r"/media/tx-deepocean/Data/code/TransUNet/data/Synapse/train_npz_3"
with open(r"/media/tx-deepocean/Data/code/TransUNet/lists/lists_Synapse/train_vessels.txt", 'w') as f:
    for sub in os.listdir(path1):
        f.write(sub.split('.')[0])
        f.write('\n')
        print(sub.split('.')[0])