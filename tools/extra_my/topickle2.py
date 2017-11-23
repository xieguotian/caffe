import pickle
import glob

data_path = 'Imagenet32_train'
all_files = glob.glob(data_path+'/*')
for file_tmp in all_files:
    with open(file_tmp, "rb") as f:
        w = pickle.load(f)
        pickle.dump(w, open(file_tmp+'_p2',"wb"), protocol=2)