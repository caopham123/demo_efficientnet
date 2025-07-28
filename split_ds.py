import os, shutil

DATASET_PATH = 'dataset'
TRANING_PATH = 'dataset/train'
VALIDATION_PATH = 'dataset/val'

os.makedirs(TRANING_PATH, exist_ok=True)
os.makedirs(VALIDATION_PATH, exist_ok=True)

lst_classes = [cls.name for cls in os.scandir(DATASET_PATH) if cls.is_dir()]
lst_classes.remove('train')
lst_classes.remove('val')
print(lst_classes)

for cls in lst_classes:
    os.makedirs(os.path.join(TRANING_PATH, cls).replace("\\","/"), exist_ok=True)
    os.makedirs(os.path.join(VALIDATION_PATH, cls).replace("\\","/"), exist_ok=True)

    lst_file = [file for file in os.scandir(os.path.join(DATASET_PATH, cls)) if file.is_file() and file.name.endswith(".jpg")]
    idx_split = int(len(lst_file)* .9)

    for idx, file_name in enumerate(lst_file):
        base_name = os.path.basename(file_name)
        print(f"file_name {base_name}")
        CLASS_PATH = os.path.join(DATASET_PATH,cls)
        if idx <= idx_split:
            shutil.copy(os.path.join(CLASS_PATH, base_name).replace("\\","/")
                        , os.path.join((TRANING_PATH+"/"+cls),base_name).replace("\\","/"))
        else:     
            shutil.copy(os.path.join(CLASS_PATH, base_name).replace("\\","/")
                        , os.path.join((VALIDATION_PATH+"/"+cls),base_name).replace("\\","/"))