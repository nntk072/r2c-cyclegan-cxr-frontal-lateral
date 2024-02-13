import os
import shutil
import re

# Define the path
path = 'datasets/CheXpert-v1.0-small'
new_path = 'datasets'
new_folder = ['trainA', 'trainB', 'testA', 'testB']
old_folder = ['train', 'train', 'valid', 'valid']
view = ['frontal', 'lateral']
old_view = ['view1', 'view2']


for folder in new_folder:
    # Check that the folder exists, if yes, delete it
    if os.path.exists(os.path.join(new_path, folder)):
        shutil.rmtree(os.path.join(new_path, folder))
    os.makedirs(os.path.join(new_path, folder), exist_ok=True)
    
count = 0
for i in range(len(old_folder)):
    old_path = os.path.join(path, old_folder[i])
    # Loop through the patient folder
    for patient in os.listdir(old_path):
        patient_path = os.path.join(old_path, patient)
        # Loop through the study folder
        for study in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study)
            # Loop through the view folder
            for j in range(len(view)):
                try:
                    view_path = os.path.join(study_path, old_view[j] + '_' + view[j] + '.jpg')
                    if len(os.listdir(study_path)) == 2:
                        if 'frontal' in view_path and 'train' in old_folder[i]:
                            shutil.copy(view_path, os.path.join(new_path, new_folder[0], patient + '_' + study + '_' + view[j] + '.jpg')) 
                        elif 'lateral' in view_path and 'train' in old_folder[i]:
                            shutil.copy(view_path, os.path.join(new_path, new_folder[1], patient + '_' + study + '_' + view[j] + '.jpg')) 
                        if 'frontal' in view_path and 'valid' in old_folder[i]:
                            shutil.copy(view_path, os.path.join(new_path, new_folder[2], patient + '_' + study + '_' + view[j] + '.jpg')) 
                        elif 'lateral' in view_path and 'valid' in old_folder[i]:
                            shutil.copy(view_path, os.path.join(new_path, new_folder[3], patient + '_' + study + '_' + view[j] + '.jpg'))
                except:
                    pass
        # count += 1
        # if count > 100:
        #     break
        

# test_folder = ['validA', 'validB']

# count = 0
# # make the valid folder
# for folder in test_folder:
#     if os.path.exists(os.path.join(new_path, folder)):
#         shutil.rmtree(os.path.join(new_path, folder))
#     os.makedirs(os.path.join(new_path, folder), exist_ok=True)

    
#     # move 20% of trainA directory to validA, 20% of trainB directory to validB and end
#     train_path = os.path.join(new_path, new_folder[count])
#     valid_path = os.path.join(new_path, folder)
#     files = os.listdir(train_path)
#     for j in range(int(len(files)*0.2)):
#         shutil.move(os.path.join(train_path, files[j]), os.path.join(valid_path, files[j]))
#     count += 1
        
        