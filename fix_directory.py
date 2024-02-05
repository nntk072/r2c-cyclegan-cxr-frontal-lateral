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
    
# count = 0
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
                        elif 'frontal' in view_path and 'valid' in old_folder[i]:
                            shutil.copy(view_path, os.path.join(new_path, new_folder[2], patient + '_' + study + '_' + view[j] + '.jpg')) 
                        elif 'lateral' in view_path and 'valid' in old_folder[i]:
                            shutil.copy(view_path, os.path.join(new_path, new_folder[3], patient + '_' + study + '_' + view[j] + '.jpg'))
                except:
                    pass
        # count += 1
        # if count > 100:
        #     break