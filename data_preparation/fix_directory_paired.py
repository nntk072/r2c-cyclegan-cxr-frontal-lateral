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


count= 0
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
                            # check if the other image is having lateral in their name
                            if 'lateral' in os.listdir(study_path)[1]:
                                shutil.copy(view_path, os.path.join(new_path, new_folder[0], patient + '_' + study + '_' + view[j] + '.jpg')) 
                                count += 1
                        elif 'lateral' in view_path and 'train' in old_folder[i]:
                            # check if the other image is having frontal in their name
                            if 'frontal' in os.listdir(study_path)[0]:
                                shutil.copy(view_path, os.path.join(new_path, new_folder[1], patient + '_' + study + '_' + view[j] + '.jpg')) 
                                count += 1
                        # if 'frontal' in view_path and 'valid' in old_folder[i]:
                        #     # check if the other image is having lateral in their name
                        #     if 'lateral' in os.listdir(study_path)[1]:
                        #         shutil.copy(view_path, os.path.join(new_path, new_folder[2], patient + '_' + study + '_' + view[j] + '.jpg')) 
                        # elif 'lateral' in view_path and 'valid' in old_folder[i]:
                        #     # check if the other image is having frontal in their name
                        #     if 'frontal' in os.listdir(study_path)[0]:
                        #         shutil.copy(view_path, os.path.join(new_path, new_folder[3], patient + '_' + study + '_' + view[j] + '.jpg'))

                except:
                    pass
        
        if count > 9999:
            break

# moving images from the order 2501 to 5000 of trainA folder to testA folder, and moving images from the order 2501 to 5000 of trainB folder to testB folder
trainA_path = os.path.join(new_path, new_folder[0])
trainB_path = os.path.join(new_path, new_folder[1])
testA_path = os.path.join(new_path, new_folder[2])
testB_path = os.path.join(new_path, new_folder[3])

# Make sure that the testA and testB folder exists
if not os.path.exists(testA_path):
    os.makedirs(testA_path, exist_ok=True)
if not os.path.exists(testB_path):
    os.makedirs(testB_path, exist_ok=True)
    
files = os.listdir(trainA_path)
for j in range(2500, 5000):
    shutil.move(os.path.join(trainA_path, files[j]), os.path.join(testA_path, files[j]))
    
files = os.listdir(trainB_path)
for j in range(2500, 5000):
    shutil.move(os.path.join(trainB_path, files[j]), os.path.join(testB_path, files[j]))
            

# moving images from the order 2001 to 2500 of trainA folder to validA folder, and moving images from the order 2001 to 2500 of trainB folder to validB folder
trainA_path = os.path.join(new_path, new_folder[0])
trainB_path = os.path.join(new_path, new_folder[1])
validA_path = os.path.join(new_path, "validA")
validB_path = os.path.join(new_path, "validB")

# Make sure that the validA and validB folder exists
if not os.path.exists(validA_path):
    os.makedirs(validA_path, exist_ok=True)
if not os.path.exists(validB_path):
    os.makedirs(validB_path, exist_ok=True)
files = os.listdir(trainA_path)
for j in range(2000, 2500):
    shutil.move(os.path.join(trainA_path, files[j]), os.path.join(validA_path, files[j]))
    
files = os.listdir(trainB_path)
for j in range(2000, 2500):
    shutil.move(os.path.join(trainB_path, files[j]), os.path.join(validB_path, files[j]))
    
