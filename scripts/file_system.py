import os
import shutil

__all__ = ['new_folder',
           'delete_all']

#==============================================================================
# new_folder
#==============================================================================
def new_folder(obj_path):
    if not os.path.exists(obj_path): 
        os.makedirs(obj_path)
    return

#==============================================================================
# delete_all
#==============================================================================
def delete_all(obj_path, subfolder=False):
    for file_name in os.listdir(obj_path):
        file_path = os.path.join(obj_path, file_name)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif subfolder == True and os.path.isdir(file_path):
            shutil.rmtree(file_path)
    return
