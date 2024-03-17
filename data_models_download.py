"""
Description: This script downloads files from Google Drive and organizes them into specified directories.
"""

import os
import platform


def download_from_google_drive(folder_id, output_folder):
    """
    Download files from a Google Drive folder and save them to the specified output folder.

    Args:
        folder_id (str): The ID of the Google Drive folder.
        output_folder (str): The name of the output folder to save the downloaded files.

    Returns:
        None
    """
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, output_folder)
    output_dir = output_dir if platform.system() != "Windows" else current_dir.replace("\\", "/")
    command = f"gdown --folder https://drive.google.com/drive/folders/{folder_id} -O {output_dir}"
    os.system(command)

def create_directories():
    """
    Create 'data' and 'models' directories if they do not exist.

    Returns:
        None
    """
    directories = ['data', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"A directory has been created: {directory}")
        else:
            print(f"The {directory} directory already exists.")

def data_download_from_google_drive():
    """
    Download data files from Google Drive and save them to the 'data' directory.

    Returns:
        None
    """
    data_folder_id = '1hm-d0Hd3dzbGtBN0ZCu19n0O7Ol7NRMl?usp=drive_link'
    data_output_folder = "data"
    download_from_google_drive(data_folder_id, data_output_folder)

def models_download_from_google_drive():
    """
    Download model files from Google Drive and save them to the 'models' directory.

    Returns:
        None
    """
    models_folder_id = '1la9BvTtTpVhbSkX0kEYX4FQTfn0XLSXJ?usp=drive_link'
    models_output_folder = "models"
    download_from_google_drive(models_folder_id, models_output_folder)


if __name__ == "__main__":
    create_directories()
    data_download_from_google_drive()
    models_download_from_google_drive()
