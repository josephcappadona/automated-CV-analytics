
def get_filename(filepath):
    return filepath.split('/')[-1]

def get_parent_folder(filepath):
    return filepath.split('/')[-2]

def remove_extension(filename):
    return filename[:filename.rfind('.')]

