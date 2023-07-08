import zipfile


path_to_zip_file        = 'F:/_PROPOSAL2022/filesFromDave/CIS_thesis.zip'
directory_to_extract_to = 'F:/_PROPOSAL2022/filesFromDave/CIS_thesis/'

with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)
    
    
    
    
    
    