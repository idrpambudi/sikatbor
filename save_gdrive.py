from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import glob
import threading
'''
- Put 'client_secrets.json' at input directory, received from Google Api Console.
- First time getting 'gdrive_redentials.txt' file require browser activity.
'''

GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = 'input/client_secrets.json'
PROJECT_FOLDER = 'bakofal'

def create_gdrive_folder(folder_name, parent_id=''):
    folder = drive.CreateFile({
        'title': folder_name,
        "parents": [{'id':parent_id}],
        "mimeType": "application/vnd.google-apps.folder"
    })
    folder.Upload()
    return folder

def upload_segmented(arr_dir, folder_id):
    for file_dir in arr_dir:
        file_name = file_dir.split('/')[-1]
        f = drive.CreateFile({
            "title": file_name,
            "parents": [{'id':folder_id}]
        })
        f.SetContentFile(file_dir)
        f.Upload()
        print(file_name)

def save_file_multithreaded(arr_dir, folder_id, thread_num=8):
    arr_len = len(arr_dir)//thread_num
    thread_arr = []
    current = 0
    for i in range(thread_num):
        if i == thread_num-1:
            arr = arr_dir[current:]
        else:
            arr = arr_dir[current:current+arr_len]
        current += arr_len
        thread = threading.Thread(target=upload_segmented, args=(arr, folder_id))
        thread.start()
        thread_arr.append(thread)
    for thread in thread_arr:
        thread.join()


gauth = GoogleAuth()
gauth.LoadCredentialsFile("input/gdrive_credentials.json")
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.CommandLineAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("input/gdrive_credentials.json")
drive = GoogleDrive(gauth)


## Check existing GDrive folder 'bakofal' at  root directory, create if not exist
file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
project_folder_exist = False
project_folder_id = ''
for f in file_list:
    if f['title'] == PROJECT_FOLDER:
        project_folder_exist = True
        project_folder_id = f['id']

if not project_folder_exist:
    f = drive.CreateFile({'title': PROJECT_FOLDER, 
        "mimeType": "application/vnd.google-apps.folder"})
    f.Upload()
    project_folder_id = f['id']

    
## Upload the output folder to bakofal folder gdrive
folder_name = input('Result folder name: ')

folder = create_gdrive_folder(folder_name, parent_id=project_folder_id)
folder_result = create_gdrive_folder('result', parent_id=folder['id'])
folder_result_no_post_proc = create_gdrive_folder('no_post_proc', parent_id=folder_result['id'])
folder_result_post_proc = create_gdrive_folder('post_proc', parent_id=folder_result['id'])

root_fdir = glob.glob('output/*.*')
result_fdir = glob.glob('output/result/*.*')
result_no_post_proc_fdir = glob.glob('output/result/no_post_proc/*.*')
result_post_proc_fdir = glob.glob('output/result/post_proc/*.*')

save_file_multithreaded(root_fdir, folder['id'])
save_file_multithreaded(result_fdir, folder_result['id'])
save_file_multithreaded(result_no_post_proc_fdir, folder_result_no_post_proc['id'])
save_file_multithreaded(result_post_proc_fdir, folder_result_post_proc['id'])
