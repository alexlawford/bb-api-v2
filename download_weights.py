import os
import requests

def download_file(url, directory):
    """
    Download a file from the given URL and save it into the specified directory.
    
    Parameters:
        url (str): The URL of the file to download.
        directory (str): The directory where the file will be saved.
        
    Returns:
        str: The path to the downloaded file.
    """

    # Make dir if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Get the file name from the URL
    file_name = url.split('/')[-1]
    
    # Path
    save_path = os.path.join(directory, file_name)
    
    # Download
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return save_path

# Download and save files
files = [
    'https://s3.us-west-1.wasabisys.com/bb-weights/queratograySketch_v10.safetensors?AWSAccessKeyId=DNT3E0UF8EO54ZBU6TSO&Expires=1710492556&Signature=UVe5KepdlLJhWAF%2BLJApjVvcpQ8%3D',
    'https://s3.us-west-1.wasabisys.com/bb-weights/RealESRGAN_x2.pth?AWSAccessKeyId=DNT3E0UF8EO54ZBU6TSO&Expires=1710492574&Signature=cjblw4uLrYkhUWmlp3NZjq5o9sg%3D',
    'https://s3.us-west-1.wasabisys.com/bb-weights/RealESRGAN_x2.pth.lock?AWSAccessKeyId=DNT3E0UF8EO54ZBU6TSO&Expires=1710492681&Signature=qKCVr6Iw2WGL25HUkfKHmfs8uJ4%3D'
]

print("Downloading\n")

for url in files:
    downloaded_file_path = download_file(url, 'weights')
    print(f"File downloaded and saved to: {downloaded_file_path}")