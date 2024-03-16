import os
import requests

def download_file(url, file_name, directory):
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
    ('https://s3.us-west-1.wasabisys.com/bb-weights/queratograySketch_v10.safetensors?AWSAccessKeyId=000R8RBVN2HP618VW0U4&Expires=1710558861&Signature=xYiwZVL%2FAT7rInb5%2F9Iq4qi2D6E%3D', 'queratograySketch_v10.safetensors'),
    ('https://s3.us-west-1.wasabisys.com/bb-weights/RealESRGAN_x2.pth?AWSAccessKeyId=000R8RBVN2HP618VW0U4&Expires=1710558893&Signature=PDVDoWN%2B2dSxsrjvJPpHmZP4gnM%3D', 'RealESRGAN_x2.pth'),
    ('https://s3.us-west-1.wasabisys.com/bb-weights/RealESRGAN_x2.pth.lock?AWSAccessKeyId=000R8RBVN2HP618VW0U4&Expires=1710558909&Signature=Z4jXfMqp%2FOLOXtCtZwX77JqWzMA%3D', 'RealESRGAN_x2.pth.lock')
]

print("Downloading\n")

for url, file_name in files:
    downloaded_file_path = download_file(url, file_name, 'weights')
    print(f"File downloaded and saved to: {downloaded_file_path}")