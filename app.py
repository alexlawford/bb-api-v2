# In-built
import time
import secrets
from io import BytesIO
from hashlib import sha256

# Packages
import requests
from flask import Flask, request
from flask_restful import Resource, Api
from flask_httpauth import HTTPTokenAuth
from PIL import Image

# Local
from generate_images import generate

def saveBytescale (data):
    headers = {
        'Authorization': 'Bearer public_12a1yrrGGApHW4eVGAfq3RnXk9uv',
        'Content-Type': 'image/png',
    }
    return requests.post('https://api.bytescale.com/v2/accounts/12a1yrr/uploads/binary', headers=headers, data=data)

app = Flask(__name__)
api = Api(app)
auth = HTTPTokenAuth(scheme='Bearer')

@auth.verify_token
def verify_token(token):
    hash = sha256(str.encode(token)).digest()
    if secrets.compare_digest(hash, b'Z\x1f\x0f\xdeP\x99\x03\x16\x0b\xd6\x9a\x04\xfdQ;\xdb\x0e\xb0\x9a3;\xfc%\x15\xe7\xd4\x88t\xc9\xed\x81s'):
        return True
    else:
        return False
    
class Predict(Resource):
    @auth.login_required
    def post(self):
        time_start = time.time()

        req = request.json
        layers=req.get("layers")
        variation=req.get("variation")

        image = generate(layers, variation)

        with BytesIO() as image_binary:
            image.save(image_binary, "png")
            image_binary.seek(0)
            result = saveBytescale(image_binary)

        # Show total time
        time_end = time.time()
        print("Total time:", time_end - time_start, "s")

        return result.json()
    
    def get(self):
        return "GET REQ: hello world"

api.add_resource(Predict, "/")

@auth.error_handler
def auth_error():
    return "Access Denied", 403