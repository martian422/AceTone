from PIL import Image
from io import BytesIO
import pickle
import traceback

from flask import Flask, request, Blueprint

root = Blueprint("root", __name__)

import torch
from transformers import AutoModelForCausalLM

# for deqa transformers==4.36.1
# how to run me?
# bash deqa_server.sh

def load_deqascore():
    model = AutoModelForCausalLM.from_pretrained(
        "outputs/pretrained/deqa",
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map=None,
    ).cuda()
    model.requires_grad_(False)
    
    @torch.no_grad()
    def compute_deqascore(images):
        score = model.score(images)
        score = score / 5
        score = [sc.item() for sc in score]
        return score

    return compute_deqascore


def create_app():
    global INFERENCE_FN
    INFERENCE_FN = load_deqascore()

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"]) 
def inference():
    print(f"received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        data = pickle.loads(data)

        images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]

        print(f"Got {len(images)} images")

        outputs = INFERENCE_FN(images)

        response = {"scores": outputs}
        response = pickle.dumps(response)

        returncode = 200
    except Exception as e:
        response = traceback.format_exc()
        print(response)
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode


HOST = "127.0.0.1"
PORT = 8085

if __name__ == "__main__":
    create_app().run(HOST, PORT)