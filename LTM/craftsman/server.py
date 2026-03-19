import argparse
import base64
import os
import time
import random
import json
from datetime import datetime
import traceback
import trimesh
import requests
from typing import Optional, Union, List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .pipeline import CraftsManPipeline

if 'SAVE_TO_R2' in os.environ and os.environ['SAVE_TO_R2'] == 'True':
    assert 'BUCKET_NAME' in os.environ, "Please provide the BUCKET_NAME in the environment variables"
    assert 'AWS_ACCESS_KEY_ID' in os.environ, "Please provide the AWS_ACCESS_KEY_ID in the environment variables"
    assert 'AWS_SECRET_ACCESS_KEY' in os.environ, "Please provide the AWS_SECRET_ACCESS_KEY in the environment variables"
    assert 'ENDPOINT_URL' in os.environ, "Please provide the ENDPOINT_URL in the environment variables"

    print("Saving to R2 with the following configurations:")
    print(f"BUCKET_NAME:", os.environ['BUCKET_NAME'])
    print(f"AWS_ACCESS_KEY_ID:", os.environ['AWS_ACCESS_KEY_ID'])
    print(f"AWS_SECRET_ACCESS_KEY:", os.environ['AWS_SECRET_ACCESS_KEY'])
    print(f"ENDPOINT_URL:", os.environ['ENDPOINT_URL'])
    print(f"PUBLIC_URL:", os.environ['PUBLIC_URL'])
    
    import boto3
    BUCKET_NAME = os.environ['BUCKET_NAME']
    s3_client = boto3.client('s3',
                    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                    endpoint_url=os.environ['ENDPOINT_URL']
                )
    public_url = os.environ['PUBLIC_URL']
else:
    s3_client = None

def parse_parameters():
    parser = argparse.ArgumentParser("CraftsMan server using FastAPI")
    parser.add_argument(
        '--model', 
        required=True,
        type=str,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        '--data-dir',
        default="data",
        type=str,
        help="Path to the data directory for storing the input and output data"
    )
    parser.add_argument(
        '--torch_dtype',
        default="float32",
        type=str,
        help="Torch dtype for the model"
    )
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--host', default="0.0.0.0", type=str)
    parser.add_argument('--port', default=12345, type=int)
    return parser.parse_args()


class MeshGenRequest(BaseModel):
    task_uuid: str = None
    return_base64: Optional[bool] = False
    images: Optional[List[str]] = None  # List of base64-encoded input images
    images_url: Optional[List[str]] = None  # List of input image urls
    config: Optional[dict] = {
        "mc_resolution": 8,
        "sample_steps": 30,
        "cfg_scale": 7.5,
        "seed": 0,
    }

class Image(BaseModel):
    b64_str: str
    """
    The base64-encoded image.
    """

class Mesh(BaseModel):
    b64_str: str
    """
    The base64-encoded generated mesh.
    """

class CraftsManResponse(BaseModel):
    created: int
    """
    Unix timestamp of when the generation was started.
    """

    input_images: List[Image]
    """
    The list of input images.
    """

    preprocessed_input_images: List[Image]
    """
    The list of preprocessed images.
    """

    textureless_models: List[Mesh]
    """
    The list of generated meshes.
    """

    seed: Optional[int] = None
    """
    The seed used for the generation.
    """

class CraftsManR2Response(BaseModel):
    created: int
    """
    Unix timestamp of when the generation was started.
    """

    input_images_url: List[str]
    """
    The list of input images.
    """

    preprocessed_input_images_url: List[str]

    """
    The list of preprocessed images.
    """

    textureless_models_url: List[str]
    """
    The list of generated meshes.
    """

    seed: Optional[int] = None
    """
    The seed used for the generation.
    """

class CraftsManLocalResponse(BaseModel):
    created: int
    """
    Unix timestamp of when the generation was started.
    """

    input_images_path: List[str]
    """
    The list of input images.
    """

    preprocessed_input_images_path: List[str]

    """
    The list of preprocessed images.
    """

    textureless_models_path: List[str]
    """
    The list of generated meshes.
    """

    seed: Optional[int] = None
    """
    The seed used for the generation.
    """


def run_server():
    cfgs = parse_parameters()

    # prepare models
    pipeline = CraftsManPipeline.from_pretrained(
        pretrained_model_name_or_path=cfgs.model,
        device=cfgs.device,
        torch_dtype=cfgs.torch_dtype,
    )

    # # prepare fastapi app
    app = FastAPI()

    @app.post("/v1/meshes/generations", response_model=Union[CraftsManResponse, CraftsManR2Response, CraftsManLocalResponse])
    async def meshes_generations(request: MeshGenRequest):
        try:
            if request.task_uuid is None:
                task_uuid = datetime.now().strftime('%Y-%m-%d/%H') + '/' + f'{datetime.now().strftime("%M-%S-%f")}' + '_' + str(os.getpid())
            else:
                task_uuid = request.task_uuid

            # create task directory
            save_dir = os.path.join(cfgs.data_dir + '/cache/craftsman', task_uuid)
            os.makedirs(save_dir, exist_ok=True)

            # check if images or images_url is provided
            if request.images is not None and request.images_url is not None:
                return JSONResponse(content={"error": "Only one of 'images' or 'images_url' must be provided"}, status_code=405) # 405 invalid input
            elif request.images is not None and request.images_url is None:
                # check if multiple images are provided
                if len(request.images) > 1:
                    return JSONResponse(content={"error": "Only one image is supported"}, status_code=406) # 406 Not Acceptable yet
                else:
                    ## save input image
                    image_path = os.path.join(save_dir, 'input_image.png')
                    with open(image_path, 'wb') as f:
                        f.write(base64.b64decode(request.images[0]))
            elif request.images is None and request.images_url is not None:
                # check if multiple images are provided
                if len(request.images_url) > 1:
                    return JSONResponse(content={"error": "Only one image is supported"}, status_code=406) # 406 Not Acceptable yet
                else:
                    ## save input image
                    image_path = os.path.join(save_dir, 'input_image.png')
                    with open(image_path, 'wb') as f:
                        f.write(requests.get(request.images_url[0]).content)
            else:
                return JSONResponse(content={"error": "Either 'images' or 'images_url' must be provided"}, status_code=405) # 405 invalid input

            ## generate mesh
            start = time.time()
            seed = request.config.get("seed", random.randint(0, 10000000))
            out = pipeline(
                image_path,
                mc_depth=request.config.get("mc_resolution", 8),
                num_inference_steps=request.config.get("sample_steps", 30),
                guidance_scale=request.config.get("cfg_scale", 7.5),
                seed=seed,
                # prompt=request.prompt, # NOT used
                )
            print(f"Time: {time.time() - start}s")

            ## save output image, we only support one image
            preprocessed_image_path = os.path.join(save_dir, f'preprocessed_input_image.png')
            out.images[0].save(preprocessed_image_path)

            ## save output mesh
            meshes_path = []
            for i, mesh in enumerate(out.meshes):
                mesh_path = os.path.join(save_dir, f'output_mesh_{i}.glb')
                os.makedirs(os.path.dirname(os.path.abspath(mesh_path)), exist_ok=True)
                mesh.export(mesh_path)
                meshes_path.append(mesh_path)

            ## return
            if not request.return_base64: # directly return the local or R2 path
                if s3_client is not None:
                    date = time.strftime("%Y-%m-%d/%H")
                    start = time.time()
                    # upload input image and mesh to cloudflare R2 using aws s3 and boto3
                    input_images_url = []
                    for i, image in enumerate([image_path]): # only one image
                        s3_client.upload_file(image, BUCKET_NAME, f"tasks/{date}/{task_uuid}/input_image_{i}.png")
                        input_images_url.append(f"{public_url}/tasks/{date}/{task_uuid}/input_image_{i}.png")

                    preprocessed_images_url = []
                    for i, image in enumerate([preprocessed_image_path]):
                        s3_client.upload_file(image, BUCKET_NAME, f"tasks/{date}/{task_uuid}/preprocessed_input_image_{i}.png")
                        preprocessed_images_url.append(f"{public_url}/tasks/{date}/{task_uuid}/preprocessed_input_image_{i}.png")

                    textureless_models_url = []
                    for i, mesh in enumerate(meshes_path):
                        s3_client.upload_file(mesh, BUCKET_NAME, f"tasks/{date}/{task_uuid}/textureless_model_{i}.glb")
                        textureless_models_url.append(f"{public_url}/tasks/{date}/{task_uuid}/textureless_model_{i}.glb")
                    print(f"Textureless model uploaded to {textureless_models_url} and takes {time.time() - start} seconds.")

                    return CraftsManR2Response(
                        created=int(start),
                        input_images_url=input_images_url,
                        preprocessed_input_images_url=preprocessed_images_url,
                        textureless_models_url=textureless_models_url,
                        seed=seed
                    )
                else:
                    return CraftsManLocalResponse(
                        created=int(start),
                        input_images_path=[image_path],
                        preprocessed_input_images_path=[preprocessed_image_path],
                        textureless_models_path=meshes_path,
                        seed=seed
                    )

            else: # return base64 encoded results
                return CraftsManResponse(
                    created=int(start),
                    input_images=[Image(b64_str=base64.b64encode(open(image_path, 'rb').read()).decode('utf-8'))],
                    preprocessed_input_images=[Image(b64_str=base64.b64encode(open(preprocessed_image_path, 'rb').read()).decode('utf-8'))],
                    textureless_models=[Mesh(b64_str=base64.b64encode(open(mesh_path, 'rb').read()).decode('utf-8')) for mesh_path in meshes_path],
                    seed=seed
                )

        except Exception as e:
            traceback.print_exc()
            print(f"generate_model error: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
    
    @app.get("/v1/models")
    async def models():
        return {"models": [cfgs.model]}

    @app.get("/health")
    async def health():
        return {"status": "OK"}

    uvicorn.run(app, host=cfgs.host, port=cfgs.port)