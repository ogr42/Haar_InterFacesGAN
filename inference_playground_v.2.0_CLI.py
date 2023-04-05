# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1stjkMUhcTj3xy8KYjqxvFM3NRVV7nkwf
# StyleGAN3 Inference Notebook
Source:
https://colab.research.google.com/github/yuval-alaluf/stylegan3-editing/blob/master/notebooks/inference_playground.ipynb
https://github.com/AnonSubm2021/TransStyleGAN


CLI code by installing modules BEFORE running script:

mkdir haar_interfacesgan
chdir haar_interfacesgan
git clone https://github.com/yuval-alaluf/stylegan3-editing
wget https://github.com/opencv/opencv/raw/master/data/haarcascades/'haarcascade_frontalface_default.xml
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 –force
pip install pyrallis
"""


import os
import os.path
import sys
import requests
import time
import copy
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
import torchvision.transforms as transforms

# clone repo
use_pydrive = False
CODE_DIR = './haar_interfacesgan'
gan_model_path = f'{CODE_DIR}/stylegan3-editing'
haar_file = 'haarcascade_frontalface_default.xml'
custom_path = f'{CODE_DIR}/data/custom'
uploads_path = f'{CODE_DIR}/uploads'
edited_faces_path = f'{CODE_DIR}/edited_faces'
edited_photo_path = f'{CODE_DIR}/edited_photos'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
experiment_types = ['restyle_e4e_ffhq', 'restyle_pSp_ffhq']

#Import Packages
os.chdir(gan_model_path)
sys.path.append(".")
sys.path.append("..")
from editing.interfacegan.face_editor import FaceEditor
from models.stylegan3.model import GeneratorType
from notebooks.notebook_utils import Downloader, ENCODER_PATHS, INTERFACEGAN_PATHS
from notebooks.notebook_utils import run_alignment, crop_image, compute_transforms
from utils.common import tensor2im
from utils.inference_utils import run_on_batch, load_encoder, get_average_image


def get_cli_args():
    """
    Parsing CLI arguments
    :return: tuple
    """
    parser = argparse.ArgumentParser(
        prog='Haar-InterfacesGAN model',
        description='Changing faces in customer\'s photos',
        epilog='Enter customer\'s photo, find end edit faces'
    )
    # Input user's photo URL
    parser.add_argument(
        '-u',
        dest='url_photo',
        default='https://www.upwork.com/profile-portraits/c1ei9qru6oT3K9chpqHcpiPgOYhnGXD6CsdWLy2J87ZzCKmmlbNK58G2TqM8YXK_2Y',
        type=str,
        help='Input URL or file path to user\'s photo'
    )
    # Select model type you wish to perform inference on.
    parser.add_argument(
        '-t',
        dest='experiment_type',
        choices=experiment_types,
        default=experiment_types[0],
        type=str,
        help=f'Select model type you wish to perform inference on: ' \
             f'{experiment_types[0]} or {experiment_types[1]}'
    )
    # Select edit direction
    edit_directions = ['age', 'smile', 'pose', 'Male']
    parser.add_argument(
        '-d',
        dest='edit_direction',
        choices=edit_directions,
        default=edit_directions[0],
        type=str,
        help=f'Select edit direction - "age", "smile", "pose", or "Male"'
    )
    # Parsing edit direction values.
    parser.add_argument(
        '-i',
        dest='min_value',
        choices=range(-10, 11),
        default=-10,
        type=int,
        help='Parsing minimum edit direction value (integer between -10 and 10).'
    )
    parser.add_argument(
        '-a',
        dest='max_value',
        choices=range(-10, 11),
        default=10,
        type=int,
        help='Parsing maximum edit direction value (integer between -10 and 10).'
    )  # option that takes a value
    args = parser.parse_args()
    return args


def download_models():
    """ Download ReStyle SG3 Encoder """
    models_path = f'{gan_model_path}/pretrained_models'
    downloader = Downloader(
        code_dir=gan_model_path,
        use_pydrive=use_pydrive,
        subdir="pretrained_models")
    for experiment_type in experiment_types:
        model_file = f'{models_path}/{experiment_type}.pt'
        if not os.path.exists(model_file) or os.path.getsize(model_file) < 1000000:
            print(f'Downloading ReStyle encoder model: {experiment_type}...')
            print(os.getcwd())
            try:
                downloader.download_file(file_id=ENCODER_PATHS[experiment_type]['id'],
                                      file_name=ENCODER_PATHS[experiment_type]['name'])
            except Exception as e:
                raise ValueError(f"Unable to download model correctly! {e}")
            # if google drive receives too many requests, we'll reach the quota limit and
            # be unable to download the model
            if os.path.getsize(model_file) < 1000000:
                raise ValueError("Pretrained model was unable to be downloaded correctly!")
            else:
                print('Done.')
        else:
            print(f'Model for {experiment_type} already exists!')


def get_file_name(image_url):
    """Getting file name from url or path to user photo"""
    if '\\' in image_url:
        file_name = image_url.rsplit('\\')[-1]
    elif  '/' in image_url:
        file_name = image_url.rsplit('/')[-1]
    else:
        print(f'Your URL or path is incorrectly, restart script again!')
        sys.exit()
    return file_name


def get_image(image_url, file_name):
    """
    Getting URL to user's photo, uploading and saving an image
    """
    for i in range(10):
        try:
            response = requests.get(url=image_url, stream=True)
            if response.status_code == 200:
                break
        except Exception as ex:
            print('Download error: ', ex, '\nSleeping 3 sec...')
            time.sleep(3)
    else:
        print('Your URL doesn\'t open, enter it again!')
        sys.exit()
    os.chdir(CODE_DIR)
    os.makedirs(custom_path, exist_ok=True)
    os.makedirs(uploads_path, exist_ok=True)
    for path in [custom_path, uploads_path]:
        with open(f'{path}/{file_name}', 'wb') as fd:
            for chunk in response.iter_content(chunk_size=128):
                fd.write(chunk)
    image = Image.open(BytesIO(response.content))
    image.show()
    return image


def detect_faces(file_name):
    """
    Detecting face(s) in the user's photo using Haar model
    """
    file_path = f'{uploads_path}/{file_name}'
    res = file_name.rsplit('.', 1)[0]
    img = cv2.imread(file_path)
    img_height, img_width, _ = img.shape
    print(f'IMAGE SIZES: height {img_height}, width {img_width}')
    face_cascade = cv2.CascadeClassifier(haar_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        # You can change the parameter 'scaleFactor' between 1.01 and 1.2 !!! It needs research...
        scaleFactor=1.1,
        minNeighbors=5
    )
    for i, (x, y, w, h) in enumerate(faces):
        print(f'FACE SIZES BY HAAR MODEL: start x {x}, start y: {y}, width {w}, height {h}')
        # x moves left, y moves up
        delta_x, delta_y = int(w * 0.2), int(w * 0.4)
        # w moves right, h moves down, delta w (or h) = delta x (or y) plus itself delta
        delta_w, delta_h = int(w * 0.6), int(w * 0.6)
        x = x - delta_x if delta_x < x else 0
        y = y - delta_y if delta_y < y else 0
        w = w + delta_w if x + w + delta_w < img_width else img_width - x
        h = h + delta_h if y + h + delta_h < img_height else img_height - y
        face_coordinates = [x, y, x + w, y + h]
        print(f'CHANGED FACE SIZES: start x {x}, start y: {y}, width {w}, height {h}')
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img.show()
        saving = input(f'Is this face detected correctly? y/n: ')
        if saving.lower() == 'y':
            roi_color = img[y+2:y + h - 4, x+2:x + w-4]
            # Saving face image to file image_name
            face_file = f'{custom_path}/face_{res}.jpg'
            cv2.imwrite(face_file, roi_color)
            face_img = Image.open(face_file)
            face_img.show()
            print(f'This face saved!')
            break
        else:
            continue
    cv2.waitKey()
    cv2.destroyAllWindows()
    return face_file, face_coordinates


def transform_face(face_file, experiment_type, edit_direction, min_value, max_value):
    """
    Prepare Environment and Download InterfacesGAN Code. Face transforming from user photo.
    """
    EXPERIMENT_DATA_ARGS = {
        "restyle_pSp_ffhq": {
            "model_path": f'{gan_model_path}/pretrained_models/restyle_pSp_ffhq.pt',
            "image_path": face_file,
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        },
        "restyle_e4e_ffhq": {
            "model_path": f'{gan_model_path}/pretrained_models/restyle_e4e_ffhq.pt',
            "image_path": face_file,
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
    }
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
    # Load ReStyle SG3 Encoder
    model_path = EXPERIMENT_ARGS['model_path']
    net, opts = load_encoder(checkpoint_path=model_path)
    # pprint.pprint(dataclasses.asdict(opts))

    # Define and Visualize Input
    image_path = Path(EXPERIMENT_DATA_ARGS[experiment_type]["image_path"])
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize((256, 256))
    original_image

    # Perform Inversion
    # Now we'll run inference. By default, we'll run using 3 inference steps.
    n_iters_per_batch = 3 # You can change the parameter in the cell below !!!
    opts.n_iters_per_batch = n_iters_per_batch
    opts.resize_outputs = False  # generate outputs at full resolution

    # Get Aligned and Cropped Input Images
    input_image = run_alignment(image_path)
    cropped_image = crop_image(image_path)
    joined = np.concatenate([input_image.resize((256, 256)), cropped_image.resize((256, 256))], axis=1)
    Image.fromarray(joined)

    # Compute Landmarks-Based Transforms
    images_dir = Path("./images")
    images_dir.mkdir(exist_ok=True, parents=True)
    cropped_path = images_dir / f"cropped_{image_path.name}"
    aligned_path = images_dir / f"aligned_{image_path.name}"
    cropped_image.save(cropped_path)
    input_image.save(aligned_path)
    landmarks_transform = compute_transforms(
        aligned_path=aligned_path,
        cropped_path=cropped_path
    )

    # Run Inference
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)
    avg_image = get_average_image(net)
    with torch.no_grad():
        tic = time.time()
        result_batch, result_latents = run_on_batch(
            inputs=transformed_image.unsqueeze(0).cuda().float(),
            net=net,
            opts=opts,
            avg_image=avg_image,
            landmarks_transform=torch.from_numpy(landmarks_transform).cuda().float()
            )
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    # Visualize Result
    result_tensors = result_batch[0]  # there's one image in our batch
    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
    final_rec = tensor2im(result_tensors[-1]).resize(resize_amount)
    input_im = cropped_image.resize(resize_amount)
    res = np.concatenate([np.array(input_im), np.array(final_rec)], axis=1)
    res = Image.fromarray(res)
    res.resize((1024, 512))

    # Save Result
    outputs_path = "./outputs"
    os.makedirs(outputs_path, exist_ok=True)
    res.save(os.path.join(outputs_path, os.path.basename(image_path)))

    # download files for interfacegan
    downloader = Downloader(
        code_dir=CODE_DIR,
        use_pydrive=use_pydrive,
        subdir="editing/interfacegan/boundaries/ffhq")
    print("Downloading InterFaceGAN boundaries...")
    for editing_file, params in INTERFACEGAN_PATHS.items():
        print(f"Downloading {editing_file} boundary...")
        downloader.download_file(
            file_id=params['id'],
            file_name=params['name'])

    # Select edit direction and values
    editor = FaceEditor(stylegan_generator=net.decoder, generator_type=GeneratorType.ALIGNED)

    # Perform Edit
    print(f"Performing edit for {edit_direction}...")
    input_latent = torch.from_numpy(result_latents[0][-1]).unsqueeze(0).cuda()
    edit_images, edit_latents = editor.edit(
        latents=input_latent,
        direction=edit_direction,
        factor_range=(min_value, max_value),
        user_transforms=landmarks_transform,
        apply_user_transformations=True
    )
    print("Done!")

    # Show Result
    if type(edit_images[0]) == list:
        edit_images = [image[0] for image in edit_images]
    res_array = np.array(edit_images[0].resize((512, 512)))
    for image in edit_images[1:]:
        res_array = np.concatenate([res_array, image.resize((512, 512))], axis=1)
    res_image = Image.fromarray(res_array).convert("RGB")
    res_image.show()

    # Save edited faces
    edited_faces_file = os.path.basename(image_path)
    os.makedirs(edited_faces_path, exist_ok=True)
    res_image.save(os.path.join(edited_faces_path, edited_faces_file))
    return edited_faces_file


def edit_users_photo(file_name, coord, edited_faces_file):
    """
    Editing user's photo with edited faces.
    """
    # face_coordinates = [x, y, x + w, y + h]
    # print(f'file_mame {file_name}, coord {coord}, edited_faces_file {edited_faces_file}')
    x, y, w, h = coord[0], coord[1], coord[2] - coord[0], coord[3] - coord[1]
    edited_photos = []
    custom_photo_path = f'{custom_path}/{file_name}'
    print(f'custom_photo_path {custom_photo_path}')
    custom_arr = cv2.imread(custom_photo_path) # np.array object
    cv2.imshow(custom_arr)
    # edited_faces_path = f'{CODE_DIR}/edited_faces'
    edited_faces_arr = cv2.imread(f'{edited_faces_path}/{edited_faces_file}')
    ed_photos_height, ed_photos_width, _ =  edited_faces_arr.shape
    ed_photos_number = int(ed_photos_width / ed_photos_height)
    print(f'There are {ed_photos_number} edited face images!')
    for n in range(ed_photos_number):
        currently_face = edited_faces_arr[0:512, n*512:(n+1)*512]
        resized_curr_face = cv2.resize(currently_face, (h-10,w-10), interpolation=cv2.INTER_AREA)
        edited_img_arr = copy.deepcopy(custom_arr)
        # You can change this values +/- 5-20 for face move !!! It needs research...
        edited_img_arr[y+5:y+w-5, x+5:x+h-5] = resized_curr_face[:]
        cv2.imwrite(f'{edited_photo_path}/{n}_{file_name}', edited_img_arr)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return edited_photos


def main():
    """Main function"""
    download_models()
    image_url, experiment_type, edit_direction, min_value, max_value = get_cli_args()
    file_name = get_file_name(image_url)
    get_image(image_url, file_name)
    face_file, face_coords = detect_faces(file_name)
    edited_faces_file = transform_face(face_file, experiment_type, edit_direction, min_value, max_value)
    edit_users_photo(file_name, face_coords, edited_faces_file)


if __name__ == "__main__":
  main()
