import cv2
import torch
from PIL import Image
import argparse, os 
from imutils import paths



def load_image(image_path, x32=False):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))

    img = torch.from_numpy(img)
    img = img/127.5 - 1.0
    return img


def convert_main(args, style = "paprika"):
    # load model 
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained= style)
    face2paint = torch.hub.load(
        'bryandlee/animegan2-pytorch:main', 'face2paint', 
        size=512, device="cpu"
    )

    # get list of images
    for imagePath in paths.list_images(args.output_dir):
        if imagePath.find("output") == -1:
            img = Image.open(imagePath).convert("RGB")
            out = face2paint(model, img)
            out.save(os.path.splitext(imagePath)[0] + '{}_output.png'.format(style))


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./pytorch_generator_Paprika.pt',
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='./samples/inputs',
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./imgs',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--x32',
        action="store_true",
    )
    args = parser.parse_args()
    print(args)
    
    convert_main(args)