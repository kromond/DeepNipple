'''
Take a folder of images, blur out the nips, write to a new folder
'''

import argparse

# from utils.code.deep_nipple import DeepNipple
from utils.code.deep_nipple import BlurNips

parser = argparse.ArgumentParser(description='Loop over images in dir, blur any nips found')
parser.add_argument('--image_dir', type=str, help='path to the input images')
# parser.add_argument('--mode', type=str, default='seg', help='seg or bbox mode')
# parser.add_argument('--show', type=bool, default=True, help='show the output')


if __name__ == '__main__':

    print('Running DeepNipple...')

    args = parser.parse_args()
    image_dir = args.image_dir


    output = BlurNips(image_dir)

