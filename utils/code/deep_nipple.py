from fastai.vision import *
import cv2
from pathlib import Path, PureWindowsPath
from PIL import Image, ImageFilter, ImageEnhance

from utils.code.aux_func import seg2bbox, predict
# from aux_func import seg2bbox, predict

def BlurNips(image_dir):
    # Blur all the nips found on any .png images in the image_dir
    # make sure image_dir doesnt end in \ 
    # make into path obj
    image_path = Path(image_dir)
    # break into compoents
    stem = image_path.stem
    parent = image_path.parent
    # assemble new destination
    new_dir = Path.joinpath(parent, '%s_blurred' % stem)
    try:
        new_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("%s Folder there" % new_dir)
    else:
        print("%s Folder created" % new_dir)    
    # get all png files in the source dir
    p = image_path.glob('*.png')
    files = [x for x in p if x.is_file()]

    # load pyrtoch model
    learner_path = 'utils/models/base-model/'
    learner = load_learner(learner_path)

    for f in files:
        # PIL images, not np
        orig = Image.open(f)
        blurred = orig.filter(ImageFilter.GaussianBlur(8))
        # get mask

        image, mask = predict(f, learner)
        # print(image.shape, mask.shape)
        # make a useful mask from this
        # keep just B and G, set R to 0
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        # note that this is float 32 values between 0 and 1
        # make it 8 bit image
        mask8 = mask * 255
        mask8 = mask8.astype(np.uint8)
        # print(mask8.dtype)
        # invert
        outmask = cv2.bitwise_not(mask8)
        
        # dialate to make a bit larger
        kernel = np.ones((12,12),np.uint8)
        outmask = cv2.dilate(outmask,kernel,iterations=1)
        # make into PIL image
        mask = Image.fromarray(outmask)
        # raise brightness
        # mask = ImageEnhance.Brightness(mask).enhance(2)
        # blur the mask
        mask_blur = mask.filter(ImageFilter.GaussianBlur(8))
        # do the composite
        result = Image.composite(blurred, orig, mask_blur)

        # plt.imshow(result)
        # plt.show()
        outfile = str(new_dir / f.name)
        print(outfile)
        result.save(outfile)



def DeepNipple(image_path, alg_mode, show=True):
    '''
    :param image_path: input image absolute path
    :param alg_mode: seg/bbox
    :param device: cpu or gpu number
    :return: segmentation mask / bounding boxes
    '''

    # load pyrtoch model
    learner_path = 'utils/models/base-model/'
    learner = load_learner(learner_path)

    image, mask = predict(image_path, learner)
    print(image.shape, mask.shape)

    # plt.imshow(mask[:, :, 1], alpha=0.6, interpolation='bilinear', cmap='magma')
    # plt.show()
    # plt.imshow(mask[:, :, 2], alpha=0.6, interpolation='bilinear', cmap='afmhot')
    # plt.show()

    if alg_mode == 'seg':
        output = mask

        plt.subplot(121)
        plt.imshow(image)
        plt.title('Original image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(image)
        plt.axis('off')
        plt.imshow(mask[:, :, 1], alpha=0.6, interpolation='bilinear', cmap='magma')
        plt.axis('off')
        plt.imshow(mask[:, :, 2], alpha=0.6, interpolation='bilinear', cmap='afmhot')
        plt.axis('off')
        plt.title('NAC segmentation')
        plt.show()

    else:
        coords = seg2bbox(mask)
        output = coords

        if show:
            for coor in coords:
                y1, y2, x1, x2 = coor[0], coor[1], coor[2], coor[3]
                cv2.rectangle(image, (x1, y1), (x2, y2), (36, 255, 12), 2, -1)

            plt.imshow(image)
            plt.show()

    return output