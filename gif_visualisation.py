import sys
from PIL import Image
import imageio
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import cv2

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    l.sort(key=alphanum_key)

def from_path_load_all_images(path):
    files = list()
    for file in os.listdir(path):
        files.append(os.path.join(path, file))
    return files

def concat_images(imga, imgb):
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    print(ha, wa)
    print(hb, wb)
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def concat_n_images(image_path_list):
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output

def make_gif_from_path(path):
    list_of_files = from_path_load_all_images("manifolds/blobs/Base/")
    sort_nicely(list_of_files)
    image_list = []
    for filename in list_of_files:
        im = imageio.imread(filename)
        image_list.append(im)
    #imageio.mimsave(path+"/"+"images.gif",image_list,duration=0.2, loop=1)
    imageio.mimsave("images.gif",image_list,duration=0.2, loop=1)

def make_gif_from_list_of_path(list_of_path, name):
    list_of_list_of_files = list()
    for path in list_of_path:
        list_of_files = from_path_load_all_images(path)
        sort_nicely(list_of_files)
        list_of_list_of_files.append(list_of_files)
    final_list = list(zip(*list_of_list_of_files))
    image_list = []
    for elem in final_list:
        create_concatenated_images(elem)
        im = imageio.imread('image.png')
        image_list.append(im)
    imageio.mimsave(name+".gif", image_list, duration=0.35, loop=1)

def create_concatenated_images(list_of_files):
    images = map(Image.open, list_of_files)
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (int(total_width/2), 2*max_height))
    x_offset, y_offset = 0, 0
    #First Im
    image = Image.open(list_of_files[0])
    new_im.paste(image, (x_offset, y_offset))
    x_offset += image.size[0]
    #First Im
    image = Image.open(list_of_files[1])
    new_im.paste(image, (x_offset, y_offset))
    x_offset = 0
    y_offset += image.size[1]
    #First Im
    image = Image.open(list_of_files[2])
    new_im.paste(image, (x_offset, y_offset))
    x_offset += image.size[0]
    #First Im
    image = Image.open(list_of_files[3])
    new_im.paste(image, (x_offset, y_offset))
    #new_im.save('image.pdf')
    new_im.save('image.pdf')

def create_concatenated_images_2(list_of_files):
    images = map(Image.open, list_of_files)
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (int(total_width), max_height))
    x_offset, y_offset = 0, 0
    #First Im
    image = Image.open(list_of_files[0])
    new_im.paste(image, (x_offset, y_offset))
    x_offset += image.size[0]
    #First Im
    image = Image.open(list_of_files[1])
    new_im.paste(image, (x_offset, y_offset))
    x_offset = 0
    y_offset += image.size[1]
    
    new_im.save('image.pdf')

if __name__ == "__main__":
    
    list_of_files = from_path_load_all_images("blobs1_pics/")
    sort_nicely(list_of_files)
    print(list_of_files)
    create_concatenated_images_2(list_of_files)
    
    #print(list_of_files)
    #images = map(Image.open, list_of_files)
    #widths, heights = zip(*(i.size for i in images))
    #total_width = sum(widths)
    #max_height = max(heights)
    #new_im = Image.new('RGB', (int(total_width), max_height))
    #x_offset, y_offset = 0, 0
    ##First Im
    #image = Image.open(list_of_files[0])
    #new_im.paste(image, (x_offset, y_offset))
    #x_offset += image.size[0]+5
    ##First Im
    #image = Image.open(list_of_files[1])
    #new_im.paste(image, (x_offset, y_offset))
    #x_offset = 0
    #y_offset += image.size[1]
    #new_im.save('image.pdf')


    #list_of_path = ["manifolds/W2V/blobs1/BASE/DISC/Normal/", "manifolds/W2V/blobs1/SELF/DISC/Normal/", "manifolds/W2V/blobs1/GANS_ADV_IS/GEN/Normal/", "manifolds/W2V/blobs1/GANS_ADV_IS/DISC/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "blobs1_AIS")
    #list_of_path = ["manifolds/W2V/blobs1/BASE/DISC/Negatives/", "manifolds/W2V/blobs1/SELF/DISC/Negatives/", "manifolds/W2V/blobs1/BASE/DISC/Negatives/", "manifolds/W2V/blobs1/SELF/DISC/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "blobs1_neg")
#
    #list_of_path = ["manifolds/W2V/blobs2/BASE/DISC/Normal/", "manifolds/W2V/blobs2/SELF/DISC/Normal/", "manifolds/W2V/blobs2/GANS_ADV_IS/GEN/Normal/", "manifolds/W2V/blobs2/GANS_ADV_IS/DISC/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "blobs2")
    #list_of_path = ["manifolds/W2V/blobs2/BASE/DISC/Negatives/", "manifolds/W2V/blobs2/SELF/DISC/Negatives/", "manifolds/W2V/blobs2/BASE/DISC/Negatives/", "manifolds/W2V/blobs2/SELF/DISC/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "blobs2_neg")

    #list_of_path = ["manifolds/s_curve/GANS_ADV_IS/GEN/Normal/", "manifolds/s_curve/GANS_ADV_IS/DISC/Normal/", "manifolds/s_curve/MALIGANS/GEN/Normal/", "manifolds/s_curve/MALIGANS/DISC/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "s_curve")
    #list_of_path = ["manifolds/s_curve/BASE/DISC/Negatives/", "manifolds/s_curve/SELF/DISC/Negatives/", "manifolds/s_curve/GANS_ADV_IS/GEN/Negatives/", "manifolds/s_curve/GANS_ADV_IS/DISC/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "s_curve_neg")


    #list_of_path = ["manifolds/swiss_roll/BASE/DISC/Normal/", "manifolds/swiss_roll/SELF/DISC/Normal/", "manifolds/swiss_roll/GANS_ADV_IS/GEN/Normal/", "manifolds/swiss_roll/GANS_ADV_IS/DISC/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "swiss_roll")
    #list_of_path = ["manifolds/swiss_roll/BASE/DISC/Negatives/", "manifolds/swiss_roll/SELF/DISC/Negatives/", "manifolds/swiss_roll/GANS_ADV_IS/GEN/Negatives/", "manifolds/swiss_roll/GANS_ADV_IS/DISC/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "swiss_roll_neg_AIS")

    #list_of_path = ["manifolds/s_curve/BASE/DISC/Normal/", "manifolds/s_curve/SELF/DISC/Normal/", "manifolds/s_curve/GANS_ADV_IS/GEN/Normal/", "manifolds/s_curve/GANS_ADV_IS/DISC/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "s_curve_AIS")
    #list_of_path = ["manifolds/s_curve/BASE/DISC/Negatives/", "manifolds/s_curve/SELF/DISC/Negatives/", "manifolds/s_curve/GANS_ADV_IS/GEN/Negatives/", "manifolds/s_curve/GANS_ADV_IS/DISC/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "s_curve_neg_AIS")

    #list_of_path = ["manifolds/blobs/BASE/DISC/Normal/", "manifolds/blobs/SELF/DISC/Normal/", "manifolds/blobs/GANS_ADV_IS/GEN/Normal/", "manifolds/blobs/GANS_ADV_IS/DISC/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "blobs_AIS")
    #list_of_path = ["manifolds/blobs/BASE/DISC/Negatives/", "manifolds/blobs/SELF/DISC/Negatives/", "manifolds/blobs/GANS_ADV_IS/GEN/Negatives/", "manifolds/blobs/GANS_ADV_IS/DISC/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "blobs_neg_AIS")

    #list_of_path = ["manifolds/moons/BASE/DISC/Normal/", "manifolds/moons/SELF/DISC/Normal/", "manifolds/moons/GANS_ADV_IS/GEN/Normal/", "manifolds/moons/GANS_ADV_IS/DISC/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "moons_AIS")
    #list_of_path = ["manifolds/moons/BASE/DISC/Negatives/", "manifolds/moons/SELF/DISC/Negatives/", "manifolds/moons/GANS_ADV_IS/GEN/Negatives/", "manifolds/moons/GANS_ADV_IS/DISC/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "moons_neg_AIS")


    #list_of_path = ["manifolds/100_blobs/Base/Negatives/", "manifolds/100_blobs/Self/Negatives/", "manifolds/100_blobs/AIS/Negatives/", "manifolds/100_blobs/MALIGAN/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "blobs")
    #list_of_path = ["manifolds/swiss_roll/Base/Negatives/", "manifolds/swiss_roll/Self/Negatives/", "manifolds/swiss_roll/AIS/Negatives/", "manifolds/swiss_roll/MALIGAN/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "blobs")
    #list_of_path = ["manifolds/blobs/Base/Negatives/", "manifolds/blobs/Self/Negatives/", "manifolds/blobs/AIS/Gen/Negatives/", "manifolds/blobs/MALIGAN/Gen/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "blobs")
    #list_of_path = ["manifolds/s_curve/Base/Negatives/", "manifolds/s_curve/Self/Negatives/", "manifolds/s_curve/AIS/Gen/Negatives/", "manifolds/s_curve/MALIGAN/Gen/Negatives/"]
    #make_gif_from_list_of_path(list_of_path, "s_curve")
    #list_of_path = ["manifolds/moons/Base/Normal/", "manifolds/moons/Self/Normal/", "manifolds/moons/AIS/Gen/Normal/", "manifolds/moons/MALIGAN/Gen/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "moons")

    #list_of_path = ["manifolds/moons/AIS/Gen/Normal/", "manifolds/moons/AIS/Disc/Normal/", "manifolds/moons/MALIGAN/Gen/Normal/", "manifolds/moons/MALIGAN/Disc/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "moons")
    #list_of_path = ["manifolds/100_blobs/AIS/Gen/Normal/", "manifolds/100_blobs/AIS/Disc/Normal/", "manifolds/100_blobs/MALIGAN/Gen/Normal/", "manifolds/100_blobs/MALIGAN/Disc/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "100_blobs")
    #list_of_path = ["manifolds/swiss_roll/AIS/Gen/Normal/", "manifolds/swiss_roll/AIS/Disc/Normal/", "manifolds/swiss_roll/MALIGAN/Gen/Normal/", "manifolds/swiss_roll/MALIGAN/Disc/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "swiss_roll")
    #list_of_path = ["manifolds/s_curve/AIS/Gen/Normal/", "manifolds/s_curve/AIS/Disc/Normal/", "manifolds/s_curve/MALIGAN/Gen/Normal/", "manifolds/s_curve/MALIGAN/Disc/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "s_curve")
    #list_of_path = ["manifolds/blobs/AIS/Gen/Normal/", "manifolds/blobs/AIS/Disc/Normal/", "manifolds/blobs/MALIGAN/Gen/Normal/", "manifolds/blobs/MALIGAN/Disc/Normal/"]
    #make_gif_from_list_of_path(list_of_path, "blobs")
