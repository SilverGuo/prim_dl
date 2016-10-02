from PIL import Image, ImageDraw, ImageFont
import sys
from segmentation import segmentation
from vote import vote
import numpy as np 
from enum import Enum
from labelseg import labelseg

def read_label_batch(label_file):
        label = []
        with open(label_file) as f:
            for line in f:
                label.append(map(int,line.split()))
        return np.array(label)
        
class Label(Enum):
    sky = 0
    tree = 1
    road = 2
    grass = 3
    water = 4
    building = 5
    mountain = 6
    object = 7

def labeling(feature,image):
        output_file = '/Users/xchen/Desktop/Segmentation/output/photo_o.jpg'
        output_label = '/Users/xchen/Desktop/Segmentation/output/photo_l.jpg'
        
        feature_class = read_label_batch(feature)
        
        forest = segmentation(image,output_file)
        
        forest_vote = vote(forest,feature_class)
        
        label_matrix = labelseg(forest,feature_class)        
        
        image_label = Image.open(image)
        draw = ImageDraw.Draw(image_label)
        drawfont = ImageFont.truetype('/Library/Fonts/Times New Roman.ttf',8)
        
        for i in forest.superpixel:
            y = i%240
            x = i/240           
            if forest_vote[i]>=0:
                draw.text((x,y),Label(forest_vote[i]).name,font = drawfont)
            else:
                draw.text((x,y),'unknown',font = drawfont)       
        
        del draw
        image_label.save(output_label)
       # print forest_vote;
        print label_matrix
        return label_matrix
        
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Invalid number of arguments passed.'
        print 'Correct usage: python feature_class image_origin'
    else:
        labeling(sys.argv[1],sys.argv[2])