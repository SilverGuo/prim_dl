from PIL import Image
import sys
from graph import build_graph, segment_graph
from gaussian_filter import gaussian_grid, filter_image
from random import random
from numpy import sqrt

def diff_rgb(img, x1, y1, x2, y2):

    r = (img[0][x1, y1] - img[0][x2, y2]) ** 2
    g = (img[1][x1, y1] - img[1][x2, y2]) ** 2
    b = (img[2][x1, y1] - img[2][x2, y2]) ** 2
    return sqrt(r + g + b)

def diff_grey(img, x1, y1, x2, y2):
    v = (img[x1, y1] - img[x2, y2]) ** 2
    return sqrt(v)

def threshold(size, const):
    return (const / size)

def generate_image(forest, width, height):
    random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
    colors = [random_color() for i in xrange(width*height)]

    img = Image.new('RGB', (width, height))
    im = img.load()
    for y in xrange(height):
        for x in xrange(width):
            comp = forest.find(y * width + x) # find the initial parent vertice of the component
            im[x, y] = colors[comp]
    
    return img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)

def segmentation(image,output,k=float(300),min_size_in=int(20)):
    """
    what it does

    args:
        name(type):
        
    returns:
        name(type):
    """
    image_origin = Image.open(image)
    sigma = float(0.8)
    K = k
    min_size = min_size_in

    size = image_origin.size
    print 'Image info: ', image_origin.format, size, image_origin.mode
        
    grid = gaussian_grid(sigma)

    if image_origin.mode == 'RGB':
        image_origin.load()
        
        r, g, b = image_origin.split()
        # Using a Gaussian filter to smooth the image slightly before computing the
        # edge weights, in order to compensate for digitization artifects
        r = filter_image(r, grid)
        g = filter_image(g, grid)
        b = filter_image(b, grid)

        smooth = (r, g, b)
        diff = diff_rgb
    else:
        smooth = filter_image(image_origin, grid)
        diff = diff_grey

    graph = build_graph(smooth, size[1], size[0], diff, True)
    forest = segment_graph(graph, size[0]*size[1], K, min_size, threshold)
        
    image = generate_image(forest, size[1], size[0])
    image.save(output)
        
    print 'Number of components: %d' % forest.num_sets
    return forest

if __name__ == '__main__':    
   if len(sys.argv) != 3:
            print 'Invalid number of arguments passed.'
            print 'Correct usage: python segmentation.py input_file output_file'
   else: 
       segmentation(sys.argv[1],sys.argv[2])
       
