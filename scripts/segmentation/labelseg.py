import numpy as np 

def labelseg(forest,feature_class):
    
    width = 240;    
    
    label_matrix = np.zeros(shape=(240,320))
    
    for k in forest.superpixel:
        for i in forest.superpixel[k]:
                x = i% width
                y = i/ width
                label_matrix[x][y] = feature_class[x][y]                
                
    print label_matrix.shape
    return label_matrix