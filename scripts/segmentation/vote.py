import numpy as np

def vote(forest,feature_class):
    
    width = 240;    
    
    superpixel_vote = {k:[] for k in forest.superpixel}
    label_matrix = np.zeros(shape=(240,320))
    
    for k in forest.superpixel:
        for i in forest.superpixel[k]:
                x = i% width
                y = i/ width
                superpixel_vote[k].append(feature_class[x][y])
                label_matrix[x][y] = feature_class[x][y]                
                
    
    for l in superpixel_vote:
        (values,counts) = np.unique(superpixel_vote[l],return_counts=True)
        ind=np.argwhere(counts == np.amax(counts))
        #print values[ind].flatten().tolist(), len(values[ind])
        
        if len(values[ind]) == 1:
            superpixel_vote[l] = int(values[ind])
        else:
            superpixel_vote[l] = int(values[ind[0]])
    
    return superpixel_vote