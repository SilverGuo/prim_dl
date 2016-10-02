class Node:
    def __init__(self, parent, rank=0, size=1):
        self.id = parent        
        self.parent = parent
        self.rank = rank
        self.size = size   # every node has a initial size 1
    
    def __repr__(self):
        return '(id=%s, parent=%s, rank=%s, size=%s)' % (self.id,self.parent, self.rank, self.size)

class Forest:
    def __init__(self, num_nodes):
        self.nodes = [ Node(i) for i in xrange(num_nodes)]
        self.num_sets = num_nodes
        self.superpixel = {i:[i] for i in xrange(num_nodes)}
    
    def size_of(self, i):
        return self.nodes[i].size

    # try to find the initial node of this component
    def find(self, n):
        temp = n
        while temp != self.nodes[temp].parent:
            temp = self.nodes[temp].parent
       
        self.nodes[n].parent = temp
        return temp
    
    def merge(self, a, b):
        # the bigger rank stands for a bigger component
        # we use the name of the parent to represent the component 
        # The size of a component after a merge is simply the sum of the sizes of the two components
        if self.nodes[a].rank > self.nodes[b].rank:
            self.nodes[b].parent = a
            #print self.superpixel[b]
            self.superpixel[a].extend(self.superpixel[b])            
            #print self.superpixel[a]            
            del self.superpixel[b]
            #print self.superpixel.get(a).append(b)
            self.nodes[a].size = self.nodes[a].size + self.nodes[b].size
        else:
            self.nodes[a].parent = b
            self.superpixel[b].extend(self.superpixel[a])
            del self.superpixel[a]
            self.nodes[b].size = self.nodes[b].size + self.nodes[a].size
            
            if self.nodes[a].rank == self.nodes[b].rank:
                self.nodes[b].rank = self.nodes[b].rank + 1
        self.num_sets = self.num_sets - 1

def print_nodes(self):
    for node in self.nodes:
        print node

def create_edge(img, width, x, y, x1, y1, diff):
    vertex_id = lambda x, y: y * width + x
    w = diff(img, x, y, x1, y1)
    return (vertex_id(x, y), vertex_id(x1, y1), w)

def build_graph(img, width, height, diff, neighborhood_8=True):
    graph = []

    for y in xrange(height):
        for x in xrange(width):
            if x > 0:
                graph.append(create_edge(img, width, x, y, x-1, y, diff))

            if y > 0:
                graph.append(create_edge(img, width, x, y, x, y-1, diff))

            if neighborhood_8:
                if x > 0 and y > 0:
                    graph.append(create_edge(img, width, x, y, x-1, y-1, diff))

                if x > 0 and y < height-1:
                    graph.append(create_edge(img, width, x, y, x-1, y+1, diff))

    return graph
    
def remove_small_components(forest, graph, min_size):
    for edge in graph:
        a = forest.find(edge[0])
        b = forest.find(edge[1])
        
        if a != b and (forest.size_of(a) < min_size or forest.size_of(b) < min_size):
            forest.merge(a, b)
        
    
    return  forest

def segment_graph(graph, num_nodes, const, min_size, threshold_func):
    weight = lambda edge : edge[2]    
    forest = Forest(num_nodes)
    sorted_graph = sorted(graph, key = weight)
    # For each node, creating a corresponding threshold
    threshold = [threshold_func(1, const)] * num_nodes
    
    for edge in sorted_graph:
        parent_a = forest.find(edge[0])
        parent_b = forest.find(edge[1])
        # comparison between the internal difference and the edge weight
        a_condition = weight(edge) <= threshold[parent_a]
        b_condition = weight(edge) <= threshold[parent_b]
        
        if parent_a != parent_b and a_condition and b_condition:
            forest.merge(parent_a, parent_b)
            # change the threshold when two components merge result in larger component
            a = forest.find(parent_a)
            threshold[a] = weight(edge) + threshold_func(forest.nodes[a].size, const)
    print 'Number of components before removing the small components: %d' % forest.num_sets 

    return remove_small_components(forest, sorted_graph, min_size)   
    #return generate_superpixel(forest_s)