import torch
from torch import nn
import torch
torch.manual_seed(125)
import random
random.seed(125)
import numpy as np
import torch_f as torch_f

use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

def traverse(root, tree):
       
        if root is not None:
            traverse(root.left, tree)
            tree.append((root.radius, root.data))
            traverse(root.right, tree)
            return tree

def count_fn(f):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return f(*args, **kwargs)
    wrapper.count = 0
    return wrapper

@count_fn
def createNode(data, radius, left = None, right = None, ):
        """
        Utility function to create a node.
        """
        return Node(data, radius, left, right)

def deserialize(data):
    if  not data:
        return 
    nodes = data.split(';')  
    def post_order(nodes):
        if nodes[-1] == '#':
            nodes.pop()
            return None
        node = nodes.pop().split('_')
        data = int(node[0])
        radius = node[1]
        rad = radius.split(",")
        rad [0] = rad[0].replace('[','')
        rad [3] = rad[3].replace(']','')
        r = []
        for value in rad:
            r.append(float(value))
        r = torch.tensor(r, device=device)
        root = createNode(data, r)
        root.right = post_order(nodes)
        root.left = post_order(nodes)
        
        return root    
    return post_order(nodes)    


def read_tree(filename, dir):
    with open('./' +dir +'/' +filename, "r") as f:
        byte = f.read() 
        return byte

def numerarNodos(root, count):
    if root is not None:
        numerarNodos(root.left, count)
        root.data = len(count)
        count.append(1)
        numerarNodos(root.right, count)
        return 


def traverseFeatures(root, features):
       
    if root is not None:
        traverseFeatures(root.left, features)
        features.append(root.radius)
        traverseFeatures(root.right, features)
        return features


def searchNode(node, key):
     
    if (node == None):
        return False
    if (node.data == key):
        return node
        
    """ then recur on left subtree """
    res1 = searchNode(node.left, key)
    # node found, no need to look further
    if res1:
        return res1
 
    """ node is not found in left,
    so recur on right subtree """
    res2 = searchNode(node.right, key)
    return res2

def getLevelUtil(node, data, level):
    if (node == None):
        return 0
 
    if (node.data == data):
        return level
 
    downlevel = getLevelUtil(node.left, data, level + 1)

    if (downlevel != 0):
        return downlevel
 
    downlevel = getLevelUtil(node.right, data, level + 1)
    return downlevel
 
# Returns level of given data value
 
 
def getLevel(node, data):
    return getLevelUtil(node, data, 1)
 

def setLevel(data_loader):
    for d in data_loader:
        for data in d:
            max_level = 0
            tree = list(data.keys())[0]
            n_nodes = data[tree]#[0]
            count = []
            numerarNodos(tree, count)
            for x in range(0, n_nodes):
                level = getLevel(tree, x)
                if level > max_level:
                    max_level = level
                if (level):
                    node = searchNode(tree, x)
                    node.level = getLevel(tree, x)
                else:
                    print(x, "is not present in tree")
            tree_level = []
            tree.getTreeLevel(tree, tree_level)
            tree_level = [max_level - nodelevel for nodelevel in tree_level]
            tree.setTreeLevel(tree, sum(tree_level))
            tree.setMaxLevel(tree, max_level)

'''
def StructureLoss(cl_p, original, mult):
        
        if original is None:
            return
        ce = nn.CrossEntropyLoss(weight = mult)

        if original.childs() == 0:
            vector = [1, 0, 0] 
        if original.childs() == 1:
            vector = [0, 1, 0]
        if original.childs() == 2:
            vector = [0, 0, 1] 

        c = ce(cl_p, torch.tensor(vector, device=device, dtype = torch.float).reshape(1, 3))
        return c
'''
def numberNodes(data_loader, batch_size):
    n_no = []
    qzero = 0
    qOne = 0
    qtwo = 0
    for batch in data_loader:
        for treed in batch:
            tree = list(treed.keys())[0]
            n = treed[tree]
            n_no.append(n)
            li = []
            tree.traverseInorderChilds(tree, li)
            zero = [a for a in li if a == 0]
            one = [a for a in li if a == 1]
            two = [a for a in li if a == 2]
            qzero += len(zero)
            qOne += len(one)
            qtwo += len(two)

    qzero /= len(data_loader)*batch_size
    qOne /= len(data_loader)*batch_size
    qtwo /= len(data_loader)*batch_size
    if round(qzero) == 0:
        qzero = 1
    if round(qOne) == 0:
        qOne = 1
    if round(qtwo) == 0:
        qtwo = 1
    mult = torch.tensor([1/round(qzero),1/round(qOne),1/round(qtwo)], device = device)
    return mult
    

class Node:
    """
    Class Node
    """
    def __init__(self, value, radius, left = None, right = None, level = None, treelevel = None, maxlevel = None):
        self.left = left
        self.data = value
        self.radius = radius
        self.right = right
        self.children = [self.left, self.right]
        self.level = level
        self.treelevel = treelevel
        self.maxlevel = maxlevel
    
    def agregarHijo(self, children):

        if self.right is None:
            self.right = children
        elif self.left is None:
            self.left = children

        else:
            raise ValueError ("solo arbol binario ")


    def isLeaf(self):
        if self.right is None and self.left is None:
            return True
        else:
            return False

    def isTwoChild(self):
        if self.right is not None and self.left is not None:
            return True
        else:
            return False

    def isOneChild(self):
        if self.isTwoChild():
            return False
        elif self.isLeaf():
            return False
        else:
            return True

    def childs(self):
        if self.isLeaf():
            return 0
        if self.isOneChild():
            return 1
        else:
            return 2
    
    
    def traverseInorder(self, root):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorder(root.left)
            print (root.data, root.radius)
            self.traverseInorder(root.right)

    

    def traverseInorderwl(self, root):
        """
        traverse function will print all the node in the tree, including node level and tree level.
        """
        if root is not None:
            self.traverseInorderwl(root.left)
            print (root.data, root.radius, root.level)
            #print (root.data, root.radius, root.level, root.treelevel, (root.maxlevel+1-root.level)/root.treelevel)
            #print (root.data, root.radius, root.level, root.treelevel, root.level/root.treelevel)
            self.traverseInorderwl(root.right)

    def getTreeLevel(self, root, c):
        """
        
        """
        if root is not None:
            self.getTreeLevel(root.left, c)
            c.append(root.level)
            self.getTreeLevel(root.right, c)

    def setTreeLevel(self, root, c):
        """
        
        """
        if root is not None:
            self.setTreeLevel(root.left, c)
            root.treelevel = c
            self.setTreeLevel(root.right, c)

    def setMaxLevel(self, root, m):
        """
        
        """
        if root is not None:
            self.setMaxLevel(root.left, m)
            root.maxlevel = m
            self.setMaxLevel(root.right, m)



    def traverseInorderChilds(self, root, l):
        """
        
        """
        if root is not None:
            self.traverseInorderChilds(root.left, l)
            l.append(root.childs())
            self.traverseInorderChilds(root.right, l)
            return l



    def height(self, root):
    # Check if the binary tree is empty
        if root is None:
            return 0 
        # Recursively call height of each node
        leftAns = self.height(root.left)
        rightAns = self.height(root.right)
    
        # Return max(leftHeight, rightHeight) at each iteration
        return max(leftAns, rightAns) + 1

    # Print nodes at a current level
    def printCurrentLevel(self, root, level):
        if root is None:
            return
        if level == 1:
            print(root.data, end=" ")
        elif level > 1:
            self.printCurrentLevel(root.left, level-1)
            self.printCurrentLevel(root.right, level-1)

    def printLevelOrder(self, root):
        h = self.height(root)
        for i in range(1, h+1):
            self.printCurrentLevel(root, i)


    def countNodes(self, root, counter):
        if   root is not None:
            self.countNodes(root.left, counter)
            counter.append(root.data)
            self.countNodes(root.right, counter)
            return counter

    
    def serialize(self, root):
        def post_order(root):
            if root:
                post_order(root.left)
                post_order(root.right)
                ret[0] += str(root.data)+'_'+ str(root.radius) +';'
                
            else:
                ret[0] += '#;'           

        ret = ['']
        post_order(root)
        return ret[0][:-1]  # remove last ,

    def toGraph( self, graph, index, dec, flag, proc=True):
        
        radius = self.radius.cpu().detach().numpy()
        if dec:
            radius= radius[0]
       
        if flag == 0:
            b = True
            flag = 1
        else:
            b = False
        graph.add_nodes_from( [ (self.data, {'posicion': radius[0:3], 'radio': radius[3], 'root': b} ) ])

        
        if self.right is not None:
            self.right.toGraph( graph, index + 1, dec, flag = 1)#
            graph.add_edge( self.data, self.right.data )
           
        if self.left is not None:
            self.left.toGraph( graph, 0, dec, flag = 1)#

            graph.add_edge( self.data, self.left.data)
           
        else:
            return

class Sampler(nn.Module):
    
    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)
        self.LeakyReLu = nn.LeakyReLU()
        self.latent_dim = feature_size
        self.dropout = nn.Dropout(0.1)

        
    def forward(self, input):
        encode = self.LeakyReLu(self.mlp1(input))
        #encode = self.dropout(encode)
        mu = self.mlp2mu(encode)
        logvar = self.mlp2var(encode)
        
   
        std = logvar.mul(0.5).exp_() # calculate the STDEV
        eps = torch.Tensor(std.size()).normal_().cuda() # random normalized noise

        KLD_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.training:
            out = torch.cat([eps.mul(std).add_(mu), KLD_element], 1)
        else:
            out = mu
        return out

class InternalEncoder(nn.Module):
    
    def __init__(self, input_size: int, feature_size: int, hidden_size: int):
        super(InternalEncoder, self).__init__()
        
        # Encoders atributos
        self.attribute_lin_encoder_1 = nn.Linear(input_size,hidden_size)
        self.attribute_lin_encoder_2 = nn.Linear(hidden_size,feature_size)

        # Encoders derecho e izquierdo
        self.right_lin_encoder_1 = nn.Linear(feature_size,hidden_size)
        self.right_lin_encoder_2 = nn.Linear(hidden_size,feature_size)
       

        self.left_lin_encoder_1  = nn.Linear(feature_size,hidden_size)
        self.left_lin_encoder_2  = nn.Linear(hidden_size,feature_size)
        
        # Encoder final
        self.final_lin_encoder_1 = nn.Linear(2*feature_size, feature_size)

        # Funciones / Parametros utiles
        self.LeakyReLu = nn.LeakyReLU() 
        self.feature_size = feature_size


    def forward(self, input, right_input, left_input):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Encodeo los atributos
        attributes = self.attribute_lin_encoder_1(input)
        attributes = self.LeakyReLu(attributes)
        attributes = self.attribute_lin_encoder_2(attributes)
        attributes = self.LeakyReLu(attributes)


        # Encodeo el derecho
        if right_input is not None:
            context = self.right_lin_encoder_1(right_input)
            context = self.LeakyReLu(context)
            context = self.right_lin_encoder_2(context)
            context = self.LeakyReLu(context)
            
            # Encodeo el izquierdo
            if left_input is not None:
                left = self.left_lin_encoder_1(left_input)
                #print("izquierdo", left.shape)
                left = self.LeakyReLu(left)
                context += self.left_lin_encoder_2(left)
                context = self.LeakyReLu(context)
        else:
            context = torch.zeros(input.shape[0],self.feature_size, requires_grad=True, device=device)
        
        feature = torch.cat((attributes,context), 1)
        feature = self.final_lin_encoder_1(feature)
        feature = self.LeakyReLu(feature)
        return feature

class GRASSEncoder(nn.Module):
    
    def __init__(self, input_size: int, feature_size : int, hidden_size: int):
        super(GRASSEncoder, self).__init__()
        self.leaf_encoder = InternalEncoder(input_size,feature_size, hidden_size)
        self.internal_encoder = InternalEncoder(input_size,feature_size, hidden_size)
        self.bifurcation_encoder = InternalEncoder(input_size,feature_size, hidden_size)
        self.sample_encoder = Sampler(feature_size = feature_size, hidden_size = hidden_size)
        
    def leafEncoder(self, node, right=None, left = None):
        return self.internal_encoder(node, right, left)
    def internalEncoder(self, node, right, left = None):
        return self.internal_encoder(node, right, left)
    def bifurcationEncoder(self, node, right, left):
        return self.bifurcation_encoder(node, right, left)
    def sampleEncoder(self, feature):
        return self.sample_encoder(feature)

class NodeClassifier(nn.Module):
    
    def __init__(self, latent_size : int, hidden_size : int):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(latent_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, 3)
        self.LeakyReLu = nn.LeakyReLU()

    def forward(self, input_feature):
        output = self.mlp1(input_feature)
        output = self.LeakyReLu(output)
        output = self.mlp2(output)
        output = self.LeakyReLu(output)
        output = self.mlp3(output)
        return output

class SampleDecoder(nn.Module):
    """ Decode a randomly sampled noise into a feature vector """
    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, feature_size)
        #self.mlp4 = nn.Linear(hidden_size, feature_size)
        #self.mlp5 = nn.Linear(feature_size, feature_size)
        #self.dropout = nn.Dropout(0.1)
        
        self.LeakyReLu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, input_feature):
        output = self.LeakyReLu(self.mlp1(input_feature))
        #output = self.dropout (output)
        output = self.tanh(self.mlp2(output))
        output = self.tanh(self.mlp3(output))
        #output = self.LeakyReLu(self.mlp4(output))
        #output = self.LeakyReLu(self.mlp5(output))
        
        return output

class Decoder(nn.Module):
    
    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self, latent_size : int, hidden_size : int):
        super(Decoder, self).__init__()
        
        self.mlp = nn.Linear(latent_size,hidden_size)
        self.mlp_left = nn.Linear(hidden_size, hidden_size)
        self.mlp_left2 = nn.Linear(hidden_size, latent_size)
        self.mlp_right = nn.Linear(hidden_size, hidden_size)
        self.mlp_right2 = nn.Linear(hidden_size, latent_size)
        self.mlp2 = nn.Linear(hidden_size,latent_size)
        #self.mlp4 = nn.Linear(hidden_size,latent_size)
        self.mlp3 = nn.Linear(latent_size,4)
        self.LeakyReLu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        #self.dropout = nn.Dropout(0.5)

    def common_branch(self, parent_feature):
      
        vector = self.mlp(parent_feature)
        vector = self.LeakyReLu(vector)
        return vector

    def attr_branch(self, vector):
        vector = self.mlp2(vector)
        vector = self.LeakyReLu(vector)
        #vector = self.dropout(vector)
        vector = self.mlp3(vector) 
        vector = self.LeakyReLu(vector)    
        return vector

    def right_branch(self, vector):
        right_feature = self.mlp_right(vector)
        right_feature = self.LeakyReLu(right_feature)
        #right_feature = self.dropout(right_feature)
        right_feature = self.mlp_right2(right_feature)
        right_feature = self.LeakyReLu(right_feature)
        return right_feature

    def left_branch(self, vector):
        left_feature = self.mlp_left(vector)
        left_feature = self.LeakyReLu(left_feature)
        #left_feature = self.dropout(left_feature)
        left_feature = self.mlp_left2(left_feature)
        left_feature = self.LeakyReLu(left_feature)
        return left_feature

    def forward(self, parent_feature):
      
        vector      = self.common_branch(parent_feature)
        attr_vector = self.attr_branch(vector)

        return attr_vector 

    def forward1(self, parent_feature):

        vector       = self.common_branch(parent_feature)
        attr_vector  = self.attr_branch(vector)
        right_vector = self.right_branch(vector)
        return right_vector, attr_vector

    def forward2(self, parent_feature):
       

        vector       = self.common_branch(parent_feature)
        attr_vector  = self.attr_branch(vector)
        right_vector = self.right_branch(vector)
        left_vector  = self.left_branch(vector)
        return left_vector, right_vector, attr_vector



class GRASSDecoder(nn.Module):
 
    def __init__(self, latent_size : int, hidden_size: int, mult: torch.Tensor):
        super(GRASSDecoder, self).__init__()
        self.decoder = Decoder(latent_size, hidden_size)
        self.node_classifier = NodeClassifier(latent_size, hidden_size)
        self.sample_decoder = SampleDecoder(feature_size = latent_size, hidden_size = hidden_size)
        self.mseLoss = nn.MSELoss()  # pytorch's mean squared error loss
        self.ceLoss = nn.CrossEntropyLoss(weight = mult)  # pytorch's cross entropy loss (NOTE: no softmax is needed before)
        self.alfa = 0.3

    def featureDecoder(self, feature):
        return self.decoder.forward(feature)

    def internalDecoder(self, feature):
        return self.decoder.forward1(feature)

    def bifurcationDecoder(self, feature):
        return self.decoder.forward2(feature)

    def nodeClassifier(self, feature):
        return self.node_classifier(feature)

    def sampleDecoder(self, feature):
        return self.sample_decoder(feature)
        
    def calcularLossAtributo(self, nodo, radio):
        
        if nodo is None:
            return
        else:
           
            nodo = torch.stack(nodo)
            #print("nodo", radio)
            l = [self.mseLoss(b.reshape(1,4), gt.reshape(1,4)).mul(1-self.alfa) for b, gt in zip(radio.reshape(-1,4), nodo.reshape(-1,4))]
            
            return l
        
    def calcularLossAtributo2(self, nodo, radio):
        
        if nodo is None:
            return
        else:
            nodo = torch.stack([nodo[:3]])
            radio = radio[0][:3]
            l = [self.mseLoss(b.reshape(1,3), gt.reshape(1,3)).mul(1-self.alfa) for b, gt in zip(radio.reshape(-1,3), nodo.reshape(-1,3))]
            
            return l


    def classifyLossEstimator(self, label_vector, original):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if original is None:
            return
        else:
           
            v = []
            for o in original:
                if o == 0:
                    vector = torch.tensor([1, 0, 0], device = device, dtype = torch.float)
                if o == 1:
                    vector = torch.tensor([0, 1, 0], device = device, dtype = torch.float)
                if o == 2:
                    vector = torch.tensor([0, 0, 1], device = device, dtype = torch.float)
                v.append(vector)

            v = torch.stack(v)
            l = [self.ceLoss(b.reshape(1,3), gt.reshape(1,3)).mul(self.alfa) for b, gt in zip(label_vector.reshape(-1,3), v.reshape(-1,3))]
            return l
            
    def vectorAdder(self, v1, v2):
   
        v = v1.add(v2)
       
        return v

    def vectorMult(self, m, v):
        #print("v", v)
        #print("m", m)
       
        z = zip(v, m)
        r = []
        for c, d in z:
            r.append(torch.mul(c, d))
        return r