from __future__ import absolute_import
import collections
import torch

class Fold(object):

    class Node(object):
        def __init__(self, op, step, index, *args):
            self.op = op
            self.step = step
            self.index = index
            self.args = args
            self.split_idx = -1
            self.batch = True

        
        def split(self, num): ##lo uso cuando la red da mas de un tensor como output
            u"""Split resulting node, if function returns multiple values."""
            #print("op", self.op)
            #print("step", self.step)
            #print("arg", self.args)

            nodes = []
            for idx in range(num):
                nodes.append(Fold.Node(
                    self.op, self.step, self.index, *self.args))
                nodes[-1].split_idx = idx
                #print("idx", idx)
                #print("nodes", nodes)
                #print("nodes", nodes[-1].split_idx)
            return tuple(nodes)
            

        def nobatch(self):
            self.batch = False
            return self
            

        def get(self, values):
            
            if self.split_idx >= 0:
                #print("split index", self.split_idx)
                #print("v0",values[self.step][self.op])
                #print("v1",values[self.step][self.op][self.split_idx])
                #print("v2",values[self.step][self.op][self.split_idx][self.index])
                return values[self.step][self.op][self.split_idx][self.index]
            else:
                return values[self.step][self.op][self.index]

        def __repr__(self):
            return u"[%d:%d]%s" % (
                self.step, self.index, self.op)

    def __init__(self, volatile=False, cuda=False, variable=True):
        self.steps = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.cached_nodes = collections.defaultdict(dict)
        self.total_nodes = 0
        self.volatile = volatile
        self._cuda = cuda
        self._variable = variable

    def __repr__(self):
        return str(self.steps.keys())

    def cuda(self):
        self._cuda = True
        return self

    def add(self, op, *args):
        u"""Add op to the fold."""

        self.total_nodes += 1
        # si el nodo no fue visitado antes 

        if args not in self.cached_nodes[op]:
  
            #arg a veces son solo los features del nodo, a veces tiene info de los hijos tambien
            step = max([0] + [arg.step + 1 for arg in args if isinstance(arg, Fold.Node)]) #step es nivel
            node = Fold.Node(op, step, len(self.steps[step][op]), *args)#voy creando nodos fold y agregndolos a cached nodes
            #len(self.steps[step][op] es index, cuenta los nodos por nivel
            #en steps guardo los nodos, por "step"=nivel, y operacion
        
            self.steps[step][op].append(args)
            self.cached_nodes[op][args] = node
       
        return self.cached_nodes[op][args]


    def _batch_args(self, arg_lists, values, op):
        res = []
        for arg in arg_lists:
            #print("arg apply", arg)
            r = []
            #si es un nodo de fold
            #si viene un "nodo" fold, obtengo todos los argumentos que tiene ese nodo y los concateno en un solo vector
            #print("op", op)
            if isinstance(arg[0], Fold.Node):
                
                #print("arg", arg)
                if arg[0].batch:
                    for x in arg:
                        #print("x", x)
                        r.append(x.get(values))
                    #print("r sin stack", r)
                    #print("r con stack", torch.stack(r))
                    res.append(torch.stack(r))
                #if op == 'sampleEncoder':
                #    print("arg", arg)
                #    print("r", r)
                    
                
                #nunca uso este caso
                '''
                else:
                    for i in range(2, len(arg)):
                        if arg[i] != arg[0]:
                            raise ValueError(u"Can not use more then one of nobatch argument, got: %s." % str(arg))
                    x = arg[0]
                    res.append(x.get(values))
                    '''
            else:           
                #print("else")
                #si es un tensor de atributos     
                if isinstance(arg[0], torch.Tensor):  
         
                    var = torch.stack(arg)
                    res.append(var)
                
                #si es un nodo de arbol
                else:
                 
                    if op != "classifyLossEstimator" and op != "calcularLossAtributo" and op != "vectorMult" and op != "sampleEncoder": #en caso de que op sea alguna red
                        var = arg[0].radius
                    elif op == 'sampleEncoder':
                        print("arg", arg)
                        var = arg
                    elif op == "calcularLossAtributo": #en caso de estar calculano mse
                        #var = [(a.radius, a.childs()) for a in arg]
                        var = [a.radius for a in arg]
                        #print("var", var)
                    elif op == "classifyLossEstimator":
                        var = [a.childs() for a in arg] #en caso de estar calculando cross entropy
                    elif op == "vectorMult":
                        #print("arg",arg)
                        if isinstance(arg, torch.Tensor):
                            var = arg
                        else:
                            var = list(arg)
                            #print("var",var)
                    
                    res.append(var)
            #if op == 'sampleEncoder':
            #    print("res", res)

        return res

    def apply(self, nn, nodes):
        u"""Apply current fold to given neural module."""
        values = {}
        for step in sorted(self.steps.keys()):
            
            values[step] = {}
            for op in self.steps[step]:
                
                func = getattr(nn, op)
                #if op == 'sampleEncoder':
                #    print("nodes", nodes)
                ##junto los atributos de los nodos que estan en el mismo step y op
                try:                    
                    batched_args = self._batch_args(
                        zip(*self.steps[step][op]), values, op)
                except Exception:
                    print("Error while executing node %s[%d] with args: %s" % (op, step, self.steps[step][op]))
                    raise
               
                
                res = func(*batched_args)
                #if op == 'bifurcationDecoder':
                #    print("res", res)
                
                if isinstance(res, (tuple, list)):
                    values[step][op] = []
                    for x in res:
                        #values[step][op].append(torch.chunk(x, arg_size))
                        values[step][op].append(x)
                else:
                    if len(res.shape) == 1 and op != 'vectorAdder' and op != 'vectorMult':
                        values[step][op] = res.reshape(-1, 4)
                    else: #los vectores de output del clasificador tienen tres elementos, no hago el reshape
                        values[step][op] = res
                        

                       
        try:

            return self._batch_args(nodes, values, op)
        except Exception:
            print("cannot batch")
            raise