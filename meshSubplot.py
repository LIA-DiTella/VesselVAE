import uuid
from ipywidgets import Output, HBox
import meshplot as mp
rendertype = 'JUPYTER'
import numpy as np
import networkx as nx

class Subplot():
    def __init__(self, data, view, s):
        if data == None:
            self.rows = []
            self.hboxes = []
        else:
            self.rows = data.rows
        if s[0] != 1 or s[1] != 1:
            if data == None: # Intialize subplot array
                cnt = 0
                for r in range(s[0]):
                    row = []
                    for c in range(s[1]):
                        row.append(Output())
                        cnt += 1
                    self.rows.append(row)

                for r in self.rows:
                    hbox = HBox(r)
                    if rendertype == "JUPYTER":
                        display(hbox)
                    self.hboxes.append(hbox)
        
            out = self.rows[int(s[2]/s[1])][s[2]%s[1]]
            if rendertype == "JUPYTER":
                with out:
                    display(view._renderer)
            self.rows[int(s[2]/s[1])][s[2]%s[1]] = view

    def save(self, filename=""):
        if filename == "":
            uid = str(uuid.uuid4()) + ".html"
        else:
            filename = filename.replace(".html", "")
            uid = filename + '.html'

        s = ""
        imports = True
        for r in self.rows:
            for v in r:
                s1 = v.to_html(imports=imports, html_frame=False)
                s = s + s1
                imports = False

        s = "<html>\n<body>\n" + s + "\n</body>\n</html>"
        with open(uid, "w") as f:
            f.write(s)
        print("Plot saved to file %s."%uid)

    def to_html(self, imports=True, html_frame=True):
        s = ""
        for r in self.rows:
            for v in r:
                s1 = v.to_html(imports=imports, html_frame=html_frame)
                s = s + s1
                imports = False

        return s

def subplot(f, c = 'red', uv=None, n=None, shading={}, s=[1, 1, 0], data=None, **kwargs):
    
    shading={'point_size':0.05, "point_color": c, "line_color": c, "width":400, "height":400}
    view = mp.Viewer(settings = {"width": 500, "height": 500, "antialias": True, "scale": 1.5, "background": "#ffffff",
                "fov": 30})

    #obj = view.add_points(np.array([ f.nodes[v]['posicion'] for v in f.nodes]), shading=shading)
    obj = view.add_points( np.array([f.nodes[v]['posicion'] for v in f.nodes if f.nodes[v]['root'] == True]), shading={'point_size':.02, 'point_color':'red'})
    if len(np.array([ f.nodes[v]['posicion'] for v in f.nodes if f.nodes[v]['root'] == False])) != 0:
        obj = view.add_points( np.array([f.nodes[v]['posicion'] for v in f.nodes if f.nodes[v]['root'] == False]), shading={'point_size':.02, 'point_color':'black'})
    for arista in f.edges:
        obj = view.add_lines( f.nodes[arista[0]]['posicion'], f.nodes[arista[1]]['posicion'], shading  = shading)
   
    subplot = Subplot(data, view, s)
    return subplot

def plotTree( root, dec ):
    graph = nx.Graph()
    root.toGraph( graph, 0, dec, 0)
    edges=nx.get_edge_attributes(graph,'procesada')

    p = mp.plot( np.array([ graph.nodes[v]['posicion'] for v in graph.nodes if graph.nodes[v]['root'] == True]), shading={'point_size':0.05, 'point_color':'red'}, return_plot=True)
    if len(np.array([ graph.nodes[v]['posicion'] for v in graph.nodes if graph.nodes[v]['root'] == False])) != 0:
        p.add_points( np.array([graph.nodes[v]['posicion'] for v in graph.nodes if graph.nodes[v]['root'] == False]), shading={'point_size':.05, 'point_color':'black'})
    for arista in graph.edges:
        p.add_lines( graph.nodes[arista[0]]['posicion'], graph.nodes[arista[1]]['posicion'])

    return 


def sTree( root, dec, s, c, d=None):
    "plot trees next to each other"
    graph = nx.Graph()
    root.toGraph( graph, 0, dec, 0)
  
    if d:
        subplot(graph, c=c, s=s, data = d)
    else:
        
        d = subplot(graph, c=c, s=s)

    return d



