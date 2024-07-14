import torch
from torchviz import make_dot

def main():
    x=torch.ones(2, requires_grad=True)
    y=2*x
    z=3+x
    r=(y+z).sum()
    graph = make_dot(r)
    graph.format = 'png'
    graph.render('graph')

if __name__ == "__main__":
    main()