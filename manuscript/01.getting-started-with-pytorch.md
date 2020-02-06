
# Getting Started with PyTorch

[PyTorch](https://pytorch.org/) is:

> An open source machine learning framework that accelerates the path from research prototyping to production deployment.

In my humble opinion, PyTorch is the [sweet](https://www.youtube.com/watch?v=-h4spfuMGDI) way to solve Machine Learning problems, in the real world! The vast community allows you to work state-of-the-art models and deploy them to production in no time (relatively speaking). Let's get started!

    In [0]: !pip install -q -U torch watermark

    In [0]: %load_ext watermark
            %watermark -v -p numpy,torch
    
    Out[0]: CPython 3.6.9
            IPython 5.5.0
            
            numpy 1.17.5
            torch 1.4.0




## PyTorch ❤ NumPy

Do you know NumPy? If you do, learning PyTorch will be a breeze! If you don't, prepare to learn the skills that will guide you on your journey Machine Learning Mastery!

Let's start with something simple:

    In [0]: import torch
            import numpy as np

    In [0]: a = np.array([1, 2])
            b = np.array([8, 9])
            
            c = a + b
            c
    
    Out[0]: array([ 9, 11])


Adding the same arrays with PyTorch looks like this:

    In [0]: a = torch.tensor([1, 2])
            b = torch.tensor([8, 9])
            
            c = a + b
            c
    
    Out[0]: tensor([ 9, 11])


Fortunately, you can go from NumPy to PyTorch:

    In [0]: a = torch.tensor([1, 2])
            
            a.numpy()
    
    Out[0]: array([1, 2])


 and vice versa:

    In [0]: a = np.array([1, 2])
            torch.from_numpy(a)
    
    Out[0]: tensor([1, 2])


The good news is that the conversions incur almost no cost on the performance of your app. The NumPy and PyTorch store data in memory in the same way. That is, PyTorch is reusing the work done by NumPy.

## Tensors

Tensors are just n-dimensional number (including booleans) containers. You can find the complete list of supported data types at [PyTorch's Tensor Docs](https://pytorch.org/docs/stable/tensors.html).

So, how can you create a Tensor (try to ignore that I've already shown you how to do it)?



    In [0]: torch.tensor([[1, 2], [2, 1]])
    
    Out[0]: tensor([[1, 2],
                    [2, 1]])


You can create a tensor from floats:

    In [0]: torch.FloatTensor([[1, 2], [2, 1]])
    
    Out[0]: tensor([[1., 2.],
                    [2., 1.]])


Or define the type like so:

    In [0]: torch.tensor([[1, 2], [2, 1]], dtype=torch.bool)
    
    Out[0]: tensor([[True, True],
                    [True, True]])


You can use a wide range of factory methods to create Tensors without manually specifying each number. For example, you can create a matrix with random numbers like this: 

    In [0]: torch.rand(3, 2)
    
    Out[0]: tensor([[0.6686, 0.7622],
                    [0.0341, 0.5835],
                    [0.2423, 0.0651]])


Or one full of ones:

    In [0]: torch.ones(3, 2)
    
    Out[0]: tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.]])


PyTorch has a variety of useful operations:

    In [0]: x = torch.tensor([[2, 3], [1, 2]])
            print(x)
            print(f'sum: {x.sum()}')
    
    Out[0]: tensor([[2, 3],
                    [1, 2]])
            sum: 8



Get the transpose of a 2-D tensor:

    In [0]: x.t()
    
    Out[0]: tensor([[2, 1],
                    [3, 2]])


Get the shape of each dimension:

    In [0]: x.size()
    
    Out[0]: torch.Size([2, 2])


Generally, performing some operation creates a new Tensor:

    In [0]: y = torch.tensor([[2, 2], [5, 1]])
            z = x.add(y)
            z
    
    Out[0]: tensor([[4, 5],
                    [6, 3]])


But you can do it in-place:

    In [0]: x.add_(y)
            x
    
    Out[0]: tensor([[4, 5],
                    [6, 3]])


Almost all operations have an in-place version - the name of the operation, followed by an underscore.


## Running on GPU

At this point, you might be like: "Why do I need PyTorch at all? All of this is perfectly doable with NumPy?". PyTorch has three major superpowers: 
- you can run your operations on the GPU(s) (or something else)
- [Autograd: automatic differentiation](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- A set of tools to build Neural Networks. Including several additional packages for working [with text](https://github.com/pytorch/text) or [images](https://github.com/pytorch/vision).

Doing your Deep Learning computations on the GPU speeds up your experiment by a lot! And PyTorch makes it ridiculously easy to do it. Let's start by checking if GPU is available:

    In [0]: device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            device
    
    Out[0]: device(type='cuda')


Good, we have a [CUDA](https://en.wikipedia.org/wiki/CUDA)-enabled GPU device on our hands. Let's store a Tensor on it:

    In [0]: x = torch.tensor([[2, 3], [1, 2]])
            x.to(device)
    
    Out[0]: tensor([[2, 3],
                    [1, 2]], device='cuda:0')


Notice that our Tensor is now on device `cuda:0`. What can we do with it? Pretty much everything as before:

    In [0]: x = x.to(device)
            
            y = torch.tensor([[2, 2], [5, 1]])
            y = y.to(device)
            
            x.add(y)
    
    Out[0]: tensor([[4, 5],
                    [6, 3]], device='cuda:0')



## Common Issues

I got to be honest with you. You will fuck up, multiple times, before understanding how this whole thing works out. That's alright!

However, there are a couple of things you can do that might minimize the frustrations along your journey:

- Doing operations between GPU and CPU Tensors is **not allowed**
- Size mismatch between Tensors occurs often and is (almost every time) easy to fix:



    In [0]: a = torch.ones(2, 2) 
            b = torch.ones(1, 3)
            a * b
    
    Out[0]: ---------------------------------------------------------------------------
            RuntimeError                              Traceback (most recent call last)
            <ipython-input-21-b2a4e8765762> in <module>()
                  1 a = torch.ones(2, 2)
                  2 b = torch.ones(1, 3)
            ----> 3 a * b
            
            RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1



PyTorch is very descriptive in this case. When doing more complex stuff, you would want to check the shape of your Tensors obsessively, after every operation. Just print the size!

- Running out of GPU memory: You might be leaking memory or too large of a dataset/model. Faster/better GPU always helps. But remember, you can solve really large problems with a single powerful GPU these days. Think carefully if that is not enough for you - why that is?

## Conclusion

Welcome to the dark side! You might've been working with Keras, TensorFlow, or another Deep Learning framework, until recently. Almost every framework is great, but PyTorch has really solid roots. Easy to use and understand, allows for fast experimentation and standard debugging tools apply! Enjoy!


## References

- ["PyTorch: A Modern Library for Machine Learning" with Adam Paszke](https://www.youtube.com/watch?v=5bSAipCNqXo)
- [Recitation 1 | Your First Deep Learning Code](https://www.youtube.com/watch?v=KrCp_yPVOxs)
- [What is PyTorch?](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)

