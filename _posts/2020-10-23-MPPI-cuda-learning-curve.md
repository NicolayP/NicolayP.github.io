---
layout: single
title: "Model predictive path integral cuda implementation"
date: 2020-10-23
categories: control rl model-based cuda gpu
permalink: "MPPI-implementation"
modified: 2020-10-23
description: Documentation of learning curve of cuda implementation around MPPI.
tags:
     - Reinforcement Learning
     - model based
     - cuda
     - GPU
     - MPPI
     - Tutorial
header:
---

## Content
  1. [Introduction](#chap:intro)
    1. [GPU operation](#sec:gpu)
    2. [CUDA framework](#sec:cuda)
  2. [Implementation](#chap:impl)
    1. [Simple Example](#sec:simple)
    2. [First Class](#sec:class)
    3. [Class with Pointers](#sec:classptr)
  3. [MPPI implementation](#chap:mppi)
  4. [Comments](#sec:disc)

# Introduction<a name="chap:intro"></a>
 In this post we document our implementation of MPPI in on a GPU-cuda compatible
 Device. In the same time, this can also be used as tutorial to learn how to program
 in an OOP paradigm on a GPU.

## How does a GPU operates.<a name="sec:gpu"></a>
 How does a GPU compares to a CPU? Well the answer is pretty simple: It's almost the same. While there a bit more complexity in the actual architecture, a GPU is mainly composed of a memory and a LOT of computation units. A bit like the multiple cores you have on your CPU, you can see the GPU as a CPU with a LOT of cores. Starting from now we will refere to the GPU as the device and the CPU as the host. Now lets see how the CUDA language allows us to program
 ou GPU.

## CUDA framework:<a name="sec:cuda"></a>

  The CUDA language provides a compiler nvcc with a library that allows us to write c++ code and run bits of the code on the device. The main process of a CUDA code will be the following:
  - run sequential code.
  - copy data from host to device memory.
  - launch a kernel that uses the GPU to compute operation in parallel on the device memory data.
  - copy the data from the device to the host.
  - run sequential code.

  Now let's introduce some keywords from CUDA. CUDA has three decorator for functions.
  - __\__global\____: This decorator specifies that a function will be called from the host and executed on the device. Note: NONE OF YOUR MEMEBER FUNCTION CAN HAVE THE GLOBAL DECORATOR. I'll later why when we'll reach the classe stored on the device.
  - __\__host\____: Pretty intuitively those functions can be called from the host and run on the host.
  - __\__device\____: alternative those functions can be called from the device and run on the device.

  Next the following function will be useful for us:
  - cudaMalloc((void**) *devptr*, size_t *size*);
  - cudaMemcpy((void\*) *destptr*, (void\*) *srcptr*, size_t *bytes*, *cudaDir*);

  The first function works similar to malloc in C. It will allocate the data for the pointer on the device.

  The second function copies a *bytes* size of data from the *srcptr* and the *destptr*. The last argument of this function specifies the direction of the copy. The two different values are *cudaMemcpyDeviceToHost* and *cudaMemcpyHostToDevice*. Both explicit enough i think.

  Finally to run __\__global\____ functions on the device we need to call the function using the following decorator __<<<blockDim, blockSize>>>__ This will tell the compiler that this function will run in parallel. Okay so now that we have all our tools let's get started with some simple examples.

  __blockDim__:

  __blockSize__:

  If you want more information on a GPU architecture and the CUDA library, there is plenty of documentation out there for you to read. Depending on the results of the first full implementation of MPPI, we will writ articles about caching the data with common threads, see how to parallelize the code even more etc.

# Implementation on GPU with cuda.<a name="chap:impl"></a>

## Simple example<a name="sec:simple"></a>

  So let's start with a very simple application. The goal will be to add two arrays together.
  A simple solution is thus to write a for loop

``` c++
    void add(int* o, int* a, int* b, int n){
      for (int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
      }
    }
```
  So now let's add parallelization into this computation. To do so let us first modify our function to be a device function callable by the host code. Then we will setup the data and finally we create a kernel where the code will run.

``` c++
    __global__ void add(int* o, int* a, int* b){
      for (int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
      }
    }
    //
    int main(){

      int* a = (int*) malloc(sizeof(int)*n);
      int* b = (int*) malloc(sizeof(int)*n);
      int* o = (int*) malloc(sizeof(int)*n);

      int* d_a;
      int* d_b;
      int* d_o;

      //init data
      for (int i = 0; i < n; i++){
        a[i] = 1;
        b[i] = 2;
      }

      //allocate device memory and copy the data to the device
      cudaMalloc((void**)&d_a, sizeof(int)*n);
      cudaMalloc((void**)&d_b, sizeof(int)*n);
      cudaMalloc((void**)&d_o, sizeof(int)*n);
      cudaMemcpy(d_a, a, sizeof(int)*n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, b, sizeof(int)*n, cudaMemcpyHostToDevice);


      add<<<1, 1>>>(d_o, d_a, d_b, n);
      cudaMemcpy(o, d_h, sizeof(int)*n, cudaMemcpyDeviceToHost);

      // Free memory
    }
```

  The entire code for those examples can be found on my [github](h) as well as a CMakeLists file.

  But this runs but the performances isn't improved! That's because when we launched the kernel with only one thread on one block. To understand this, we need understand how the code is running on the GPU. When I told you earlier that a GPU can be viewed as a bunch o cores, it wasn't entirely correct. A GPU is composed of multiple SM (Streaming multiprocessor) Each of these multiprocessor is able to run thread concurrently. Thus when launching a kernel, __\__blockDim\____ corresponds to the number of block, each block being assigned to one SM when it is idle. __\__blockSize\____ indicates the number of thread to be run on a block. But if we do so as it is it will not work. Why? Because the kernel is running the same instruction in parallel. So how can we do this?
  Cuda exposes three variables to us __blockDim__, __blockIdx__ and __threadIdx__. This three variables are intuitive to understand and are assigned to every thread on a block.

  So how can we parallelize the code? We can clearly see that all the operations at different index are independent from one another. So we can imagine chopping our array in __blockDim__ chuncks with length __blockSize__. Thus using the three different value to create a unique index for every entry of the table. This is illustrated in the following chunk of code:

```c++
    __global__ void add(int* o, int* a, int* b, int n){
      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      if (tid < n)
        out[tid] = a[tid] + b[tid];
    }

    ...

    add<<<1 + n/256, 256>>>(d_o, d_a, d_b, n);
```

  - Note 1: the attribute x isn't the only one of the three structures, it also contains y an z which can be used to perform multidimensional operation or different level of parallelization. Ultimately the three structures are instance of the dim structure.

  - Note 2: We use a if condition because it isn't always possible to divide your array in a round number. In addition to this we compute the number of blocks and thread such that every entry is computed once and once only. Knowing the the __blockSize__ has to be a power of 2, we allocate $$1 + [\frac{n}{256}]$$ blocks of size $$256$$. Where the brackets indicate the rounding operation. Verify yourself with different values of n is all entry are computed.

  So now we got a function running in parallel over an entire array. We need yet to write this in a OOP fashion. So let's see how to do this.

## OOP example.<a name="sec:class"></a>

  In this example we want to write a code that appears to an user as a simple class but will then run different methods on a GPU. Our Code will contain two classes, one exposed to the user and one the will be a purely device class that will run a simulation on the GPU. You can see this as a Class that contains a pointer to multiple simulation objects stored on the device. Every simulation object will then be responsible or running its own simulation.

  So let's first define our device class:

```c++
class ModelGpu{
  public:
    __host__ __device__ void advance();
    __host__ __device__ void setX(int x);
    __host__ __device__ void setN(int n);
    __host__ __device__ int getX();
    __host__ __device__ int getN();
    __host__ __device__ ModelGpu();
    __host__ __device__ void init(int x);

  private:
    int x_;
    int n_;
};
```
  You see that we declared everything to be __\__host\____ and __\__device\____. This is because we compare everything to a single thread computaion on the CPU and had to be able to run the code locally. Not to worry, a method can be both an NVCC will create two version of executable, one when the method is called on the device and one for the CPU.

  Apart from that the class is pretty standard and I leave the exercise to you to figure out the implementation. In this scenario, the advance method only add +1 to the current x. My solution can be found on my github.

  Note: the init method might not seem useful at first glance but I haven't figured out how to call a constructor on the device. I use this method as a initialization method.

  Okay that's great but how do we run this on our device?

  Let us first create a other class that will store the device pointer and run the different simulation:

```c++
class Model{
  public:
    Model(int n);
    ~Model();
    void sim();
    void memcpy_set_data(int* x);
    void memcpy_get_data(int* x);
  private:
    int n_;
    int* d_x;
    ModelGpu* d_models;
};
```

  This class is very simple as well. It stores the device pointers __d_models__, a pointer to the data __d_x__ and the number of simulations __n\___. So what do we need to do in the different methods. In the constructor we allocate the data on the device for every pointer as well as for the data. The sim method will launch a kernel to run our simulations. the memcpy methods are two convenient methods that allows us to set and the the useful data.

  Now we notice that none of the methods are __\__global\____ that because we can create a class on the GPU and this would lead to a device methods trying to call a global method. Which isn't allowed. To work around this we can create __\__global\____ functions outside the class. In this test example, we only need two of those:
  ```c++

    __global__ void sim_gpu_kernel(ModelGpu* d_models, int* x_, int n_);

    __global__ void set_data(ModelGpu* d_models, int* x, int n_);
  ```

  The first function calls the advance methods of our ModelGpu objects. The second one sets the data on the simulation objects. Mainly this calls the init methods of our ModelGpu class. Again I let you figure out the code by yourself and my implementation of it is still on my github.

## Class with Pointers attribute <a name="sec:classptr"></a>

  In this example we will change the state of our simulation to represent a array of all the past explored state. This will be used to store the data and learn a transition function.

  A first intuition would be two create a 2d array stored on the device and that every simulation contains a pointer to one row of the 2d array. It is however very though to implement this. To give you an idea why, imagine you allocate the space for the pointer to pointer using cudaMalloc. Next you want to allocate data for d_x[0] by running cudaMalloc again but you try to access x[0] which is stored on the device leading to an error. There might be some lib out there to do so but I've flatten my 2d array to allocate all in one path. The simulation object then will recieve the address of the first element of their allocated space. There is not much more to discuss here and I leave the implementation to you. Again my implementation of it can be found on my github.

# MPPI implementation<a name="chap:mppi"></a>



  On my CPU *Intel® Core™ i9-9980HK CPU @ 2.40GHz × 16*
