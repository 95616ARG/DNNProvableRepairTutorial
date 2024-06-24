# Algorithms and Applications for Provable Repair of Deep Neural Networks

Deep neural networks (DNNs) are becoming increasingly important components of software, and are considered the state-of-the-art solution for a number of problems, such as image recognition. However, DNNs are far from infallible, and incorrect behavior of DNNs can have disastrous real-world consequences. In this tutorial, we discuss recent advances in the problem of provable repair of DNNs. Given a trained DNN and a repair specification, provable repair modifies the parameters of the DNN to guarantee that the repaired DNN satisfies the given specification while still ensuring high accuracy. The tutorial will describe algorithms for provable repair that support different DNN architectures as well as various types of repair specifications (pointwise and V-polytope). The tutorial will demonstrate the utility of provable repair using examples from a variety of application domains, including image recognition, natural language processing, and autonomous drone controllers. The attendees will get hands on experience with provable repair tools build using PyTorch, a popular machine learning library.


## Tutorials
- [Pointwise Repair](./tutorial_pointwise_repair.ipynb): Interactive toy example for provable pointwise repair of DNNs.
    - [Case Study: ImageNet Pointwise Repair](./tutorial_imagenet_pointwise_repair.ipynb)
- [Polytope Repair](./tutorial_polytope_repair.ipynb): Interactive toy example for provable polytope repair of DNNs.
    - [Case Study: MNIST Polytope Repair](./tutorial_mnist_polytope_repair.ipynb)
- [SyTorch Overview](./tutorial_sytorch_overview.ipynb): API comparison with Torch.

See [95616ARG/APRNN](https://github.com/95616ARG/APRNN) for more experiment scripts.

## Installation

Create an environment of python 3.9.7, like: 
```
conda create -n pldi24-tutorial python==3.9.7
```

Install dependencies:
- without CUDA:
```
pip install -r requirements.txt
```

- with CUDA, take `cu113` as an example:
```
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu113
```

#### (Optional) Download and Extract Datasets

- MNIST
    ```
    make datasets-mnist
    ```

- ImageNet
    ImageNet requires ImageNet-C and ImageNet validation datasets. Please
    download the [official ImageNet validation set
    (`ILSVRC2012_img_val.tar`)](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5)
    via torrent and place it to `data/ILSVRC2012/ILSVRC2012_img_val.tar`. The
    following command will download ImageNet-A and extract both ImageNet-A and
    Imagenet validation datasets.
    ```
    make datasets-imagenet
    ```

#### (Optional) Setup Gurobi License

Please visit [Gurobi academic
license](https://www.gurobi.com/academia/academic-program-and-licenses) to
generate an "Academic WLS License" (for containers). Aside from the official
instructions, the following steps might be helpful.

- Login to the Gurobi user portal.
- Go to the "License - Request" tab, genearte a "WLS Academic" license if you don't have
  one. If you already have a "WLS Academic" license, you might get an
  "[LICENSES_ACADEMIC_EXISTS] Cannot create academic license as other academic
  licenses already exists" error.
- Go to the "Home" tab, click "Licenses - Open the WLS manager" to open the WLS
  manager.
- In the WLS manager, you should see a license under the "Licenses" tab. Click
  "extend" if it has expired (it might take some time to take effect).
- Go to the "API Keys" tab, click the "CREATE API KEY" button to create a new
  license, download the generated `gurobi.lic` file and place it in
  `/opt/gurobi/gurobi.lic` inside the container.
