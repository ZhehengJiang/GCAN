# GCAN
This is a PyTorch implementation of our paper: [Graph-context Attention Networks for Size-varied Deep Graph Matching ](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_Graph-Context_Attention_Networks_for_Size-Varied_Deep_Graph_Matching_CVPR_2022_paper.pdf) 
## Requirements

* **[PyTorch](https://pytorch.org/get-started/locally/)** (>=1.6.0)
* **[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)** (>=1.6.0)
* **[Gurobi Python](https://www.gurobi.com/documentation/9.5/quickstart_linux/cs_using_pip_to_install_gr.html)** (>=9.1.2)
* **[networkx](https://pypi.org/project/networkx/)** (>=2.6.3)

## Datasets:
   1. Pascal VOC Keypoint:
     * Download and tar [VOC2011 keypoints](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html), and the path looks like: ``./data/PascalVOC/VOC2011``.
     * Download and tar [Berkeley annotation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz), and the path looks like: ``./data/PascalVOC/annotations``.
     * The train/test split of Pascal VOC Keypoint is available in: ``./data/PascalVOC/voc2011_pairs.npz``.
   2. Willow Object Class dataset:
     * Download and unzip [Willow ObjectClass dataset](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip), and the path looks like: ``./data/WILLOW-ObjectClass``.

## Experiment:
Run training and evaluation on Pascal VOC Keypoint for Size-equal graph matching problem:
``python train_eval.py --cfg ./experiments/GCAN_voc.yaml``
Run training and evaluation on Pascal VOC Keypoint for Size-equal graph matching problem:
``python train_eval.py --cfg ./experiments/GCAN_voc_varied_size.yaml``
Unsupervised training code is coming soon


## Citation
```text
@InProceedings{Jiang_2022_CVPR,
    author    = {Jiang, Zheheng and Rahmani, Hossein and Angelov, Plamen and Black, Sue and Williams, Bryan M.},
    title     = {Graph-Context Attention Networks for Size-Varied Deep Graph Matching},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {2343-2352}
}
