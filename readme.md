## Mamba-DCAU  : State Space Dual Attention Center-sampling U-Net for Hyperspectral Image  Classification

## Abstract

Hyperspectral image classification, as a core research direction in the field of remote sensing, has seen significant performance improvements in recent years due to breakthroughs in deep learning technologies. Although computer vision methods based on V-Mamba and U-Net have demonstrated remarkable results, existing frameworks still face dual challenges: the unidirectional scanning mechanism of the Mamba model limits its ability to jointly model spatial neighborhood features, while the inherent high-dimensional characteristics and large-scale nature of hyperspectral images create significant computational efficiency bottlenecks in traditional whole-image input paradigms. This study introduces a novel hybrid model, Mamba-DCAU, which integrates Mamba, U-Net, and attention mechanisms to improve classification accuracy. The model synergistically optimizes local spectral-spatial feature representation and global semantic segmentation capabilities, offering a unique approach to feature extraction and segmentation. The proposed bidirectional dynamic attention mechanism constructs horizontal and vertical dual-path attention branches through feature map transposition, employing a learnable weight coefficient-based linear interpolation fusion strategy to achieve bidirectional adaptive modeling of neighboring relationships, effectively addressing the local correlation deficiency caused by Mamba's unidirectional scanning mechanism. A center-sampling training method is designed to constrain gradient backpropagation through the center pixels of U-Net outputs, establishing a mapping between local features and global parameters while maintaining the advantages of convolutional kernel weight sharing, enabling end-to-end efficient training. By combining Mamba's global modeling capability with U-Net's multi-scale feature extraction, feature selection is enhanced through attention mechanisms. The model was evaluated on three commonly used datasets: Indian Pines, University of Pavia, and Houston 2013. Experimental results demonstrate that it achieves outstanding classification accuracies of 98.29%, 99.82%, and 99.02%, respectively, outperforming baseline methods in terms of both accuracy and robustness. This study provides an innovative solution to overcome the feature modeling efficiency bottleneck in hyperspectral image classification, and its methodological framework offers universal reference value for related fields.

## Requirements:

- Python 3.7
- PyTorch >= 1.12.1

## Usage:

python main.py

