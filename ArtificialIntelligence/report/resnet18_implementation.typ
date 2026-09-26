#import "style.typ" : *

#show: paper.with(
  title: "ResNet18 implementation and analysis on \nCIFAR-10",
  authors: ("Andrea Colombo - Artificial Intelligence course project - A.A 2025/2026",),
  abstract: [
    This report presents a comprehensive implementation and analysis of a ResNet18,
    a deep residual network architecture for image classification.
    We implement the network from scratch using PyTorch, train it on the CIFAR-10
    dataset and analyze its performance characteristics.
    This implementation achieves competitive accuracy, showcasing the effectiveness
    of residual network in enabling deep network training.
  ],
)

= Introduction 

Deep convolutional neural networks have revolutionized computer vision, but training
very deep networks has always been extremely challenging due to the degradation
problem: as the network's depth increases, accuracy saturates and then rapidly
degrades. Critically, this degradation is not caused by overfitting but, instead,
by optimization difficulties that prevent deeper networks from learning even simple
identity mappings.

Residual networks (ResNets) were introduced in @he2016deep by He et al in 2015.
These networks address this fundamental issue through an extremely
elegant architectural innovation called _skip connections_. \
A skip connection allows information to bypass layers via an identity mapping.
Instead of forcing each layer to directly learn an underlying mapping $cal(H)(bold(x))$,
ResNets reformulate the problem to learn a residual function
$cal(F)(bold(x)) = cal(H)(bold(x)) - bold(x)$.

This architectural change has very deep implications; if the optimal function is
close to an identity mapping, it's far easier to push the residual term towards
zero than to fit an identity mapping through multiple non-linear functions.
This allows the training of networks that are much deeper than the ones used before
this architecture was invented.

#figure(
  image("images/degradation.png", width: 50%),
  caption: [
    Visualization of the degradation problem through a graph of
    training errors at different depths, image from @he2016deep 
  ]
)

== Motivation and Objectives

This project implements a particular residual network architecture called ResNet18
and evaluates it on CIFAR-10, a widely used benchmark for image classification. \

Our objectives are the following:
- Implement a ResNet18 from scratch using PyTorch and adapt it to the CIFAR-10
  dataset.
- Train the model to achieve competitive accuracy
- Analyze the role of residual connections

The remainder of this document is organized as follows:
- Section 2 provides background on residual learning.
- Section 3 provides information on the used dataset and related challenges. 
- Section 4 contains the implementation details.
- Section 5 presents the training methodology.
- Section 6 analyzes experimental results.
- Section 7 contains the final thoughts about the project

= Background

== Residual Learning Framework

The fundamental innovation in residual networks is the residual block. \
Traditional neural networks learn a _direct_ mapping from input to output. ResNets,
instead, learn the residual (difference) between input and output.

Formally, given an input $bold(x)$, a traditional neural network learns the
following: $ bold(y) = cal(H)(bold(x)) $

A residual network instead learns: $ bold(y) = cal(F)(bold(x), {W_i}) + bold(x) $
<eq:residual>

where $cal(F)(bold(x), {W_i})$ represents the residual mapping learned by a number
of stacked layers, and the identity mapping $bold(x)$ is added via the skip
connection.

This formulation is based on the hypothesis that learning the residual mapping 
$cal(F)(bold(x)) = cal(H)(bold(x)) - bold(x)$ is easier than learning a complete
mapping $cal(H)(bold(x))$. Additionally, if the optimal mapping is close to the
identity mapping, the network can simply drive the residual term towards zero,
which is simpler than fitting the identity mapping through multiple non-linear
layers.

#figure(
  image("images/res_block.png", width: 50%),
  caption: [
    Residual block visual representation
  ]
)

== Related Work

The evolution towards residual networks followed several important developments
in deep learning:

- *Early CNNs*: LeNet and AlexNet demonstrated the power of convolutional
  architectures but were limited to shallow networks.
- *VGG Networks*: Showed that performance could be improved by using deeper
  networks but training became increasingly difficult.
- *The Degradation problem*: it was observed that simply adding more layers led
  to a higher training error, indicating that it was not caused by overfitting
  but rather by optimization difficulties.

= Dataset Description 

The CIFAR-10 dataset @krizhevsky2009learning is one of the most widely adopted benchmark datasets for
evaluating image classification algorithms in computer vision research. It serves
as a testbed to compare different model architectures, training methods and optimization
techniques. \
The dataset consists of 60,000 images of size 32×32 pixels, each one of them belonging
to one of ten mutually exclusive classes.
\
#figure(
  image("images/CIFAR-10.png", width: 70%),
  caption: [
    Example of images from the CIFAR-10 dataset
  ]
)\

== Class Distribution

The dataset is organized into 10 classes that represent common objects or animals.\
The classes are equally present inside the dataset, with each class containing 6,000
images (5,000 for the training set and 1,000 for the test set).

== Dataset Split <sec-dataset_split>

The dataset is pre-divided into training and test sets in the following way:
- *Training set*: 50,000 images (5,000 per class)
- *Test set*: 10,000 images (1,000 per class)

For our purpose, the training set was further divided to create a validation set
for hyperparameter tuning and early stopping, leaving us with the following split:

- *Training subset*: 45,000 images
- *Validation subset*: 5,000 images
- *Test set*: 10,000 images (unchanged)

== Dataset Characteristics 

Despite its relatively small size, the CIFAR-10 dataset poses several challenges
that make it a meaningful benchmark:

- *Low resolution*: The 32×32 pixel size requires the model to learn meaningful
  high level features
- *Intra-class variability*: The objects and animals represented in the pictures
  appear in various different poses, with different lighting and backgrounds,
  this means that the model will need to have strong generalization capabilities.
- *Inter-class similarity*: Some classes share visual similarities between them,
  making it difficult to distinguish between them.

== State Of The Art

Modern architectures are capable of achieving test accuracies exceeding 99% on the
CIFAR-10 dataset using advanced techniques such as vision transformers, extensive
data augmentation and semi-supervised learning.
Standard ResNet architectures generally achieve 92-95% accuracy. 
The goal of this project is to match or possibly exceed this baseline.

= Implementation Details

== Development environment

This project was implemented in PyTorch, a leading deep learning framework that provides
great tooling for building neural networks. \

Some of its advantages are:
- *Dynamic computation graphs* enable easier debugging and experimentation.
- *Strong community support* and extensive documentation. At the time of writing,
  PyTorch is the most used deep learning frameworks amongst researchers.
- *Efficient GPU acceleration* through CUDA integration, making it a great option
  for working with Google CoLab.
- *Built-in dataset support* via torchvision.

== Network architecture

The ResNet18 architecture consists of an initial convolutional layer followed by 
four stages, each composed of two residual blocks.
Each stage increases the number  of channels while reducing spatial dimensions,
allowing the network to learn increasingly abstract features in later layers.

=== Overall architecture

The complete network architecture described above is reported in the following
table.

#figure(
  table(
    columns: 3,
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { rgb("#e8e8e8") } else { white },
    inset: 8pt,
    align: center + horizon,
    [*Layer Name*], [*Output Size*], [*Configuration*],
    [conv1 + bn1], [32×32×64], [3×3, 64, stride=1],
    [res_1], [32×32×64], [\[3×3, 64\] #sym.times 2 #sym.times 2, stride=1],
    [res_2], [16×16×128], [\[3×3, 128\] #sym.times 2 #sym.times 2, stride=2],
    [res_3], [8×8×256], [\[3×3, 256\] #sym.times 2 #sym.times 2, stride=2],
    [res_4], [4×4×512], [\[3×3, 512\] #sym.times 2 #sym.times 2, stride=2],
    [avg_pool], [1×1×512], [4×4 average pooling],
    [linear], [10], [512 → 10],
  ),
  caption: [ResNet18 architecture for CIFAR-10. Each res_X stage contains 2 
           residual blocks. First block in stages 2-4 uses stride=2 for 
           spatial downsampling.]
) <tab:architecture>

#pagebreak()
=== Residual Block Implementation

The residual block is the core of any residual network, the implementation used
in this project uses two convolutional layers with a kernel size of 3 followed by
batch normalization. \
This implementation uses _post activation_, following the architecture proposed
in the first paper.
An alternative to this approach, called _pre activation_, was proposed in
@he2016identity by He et al. \
This technique mainly benefits really deep residual networks so it wasn't used
for this particular project.
What follows is the python code used to implement a residual block in this
project.

#code-block(
  lang: "python",
  caption: "Residual block implementation in python using PyTorch",
  ```python
  class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bnorm_1 = nn.BatchNorm2d(out_channels)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bnorm_2 = nn.BatchNorm2d(out_channels)
        # Each residual block has a skip connection, this is what separates
        # residual networks from the architectures that were previously used.
        # If the dimensions and stride allow it, the skip connection is the
        # identity mapping (used to propagate the block input forward).
        # If the dimensions do not match (or stride != 1) we have to use a
        # projection mapping to match the dimensions
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # The idea of a forward pass is to have a H(x) = F(x) + x, where F(x)
        # is the result of the first and second convolutional layers (residual
        # path) and x is the input that gets propagated via the skip connection.
        out = F.relu(self.bnorm_1(self.conv_1(x)))
        out = self.bnorm_2(self.conv_2(out))
        out += self.skip_connection(x)
        out = F.relu(out)
        return out
  ```
)
#pagebreak()

=== Full Architecture Implementation

The architecture of the full residual network, as described in the previous
sections, is implemented using the following python code:

#code-block(
  lang: "python",
  caption: "Python code for the ResNet18 architecture using PyTorch",
  ```python
  class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.bnorm_1 = nn.BatchNorm2d(64)
        # Residual blocks configuration: after the first layer we use a stride
        # of 2 in order to downsample spatially (since we are modifying the
        # number of channels)
        self.res_1 = self.make_blocks(64,  64,  2, stride=1)
        self.res_2 = self.make_blocks(64,  128, 2, stride=2)
        self.res_3 = self.make_blocks(128, 256, 2, stride=2)
        self.res_4 = self.make_blocks(256, 512, 2, stride=2)
        # Final linear layer that maps to the output classes
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu(self.bnorm_1(self.conv_1(x)))
        out = self.res_1(out)
        out = self.res_2(out)
        out = self.res_3(out)
        out = self.res_4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def make_blocks(self, in_chs, out_chs, n, stride):
        layers = []
        layers.append(ResBlock(in_chs, out_chs, stride))
        for _ in range(1, n):
            layers.append(ResBlock(out_chs, out_chs, 1))
        return nn.Sequential(*layers)
  ```
)

The network defined above contains 11,173,962 trainable parameters.

= Training Methodology <sec-training>

When training deep neural networks, we have to pay attention to make sure the
network does not overfit the dataset.
Some of the choices in the training process were made specifically to reduce the
likelihood of overfitting.

== Data Augmentation <sec-data_augment>

In order to improve the network's generalization capabilities, standard data
augmentation techniques were applied to the dataset, including random horizontal
flipping and random cropping. \
Data augmentation was *not* applied to the test and validation sets.

== Early Stopping

To prevent overfitting and reduce unnecessary computation, early stopping was
applied during the training process. \
Early stopping is a regularization technique used to prevent overfitting by
stopping the training after the model's performance on the validation set stopped
improving for a certain number of epochs (patience). \
Models trained with early stopping tend to have better generalization capabilities.

== Defining a Validation Set

In order to use early stopping, we have to validate the model using some data
and, for obvious reasons, we can't use the training data for this purpose. \
As stated in *@sec-dataset_split*, we use a 90-10 split over the training set
where the 10% will be used for validation during the training process.

== Optimization Strategy

The presented model was trained using the AdamW optimizer @loshchilov2019decoupled,
a variant of the Adam optimizer that brings some key advantages over more 
traditional optimizers like SGD.

- *Adaptive learning rates*: AdamW maintains per-parameter learning rates that
  adjust based on gradient history. This allows different layers of the network
  to learn at different speeds without manual tuning.
- *Robust convergence*: AdamW is less sensitive to hyperparameter values, making
  it a more reliable choice compared to SGD.
- *Better weight decay*: AdamW decouples weight decay from gradient updates. This
  often leads to better generalization which can be helpful on the CIFAR-10's
  small training set.

== Loss Function

We employ Cross-Entropy loss, a standard choice for multi class classification
tasks. \
Given a training example with true class $y$ and predicted probabilities 
$hat(bold(p))$, the loss is defined as follows:

$ cal(L)_"CE" = -log(hat(p)_y) = -log((e^(z_y))/(sum_(j=1)^C e^(z_j))) $

Where $z_j$ are the logits for each $C = 10$ classes.

Cross-entropy is particularly suited for the CIFAR-10 classification task for the
following reasons:

- It measures the dissimilarity between the true 
  distribution (one-hot encoded labels) and the predicted distribution, directly 
  optimizing the model to produce well-calibrated probability estimates.
- When the model is confident but wrong, the loss produces large gradients that
  drive rapid correction.

= Experimental Results

== Training Configuration

The model was trained following the methodology described in @sec-training with
early stopping based on the validation loss. \
As stated in @sec-data_augment, data augmentation techniques were only applied
to the training set.

The following table describes the hyperparameters used for the training process:

#align(center, [
  #table(
    align: center,
    columns: 2,
    [*Parameter*], [*Value*],
    [max epochs], [100],
    [batch size], [128],
    [learning rate], [1e-3],
    [weight decay], [1e-4],
    [patience], [10],
    [min delta], [1e-3]
  )]
)

== Training performance

The training was conducted for 54 epochs before early stopping was triggered.
The model showed strong initial learning, reaching approximately 85% accuracy
within the first 10 epochs. \
After that, the performance continued to improve, with the validation accuracy
reaching its peak of 93.04% at epoch 44.

The following plots show the loss and accuracy curves for the training and validation
sets.

#grid(
  columns: 2,
  figure(
    image("images/train_val_acc_vert.png", width: 100%),
  ),
  figure(
    image("images/train_val_loss_vert.png", width: 100%),
  )
)

== Results on Test Dataset

The trained model achieved a final accuracy of 92.38% against the CIFAR-10 test set.
This value falls in the standard performance baseline for ResNet18 networks,
showing a successful implementation of the architecture.

== Results Analysis

=== Per Class Performance

To better understand the model behavior, we analyzed its performance on each
individual class, using standard analysis tools like the confusion matrix. \
Using the confusion matrix helps us understand which classes are the most 
difficult to classify for our model.

#figure(
  image("images/confusion_matrix.png", width: 70%)
)

The model achieved the following per-class accuracies:

#figure(
  image("images/per_class_acc.png", width: 70%)
)

We can see how the best performing classes are the ones that present distinctive
visual features even at such a low resolution.
The ones that performed the worst are the ones with a high degree of inter-class
similarities.
In particular, the most frequent confusion occurs between cats and dogs, this is
due to the fact that, at such low resolutions, these classes have similar poses,
body shapes and textures that make classification difficult.

#figure(
  image("images/confused_classes.png", width: 100%),
  caption: "Examples of wrongly classified images in the 3 worst performing classes"
)

Despite visual similarities, the model was able to correctly distinguish between
automobiles and trucks with minimal confusion. This shows that the model was able
to successfully learn meaningful features.

= Conclusions

The implementation described in this report achieved 92.38% test accuracy on
the CIFAR-10 dataset, meeting the expected baseline for the ResNet18 architecture. \
The small gap between validation accuracy (93.04%) and test accuracy (92.38%) shows
that the model was able to learn meaningful features instead of simply memorizing data.

== Key Takeaways

- *Residual connections worked*: The skip connections enabled effective training
  of the deep network by allowing the gradients to flow directly. This architectural choice
  successfully addressed the degradation problem found in earlier architectures.
- *Regularization was effective*: The combination of data augmentation, early stopping
  and weight decay prevented severe overfitting.
- *Performance between classes*: as it was expected, the classes with the best
  accuracy were the ones with distinctive visual features, while the ones with lower
  accuracy were the ones that presented more inter-class similarities.

== Limitations

Although the implementation met the expected results, it's not without flaws, in this
section we explore some of the details of this model that could be improved.

- *Minor Overfitting*: The gap between training accuracy and test accuracy of the
  saved model of 2.76% indicates that the model memorized some patterns of the training data. \
  Although the gap is relatively small and acceptable, stronger regularization
  techniques, more advanced data augmentation and learning rate scheduling could
  possibly reduce this gap and improve the final performance of the model.

- *Single dataset performance*: The model's architecture was trained only on the CIFAR-10 dataset,
  therefore performance on other datasets and transfer learning capabilities were not explored.

== Possible Improvements

The analysis made on the model's performance showed some promising directions that could
improve upon this work.

- *Advanced augmentation*: more modern data augmentation techniques such as _AutoAugment_ could
  potentially improve the generalization capabilities of the model and result in better accuracy on test data.
- *Ablation studies*: Removing components and measuring the impact would give us a better
  understanding on which architectural choices matter more.
- *Transfer learning experiments*: Pre-training the model on other similar datasets
  (e.g. ImageNet) and then fine tuning the model on CIFAR-10 would enable us to assess
  the performance tradeoffs and whether or not transfer learning is a viable option.

#bibliography("references.bib", style: "ieee")
