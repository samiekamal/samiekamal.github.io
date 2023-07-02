---
classes: wide
header:
  overlay_image: /images/cog-sci-images/machine_learning.jpg
title: Neural Networks & Machine Learning
toc: true
toc_label: "Overview"

---

<style type="text/css">
  body{
  font-size: 13pt;
}
</style>

## Connectionism:

Connectionism is a theory that posits cognitive processes can be explained through interconnected neural networks. It suggests that the mind can be modeled as a system of nodes, akin to how neural networks operate.

In connectionism, a neural network can be trained to recognize objects in images.

 The network consists of interconnected nodes that process pixel values as inputs and produce outputs indicating the detected objects. Through training, the network learns to identify patterns and features that differentiate various objects, enabling accurate image recognition.

![](/images/cog-sci-images/neural_network.jpg)

## Neural Networks:

Neural networks are computational models inspired by the structure and function of the brain. They consist of interconnected nodes, including input units, output units, and hidden units, through which information flows and computations take place.

Activation Functions: Neural networks employ activation functions to introduce non-linearity into the computations at each node. Common activation functions include sigmoid, ReLU, and tanh, which determine the output of a node based on its input.

Deep Learning: Deep learning refers to neural networks with multiple hidden layers. It enables the learning of hierarchical representations, allowing networks to extract increasingly abstract features from the input data.

Neural networks are used for sentiment analysis, where the goal is to determine the sentiment expressed in a piece of text, such as positive, negative, or neutral. 

A neural network model can be trained on labeled datasets, where texts are annotated with their corresponding sentiments. The network learns to extract meaningful features from the text and make predictions based on the learned patterns, achieving accurate sentiment analysis.

## Unsupervised Learning - Hebbian:

Hebbian learning is a form of unsupervised learning in which connections between nodes in a neural network are strengthened based on their co-activation. The fundamental principle is "neurons that fire together wire together."

### Example: Document Clustering

Hebbian learning can be applied to cluster documents based on their similarity. 

In this example, a network is created where each document is represented as a vector. Through Hebbian learning, documents that frequently co-occur or share similar features will become more strongly connected. The network then forms clusters of related documents, enabling efficient organization and retrieval of textual data.

![](/images/cog-sci-images/unsupervised.png)


## Supervised Learning - Backpropagation:

Backpropagation is a widely used algorithm for training neural networks in supervised learning settings. It involves adjusting the weights of connections in the network based on the error between the predicted outputs and the desired outputs.

### Example: Handwritten Digit Recognition

Backpropagation is commonly used for training neural networks in tasks like handwritten digit recognition. 

A network is trained on a dataset of labeled images of handwritten digits, where each image corresponds to a specific digit (0-9). 

Through backpropagation, the network adjusts its weights to minimize the difference between the predicted digit and the actual digit label. Once trained, the network can accurately recognize handwritten digits based on their pixel values.

![](/images/cog-sci-images/handwritten.png)