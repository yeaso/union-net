# union-net
Union-net: Lightweight deep neural network model suitable for small data sets

Jingyi Zhou 1

1 Institute of Scientific and Technical Research on Archives, Beijing, 100050, China 

Abstract
Traditional deep learning models prefer large data sets, and in reality small data sets are easier to obtain. It is more practical to build models suitable for small data sets. Based on CNN, this paper proposes the concept of union convolution to build a deep learning model Union-net that is suitable for small data sets. The Union-net has small model size and superior performance. In this paper, the model is tested based on multiple commonly used data sets. Experimental results show that Union-net outperforms most models when dealing with small datasets, and Union-net outperforms other models when dealing with complex classification tasks or dealing with few-shot datasets. 

The codes are the open source codes of the article "Union-net: Lightweight deep neural network model suitable for small data sets", which are only used for testing and research.
Our test environment: python3.6 + keras2,3,1 + opencv-python 4.5.5.62.
We conducted image classification experiments using our model on the following data sets: CIFAR-10 (10 categories), CIFAR-100 (100 categories), MNIST(10 categories), Fashion-MNIST(10 categories) and 17-Flowers(17 categories) ) .

Cite this article
Zhou, J., He, Q., Cheng, G. et al. Union-net: lightweight deep neural network model suitable for small data sets. J Supercomput (2022). https://doi.org/10.1007/s11227-022-04963-w
