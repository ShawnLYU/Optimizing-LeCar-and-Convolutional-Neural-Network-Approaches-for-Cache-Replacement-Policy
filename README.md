# A-Machine-Learning-based-Cache-Management-Policy

Detailed report could be found at _[A Machine Learning based Cache Management Policy](https://github.com/ShawnLYU/A-Machine-Learning-based-Cache-Management-Policy/blob/master/report/csc2233.pdf)_.

## Description
Improving system performance with caching replacement technique is always one of the hottest topics in systematic research. In this project, we proposed a new cache eviction algorithm, namely, LeCar\_Opt, which outperforms ARC when the cache size is small. Furthermore, machine learning models are used to learn the cache replacement patterns on real-world workload. We implemented, tuned and analyzed the Multilayer Perceptron, Convolutional Neural Network, and Decision Tree on the OPT solutions. It is concluded that machine learning models are capable of learning OPT patterns with adequate hyperparameter tuning.




<p align="center">
  <img src='https://github.com/ShawnLYU/A-Machine-Learning-based-Cache-Management-Policy/blob/master/report/proj_graphs/NN.png'/>
</p>

## Getting Started

These repo will get you a copy of both our models. 

### Prerequisites

What things you need to install the software and how to install them

- Python 3.7.0
- [Pytorch](https://pytorch.org/)
- Numpy, Sklearn, Pandas





## Running the tests

In Neural networks, there are two pre-trained models: MLP and CNN. In each folder, there is a Jupyter Notebook where you could load the model and make predictions.


## Contributing

This project exists thanks to all the people who contribute. 

[Shawn](https://github.com/ShawnLYU)    
[Yilin](https://github.com/yilinhan)

## License

This project is licensed under the [MIT](LICENSE) License.

## Acknowledgments

This project borrowed ideas from Towards a ML based Cache Management Policy (maharshi Trivedi, Jay Patel, and Shehbaz Jaffer)
