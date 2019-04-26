# A-Machine-Learning-based-Cache-Management-Policy

Detailed report could be found at _[A Machine Learning based Cache Management Policy](https://github.com/ShawnLYU/A-Machine-Learning-based-Cache-Management-Policy/blob/master/report/report.pdf)_.

## Description
This is a Storage System project to improve cache replacement policies @ University of Toronto.





<p align="center">
  <img src='https://github.com/ShawnLYU/A-Machine-Learning-based-Cache-Management-Policy/blob/master/report/proj_graphs/NN.png'/>
</p>

## Getting Started

These repo will get you a copy of both our models. Sample data is provided here for to get your hands dirty.


### Prerequisites

What things you need to install the software and how to install them

- Python 3.7.0
- [Pytorch](https://pytorch.org/)
- Numpy, Sklearn, Pandas





## Running the tests

To run the model, you can cd any of the baseline model or the hybrid model, and with command:

```
allennlp train config.json -s res --include-package packages
```

## Notes

This package would save the model after each epoch and all of the metrics during the training process.

## Contributing

This project exists thanks to all the people who contribute. 

[Shawn](https://github.com/ShawnLYU)    
[Yilin](https://github.com/yilinhan)

## License

This project is licensed under the [MIT](LICENSE) License.

## Acknowledgments

This project borrowed ideas from Towards a ML based Cache Management Policy (maharshi Trivedi, Jay Patel, and Shehbaz Jaffer)
