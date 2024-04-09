## Face Mask Detection using Deep Learning

This project aims to detect whether a person is wearing a face mask or not using deep learning techniques. The model is trained on a dataset containing images of people with and without face masks.

## Dataset

The dataset used for training consists of two classes:
- **With_Mask**: Images of people wearing face masks.
- **Without_Mask**: Images of people without face masks.

The dataset used in this project is not provided due to size limitations, but similar datasets can be found online or collected through various sources.

## Installation

To run the code in this repository, you will need the following dependencies:
- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn

You can install these dependencies using pip:

```bash
pip install tensorflow numpy pandas matplotlib seaborn
```

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/alihassanml/face-mask-detection.git
```

2. Navigate to the project directory:

```bash
cd face-mask-detection
```

3. Prepare your dataset:
   - Create two folders: `with_mask` and `without_mask`.
   - Place images of people with face masks in the `with_mask` folder and images of people without face masks in the `without_mask` folder.

4. Train the model:
   - Run the `train.py` script to train the model on your dataset.

5. Test the model:
   - Run the `test.py` script to evaluate the performance of the trained model on test data.

## Model Architecture

The model architecture used in this project consists of a convolutional neural network (CNN) designed to extract features from the input images followed by fully connected layers for classification. The final layer uses softmax activation to output probabilities for each class.

## Results

After training the model, the performance metrics such as accuracy, precision, recall, and F1-score are calculated to evaluate the model's effectiveness in detecting face masks.

## Contributors

- Ali Hassan

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README according to your project specifics and requirements.
