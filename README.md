# Sentiment Analysis of Women's Clothing E-Commerce Reviews


## 📋 Project Overview

This repository contains a comprehensive machine learning project focused on sentiment analysis of customer reviews from a women's clothing e-commerce platform. The project implements and compares three different neural network architectures: **Recurrent Neural Networks (RNN)**, **Convolutional Neural Networks (CNN)**, and **Feedforward Neural Networks (FNN)** to classify customer sentiment.

### 🎯 Objective

The primary goal is to automate sentiment classification of customer reviews, which can provide valuable insights for businesses to understand customer satisfaction at scale and make data-driven decisions to improve their products and services.

## 👥 Team Information

**Group ID:** ML1_T2220

| Student ID | Name | Specialization | Contribution |
|------------|------|----------------|--------------|
| 1211309776 | Jennifer Lo Foh Wei | Data Science | CNN Implementation |
| 1191302190 | Amin Ahmed | Data Science | RNN Implementation |
| 1171103208 | Obai Ali | Data Science | FNN Implementation |

## 📊 Dataset

The project uses the **Women's Clothing E-Commerce Reviews** dataset containing:
- **23,486 customer reviews**
- Multiple attributes including review text, ratings, recommendations, and product categories
- Rich customer feedback data with various demographic and product information

### Dataset Features:
- `Review_Text`: Customer review content
- `Rating`: Product rating (1-5 stars)
- `Recommended_IND`: Binary recommendation indicator
- `Age`: Customer age
- `Title`: Review title
- `Positive_Feedback_Count`: Number of positive feedback received
- `Division_Name`, `Department_Name`, `Class_Name`: Product categorization

## 🏗️ Project Structure

```
ML1_T2220_Assignment/
├── ML1_T2220_Codes/
│   ├── Project.ipynb                    # Main Jupyter notebook with all implementations
│   ├── README.md                        # Basic project description
│   ├── Womens Clothing E-Commerce Reviews.csv  # Dataset
│   ├── logo-MMU.png                     # University logo
│   ├── glove/
│   │   └── glove.6B.100d.txt           # Pre-trained GloVe embeddings (100d)
│   └── .ipynb_checkpoints/             # Jupyter checkpoint files
├── Sentiment Analysis Project - Slides.pptx  # Project presentation
└── Comparative Analysis - Report.pdf    # Detailed analysis report
```

## 🔧 Technical Implementation

### Data Preprocessing Pipeline

1. **Text Cleaning**
   - Removal of punctuation and special characters
   - Conversion to lowercase
   - Tokenization using NLTK

2. **Text Processing**
   - Stop word removal
   - Lemmatization for word normalization
   - Removal of non-English words using PyEnchant

3. **Feature Engineering**
   - Word embedding using pre-trained GloVe vectors (100-dimensional)
   - Sequence padding for consistent input length
   - Label encoding for target variables

### Model Architectures

#### 1. Recurrent Neural Network (RNN)
- **Architecture**: LSTM-based sequential model
- **Purpose**: Capture temporal dependencies in review text
- **Implementation**: Bidirectional LSTM layers with dropout regularization

#### 2. Convolutional Neural Network (CNN)
- **Architecture**: 1D CNN with multiple filter sizes
- **Purpose**: Extract local features and patterns from text
- **Implementation**: Multiple convolutional layers with max pooling and dense layers

#### 3. Feedforward Neural Network (FNN)
- **Architecture**: Multi-layer perceptron
- **Purpose**: Learn complex non-linear relationships
- **Implementation**: Dense layers with ReLU activation and hyperparameter optimization

### Hyperparameter Optimization

The project implements hyperparameter tuning using:
- **Hyperopt library** for Bayesian optimization
- **Tree-structured Parzen Estimator (TPE)** algorithm
- Optimization of learning rates, layer sizes, and architectural parameters

## 📈 Model Performance

All models are evaluated using standard classification metrics:
- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall

### Comparative Analysis
The project includes comprehensive model comparison with:
- Training and validation accuracy/loss curves
- Performance metrics visualization
- Statistical significance testing

## 🛠️ Technologies and Libraries

### Core Libraries
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **NLTK**: Natural language processing
- **pandas**: Data manipulation
- **NumPy**: Numerical computing

### Visualization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization
- **WordCloud**: Text visualization

### Additional Tools
- **PyEnchant**: Spell checking and language detection
- **Hyperopt**: Hyperparameter optimization
- **imbalanced-learn**: Handling imbalanced datasets

## 🚀 Getting Started

### Prerequisites

```bash
# Install required packages
pip install tensorflow
pip install scikit-learn
pip install nltk
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install wordcloud
pip install pyenchant
pip install hyperopt
pip install imbalanced-learn
```

### NLTK Data Downloads

```python
import nltk
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Running the Project

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML1_T2220_Assignment
   ```

2. **Navigate to the code directory**
   ```bash
   cd ML1_T2220_Codes
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Project.ipynb
   ```

4. **Run all cells** to execute the complete pipeline from data preprocessing to model evaluation

### For Google Colab Users

If running on Google Colab, uncomment and run the installation cells at the beginning of the notebook:

```python
!apt install -qq enchant
!pip install keras-tuner
!pip install pyenchant
```

## 📊 Results and Insights

The project demonstrates:
- **Effective preprocessing** techniques for e-commerce review data
- **Comparative analysis** of different neural network architectures
- **Performance optimization** through hyperparameter tuning
- **Practical applications** for business intelligence and customer insights

### Key Findings
- Analysis of model performance across different architectures
- Insights into the effectiveness of different approaches for sentiment analysis
- Recommendations for deployment in real-world e-commerce scenarios

## 📚 Documentation

- **Project.ipynb**: Complete implementation with detailed explanations
- **Comparative Analysis Report**: In-depth analysis and results discussion
- **Presentation Slides**: Project overview and key findings

## 🤝 Contributing

This project was developed as part of an academic assignment. For questions or suggestions, please contact the team members listed above.

## 📄 License

This project is developed for educational purposes as part of the Machine Learning course at Multimedia University (MMU).

## 🙏 Acknowledgments

- **Multimedia University (MMU)** for providing the academic framework
- **GloVe team** for pre-trained word embeddings
- **Open-source community** for the various libraries and tools used
- **Dataset contributors** for providing the e-commerce review data

---

*This project demonstrates the application of deep learning techniques to real-world business problems, specifically in the domain of customer sentiment analysis for e-commerce platforms.* 
