# Movie Recommender System

A hybrid **movie recommendation engine** built using **Matrix Factorization (SVD)** and **Deep Neural Networks (DNN)**.  

---

## Overview
This project explores both collaborative filtering and deep learning methods for personalized movie recommendations.  
It implements:
- **Time-aware per-user data splitting** to mimic real-world recommendation scenarios  
- **Matrix Factorization (SVD)** for collaborative filtering  
- **Deep Neural Network (DNN)** using user, movie, and content embeddings  
- Evaluation using **RMSE, MAE, and Recall@10**

The goal is to understand and compare classical and neural approaches to recommendation systems.

---

## Results

| Model | RMSE | MAE | Recall@10 |
|--------|------|-----|-----------|
| SVD | 0.96 | 0.75 | — |
| DNN | 0.88 | 0.67 | 0.68 |

*Lower RMSE and MAE mean better prediction accuracy; higher Recall@10 indicates better recommendation ranking.*  

These results show that while SVD gives a solid baseline, the DNN model achieves higher accuracy and improved recommendation quality.

---

## Dataset
The project uses the **MovieLens dataset** (ml-latest-small.zip) by GroupLens Research.  
You can download it from:  
[https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

The dataset includes user IDs, movie IDs, ratings, timestamps, and genres.  

---

## Requirements and Setup

To run the notebook, you’ll need Python 3.8+ and the libraries listed below.  
You can install everything using:

```bash
pip install -r requirements.txt
```
### Required Libraries
- pandas → data handling (MovieLens dataset loading & manipulation)
- numpy → matrix operations
- scikit-learn → preprocessing, TF-IDF, scaling, metrics, etc.
- scipy → SVD (via svds)  
- tensorflow/keras → deep neural network model

---

## How to Run the Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/Rohmaa-f/Movie-recommender-system.git
   ```
2. **Navigate to the project directory**
   ```bash
   cd movie-recommender-system
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Open the Jupyter notebook**
    ```bash
   jupyter notebook Movie_Recommender_System_using_SVD_and_DNN.ipynb.ipynb
   ```
5. **Run all cells sequentially**


## Key Learnings

- Implemented both **collaborative filtering** and **neural recommendation models** from scratch.    
- Interpreted **RMSE**, **MAE**, and **Recall@10** to evaluate rating accuracy and ranking quality.  
- Gained experience with **machine learning pipelines**, **feature embeddings**, and **model evaluation**.  
- Strengthened understanding of **data preprocessing** and **model performance interpretation**.


