# Movie-Review-Sentiment-Analysis-using-NLP
A Natural Language Processing (NLP) project that classifies movie reviews as positive or negative using machine learning techniques. This project demonstrates text preprocessing, feature extraction with TF-IDF, and sentiment classification using multiple algorithms.

## Project Overview:
This project implements binary sentiment analysis on movie reviews using various machine learning algorithms.The system processes raw text data, extracts meaningful features using TF-IDF vectorization, and predicts whether a review expresses positive or negative sentiment with high accuracy.

## Objectives
- Build an accurate binary text classifier for sentiment analysis.
- Compare performance across multiple machine learning algorithms.
- Extract and analyze the most inflential words for sentiment production.
- Create an interpretable model that explains its predictions.
- Save and deploy the trained model for future predictions.

### Technologies and Libraries
Core Technologies
- Python 3.15
- Jupyter Notebook

Machine Learning and Data Processing
- Scikit-learn - Machine learning algorithms, preprocessing and model evaluation.
- Pandas - Data manipulation and analysis
- Numpy - Numerical computation.

Visualizations
- Matplotlib - Create plots and charts.
- Seaborn - Statistical data visualizations.

Model Persistence:
- joblib: Saving and loading trained models.
## Dataset
The project uses a movie revirew dataset containing:
- Movie reviews in text format.
- Binary sentiment labels( Positive=1, Negative=0)
- Multiple reviews covering various genres and opinions.
  
### Project Workflow
1. Data Preprocessing:
   # Text cleaning operations performed:
   - Remove special characters and punctuation.
   - Remove URLs and HTML tags.
   - Convert text to lowercase.
   - Tokenization.
   - Remove stop words
   - Handle missing or null  values.
2. Feature extraction (TF-IDF)
   - Convert text to numerical features
   Why TF-IDF?
- Captures word importance across documets.
- Reduces the weight of common words
- Highlights unique and meaningful terms.
- Better than simple word counts for text classification.
3. Model training and Comparison:
  Multiple algorithms tested:
  - Logistic Regression - Linear classifier with interpretable coefficients.
  - Random Forest - Ensemble method with good generalization.
  - Naives Bayes - Probabilistic classifier, fast training.
  - Support Vector Machine (SVM) - Maximum margin classifier.
  Each model was :
  - Trained on preprocessed TF-IDF features.
  - Evaluated on a held-out test set.
  - Compared using accuracy, precision, recall and F1 score
4. Model Evaluation
Comprehensive evaluation metrics
# Metrics calculated:
- Accuracy: Overall correctness.
- Precision: Positive prediction reliability'
- Recall: Ability to find all positive cases.
- F1- Score: Harmonic mean of precision and recall
- Confusion Matrix: True/False Positives and Negatives
5.  Feature Importance Analysis
  Extracted the most influential words using Logistic Regression coefficients
6. Model Persistence
7. Results
8. Model performance comparison.
9. Feature Importance Insights:
-  Top Positive Sentiment Words: Words with the highest positive coefficients indicate strong positive sentiment. These words frequently appear in positive reviews and has strong predictive power.
-  Top Negative Sentiment Words: Words with the most negative coefficients indicate strong negative sentiment. These words are reliable indicators of unfavourable reviews.

## Key Features
1. Comprehensive Text preprocessing:
- Multi step cleaning pipeline
- Robust handling of special characters and noise
- Preserves semantic handling while reducing dimensionality.

2. TF-IDF Feature Engineering:
- Captures word importance across corpus.
- Configurable feature limits (max features, parameters)
- Balances vocabulary size with model performance

3. Multiple Model Comparison:
- Side by side evaluation of algorithms
- Detailed performance metrics.
- Helps select the best model for deployment.

4. Model Interpretability:
- Extract and visualize important features
- Understand which words drive predictions
- Coefficient based feature ranking.
- 
5. Production-Ready Deployment:
- Save and load trained model.
- Simple prediction interface.
- Confidence scores for predictions.

6. Visualization Tools:
- Feature importance bar charts.
- Confusion matrix heatmaps
- Model performance comparisons.

## What I learned
Natural Language Processing:
- Text preprocessing techniques.
- Feature extraction with TF-IDF
- Handling high dimensional sparse data.

Machine Learning:
- Binary classification algorithms
- Model evaluation metrics.
- Cross validation and generalization.
- Feature importance analysis.

Software Engineering:
- Jupyter Notebook workflow.
- Model persistence with joblib.
- Code organization and documentation.

Data Science Skills:
- Exploratory data analysis
- Performance metrics interpretation.
- Model comparison and selection.
- Results visualisation.

Future Enhancements:
- Deep Learning Models: Implement LSTM,GRU, or BERT for improved accuracy.
- Multi-class Classification: Extend to neutral sentiment or rating scales (1-5 stars)
- Web Application: Build Flask/Streamlit interface for real time predictions.
- REST API: Create API endpoint for model serving.
- Real time analysis: Process streaming review data.
- Cross-domain Transfer: Test on product reviews, tweets, etc
- Hyperparamaetr tuning: Grid search for optimal parameters.
- Ensemble Methods: Combine multiple models for better predictions
- Deployment: Deploy to cloud (AWS,Azure or GCP).
  
Known Issues or Limitations:
- Model trained on movie reviews may not generalize perfectly to other domains.
- TF-IDF doesn't capture word order or context.
- Struggles with sarcasm and nuanced language.
- Limited to binary classification.
- Requires consistent preprocessing for new data.

### Author
Your Name
- LinkedIn : Okparaji Wisdom
- Email: wisdomokparaji@gmail.com

### Project Status
Complete - Core fuctionality implemented and tested.
- Data preprocessing pipeline
- Model training and evaluation.
- Feature importance analysis.
- Model persistence.
- Basic prediction interface

## Contributing:
Contributions, issues and feature requests are welcome! 
If found this project helpful, please star this repository
Questions? Feel free to reach out or open an issue.
