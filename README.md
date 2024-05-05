# Financial Fraud Detection

## Table of Contents
- [The Context](#the-context)
- [About the Project](#about-the-project)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)

## The Context

This project served as the culmination of my MSC in Computer Science with a specialization in Data Science, representing a dedicated effort to address the critical need for robust fraud detection algorithms in today's financial landscape. Drawing upon a comprehensive dataset from a Kaggle competition, the thesis rigorously evaluated the effectiveness of an ensemble anomaly detection algorithm comprising Isolation Forest (IF), Local Outlier Factor (LOF), and One-Class Support Vector Machine (OCSVM). Through meticulous data preprocessing, model training, and evaluation, the study aimed to enhance fraud detection accuracy, thereby safeguarding financial institutions and their clientele from the detrimental impacts of fraudulent behavior. By integrating ethical, legal, social, security, and professional considerations into the research framework, the project emphasized privacy protection and transparency, underscoring the importance of responsible AI deployment in finance.

## About the Project

The approach involved training base models and optimizing them through hyperparameter tuning using grid search with cross-validation, specifically focusing on maximizing recall. Following this, the best-performing model was selected and individually trained for each algorithm: Isolation Forest (IF), Local Outlier Factor (LOF), and One-Class Support Vector Machine (OCSVM). Subsequently, an ensemble model was constructed by combining these individual algorithms, exploring techniques such as stacking, voting, and averaging. Finally, Explainable Artificial Intelligence (XAI) techniques were applied to each model to enhance transparency and interpretability. Despite moderate overall performance, the LOF model emerged as the preferred choice due to its emphasis on minimizing false negatives, crucial for accurate fraud detection. Looking forward, the project highlighted the importance of enhanced computing capabilities and comprehensive data preprocessing to further improve key performance metrics and ultimately bolster financial security within the banking sector.

## Project Structure

Financial_Fraud_Detection/<br>
│<br>
├── data/<br>
│   ├── raw    # Raw data files downloaded from Kaggel<br>
│   └── processed             # Cleaned data files created by the 1_eda_preprocessing.ipynb file<br>
│<br>
├── notebooks/<br>
│   ├── models_selection/     # Main notebook for data preprocessing, analysis, and answering tasks<br>
│   │    ├── ensemble.ipynb    # Create ensemble models from individual anomaly decetion models and evaluate<br>
│   │    ├── if.ipynb   # Training and evaluating Isolation Forest models<br>
│   │    ├── lof.ipynb   # Training and evaluating Local Factor Outlier models<br>
│   │    └── ocsvm.ipynb   # Training and evaluating One-Class Support Vector Machine models<br>
│   │ <br>
│   ├── scratch/ # Made for quick tests on preprocessing and modeling<br>
│   │    ├── auroc.ipynb   # Testing auroc scoring for the models<br>
│   │    ├── class_balancing.ipynb # Class balaning techniques like SMOTE<br>
│   │    └── improve_data.ipynb # Trying different data preprocessing techniques<br>
│   │ <br>
│   ├── trained_models/ # Models trained from file 2_train_individual_models.ipynb will be saved here<br>
│   │ <br>
│   ├── 1_eda_preprocessing.ipynb # Data Exploration and Preprocessing<br>
│   ├── 2_train_individual_models.ipynb  # Trained and save individual anomaly detection models (IF, LOF and OCSVM)<br>
│   ├── 3_evaluate_models.ipynb   # Compare individual anomaly detection models and the ensemble model<br>
│   └── 4_xai.ipynb        # Apply XAI to the models<br>
│<br>
├── src/<br>
│   ├── __ init__.py <br>
│   └── functions.py       # Functions created for the project<br>
│<br>
├── LICENSE<br>
├── requirements.txt # Requirements for the project<br>
└── README.md                              # You are here<br>

## How to Use

1. Clone this repository to your local machine.

   ```bash
   git clone https://github.com/ndressler/Financial_Fraud_Detection/tree/main
   ```
   
2. Navigate to the project directory:

   ```bash
   cd Financial_Fraud_Detection
   ```

3. Install the required dependencies. It's recommended to use a virtual environment:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the data files from Kaggel and place them in the `raw´ folder in their csv format. 

[Kaggel IEEE Fraud Detection Competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

5. Run the file in the order:
   
`1_eda_preprocessing.ipynb`<br>
`2_train_individual_models.ipynb`<br>
`3_evaluate_models.ipynb`<br>
`4_xai.ipynb`<br>

## Contributing

Contributions to this project are welcome! If you have any suggestions, enhancements, or bug fixes, feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
