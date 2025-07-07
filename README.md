[Shared Bike Demand Forecasting.pdf](https://github.com/user-attachments/files/21108672/Shared.Bike.Demand.Forecasting.pdf)

# Bike Sharing Demand Forecasting

This project develops a machine learning model to forecast the hourly rental demand for shared bikes in Washington, D.C., using data from the "Bike Sharing Demand".

### Data Preprocessing & Feature Engineering

Data Cleaning: The data was loaded using Pandas, and the datetime column was parsed to extract time-based features such as year, month, hour, and dayofweek.



Feature Selection: To avoid multicollinearity, features like atemp (highly correlated with temp) and workingday (similar to holiday) were excluded from the model training.


Outlier Handling: The data distribution was visualized using boxplots, and a decision was made to handle outliers using the IQR method.


Target Transformation: To optimize for the competition's evaluation metric, Root Mean Squared Logarithmic Error (RMSLE), the target variable count was log-transformed using np.log1p before training.

### Modeling & Results

Model Selection: A Gradient Boosting Regressor was used, which combines multiple decision trees sequentially to create a powerful predictive model.



Hyperparameters: For reproducibility, the model was trained with specific hyperparameters, including n_estimators=2000, learning_rate=0.05, and max_depth=5.


Performance: The trained model achieved a high R-squared score of approximately 97.9% on the training data and 95.3% on the validation data.

### Future Work

Future improvements could involve optimizing model performance through hyperparameter tuning or applying other regression models like Random Forest for further enhancements.
