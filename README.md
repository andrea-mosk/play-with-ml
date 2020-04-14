# PlayWithML 

### This application allows you to quickly perform two important things:

* **Analyze a dataset**: This section allows you to take a first look at the dataset and discover the main statistics,
                         such as number of instances, missing data, etc. Moreover, you can make more advanced analysis,
                         such as inspecting each feature in details, checking values such as skewness, mean, memory usage,
                         correlation with label, etc.
                         
* **Running a prediction model**: 
                         This section allows you to train and test, on your dataset, \
                         one of the available machine learning models. The available models include XGBoost, Random forests, \
                         Support vector machines, and so on. During the experiment, you can also adjust the experiment parameters \
                         (such as the size of the test set) and the hyperparameters of the model selected (such as the number \
                         of estimators in XGBoost). The dataset will automatically be preprocessed in order to handle missing \
                         data, drop useless rows, scale features and so on.
                         

The interface of the application has been done using [Streamlit](https://www.streamlit.io/). 
