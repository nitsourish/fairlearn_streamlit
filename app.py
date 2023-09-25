import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as p
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from fairlearn.datasets import fetch_adult
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio
from fairlearn.reductions import DemographicParity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import pickle as pkl
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate
from matplotlib import pyplot as plt
from fairlearn.metrics import false_positive_rate, true_positive_rate,count
import lightgbm as lgb
import pandas as pd
# import seaborn as sns

def getClassifier(classifier):
    if classifier == 'SVM':
        c = st.sidebar.slider(label='Choose value of C' , min_value=0.0001, max_value=10.0)
        model = SVC(C=c)
    elif classifier == 'KNN':
        neighbors = st.sidebar.slider(label='Choose Number of Neighbors',min_value=1,max_value=20)
        model = KNeighborsClassifier(n_neighbors = neighbors)
    elif classifier == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 4, 10)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)    
        model = RandomForestClassifier(max_depth = max_depth , n_estimators= n_estimators,random_state= 1)
    elif classifier == 'Light GBM':
        max_depth = st.sidebar.slider('max_depth', 2, 7)
        n_estimators = st.sidebar.slider('n_estimators', 50, 200)    
        model = lgb.LGBMClassifier(max_depth = max_depth , n_estimators= n_estimators,random_state= 1)    
    else:
        model = LogisticRegression()    
    return model
sideBar = st.sidebar
sensitive_var = sideBar.selectbox('Which Classifier do you want to use?',('sex' ,'race','both'))
def getDataset():
    data = fetch_adult(as_frame=True)
    df = data.data
    data.target.replace({ "<=50K": 0, ">50K": 1 }, inplace=True)
    if sensitive_var == 'sex':
        sensitive = df['sex']
    elif sensitive_var == 'race':
        sensitive = df['race']
    else:
        sensitive = df[['sex','race']]
    train_X,test_X,train_A,test_A,train_y, test_y = train_test_split(df,sensitive,data.target,test_size=0.3,random_state=100,stratify=data.target)
    train_X = train_X.reset_index(drop=True)
    test_X = test_X.reset_index(drop=True)
    train_A = train_A.reset_index(drop=True)
    test_A = test_A.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
    return train_X,test_X,train_A,test_A,train_y, test_y


classifier = sideBar.selectbox('Which Classifier do you want to use?',('SVM' , 'KNN' , 'Random Forest','Light GBM','Logistic Regression'))
fairlearn = sideBar.checkbox('Do you want to use Fairlearn?',(True))
def get_training_pipeline(copy=True):
    numeric_preprocess = Pipeline(steps=[('simple_impute',SimpleImputer()),('scaling', StandardScaler())])
    categorical_preprocess = Pipeline(steps=[('impute',SimpleImputer(strategy='most_frequent')),('oneHot', OneHotEncoder())])
    preprocess = ColumnTransformer(transformers=[('num_preprocess',numeric_preprocess, selector(dtype_exclude='category')),('categorical',categorical_preprocess, selector(dtype_include='category'))])
    model = getClassifier(classifier)
    training_pipeline = Pipeline(steps=[('preprocess',preprocess),('model',model)])
    return training_pipeline


class model_training(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,sensitive = None,fairlearn=False):
        self.sensitive = sensitive
    #create a fit-transform method for numeric and categorical features using Pipeline and ColumnTransformer
    # for numeric features simple impute and scaling and for categorical features impute with most_frequent and onehot encoding
    
    #TypeError: All intermediate steps should be transformers and implement fit and transform or be the string 'passthrough' '' (type ) doesn't
    #fulfill this requirement
    #https://stackoverflow.com/questions/57528350/typeerror-all-intermediate-steps-should-be-transformers-and-implement-fit-and
    
    def fit(self,X,Y):
        training_pipeline = get_training_pipeline()
        if not fairlearn:
            training_pipeline.fit(X,Y)
            return training_pipeline
        else:
            threshold_optimizer = ThresholdOptimizer(estimator=training_pipeline,constraints="demographic_parity",prefit=False)
            threshold_optimizer.fit(train_X,train_y,sensitive_features=self.sensitive)
            return threshold_optimizer
        
# Title
st.title("Fairlearn in Action - Adult Income Dataset to predict income above 50K")

# Description
st.text("Choose a Classifier in the sidebar. Input your values and get a prediction")

#sidebar
st.sidebar.subheader("Input Features")

train_X,test_X,train_A,test_A,train_y, test_y = getDataset()
st.dataframe(train_X.sample(n = 5 , random_state = 1))
st.subheader("Class and sensitive feature Distribution")
st.text("Distribution of target variable")
st.text(round(train_y.value_counts(normalize=True),2).to_dict())
st.text("Distribution of sensitive features")
st.text(round(train_A.value_counts(normalize=True),2).to_dict())
st.text("Distribution of target variable")
st.bar_chart(train_y.value_counts(normalize=True))
st.text("Distribution of sensitive variables")
if sensitive_var != 'both':
    st.bar_chart(train_A.value_counts(normalize=True))
else:
    st.bar_chart(train_A.iloc[:,0].value_counts(normalize=True))
    st.bar_chart(train_A.iloc[:,1].value_counts(normalize=True))    
    above_50k = train_X[train_y == 1]
    st.text('Distribution of sensitive variables for above 50K')
    st.bar_chart(above_50k['sex'].value_counts(normalize=True))
    st.bar_chart(above_50k['race'].value_counts(normalize=True))

if fairlearn:
    training_pipeline = model_training(sensitive=train_A)
    pipe = training_pipeline.fit(train_X,train_y)
    print(pipe)
else:
    training_pipeline = model_training()
    pipe = training_pipeline.fit(train_X,train_y)

st.subheader("Pipeline config")
st.text(pipe)  

if fairlearn:
    st.subheader("Model thresholder optimizer with Fairlearn")
    y_pred = pipe.predict(test_X,sensitive_features=test_A)
    st.pyplot(plot_threshold_optimizer(pipe))
    st.text(pipe.interpolated_thresholder_.interpolation_dict)
else:
    y_pred = pipe.predict(test_X)

if fairlearn:
    st.subheader("Model evaluation and fairness with Fairlearn")
else:
    st.subheader("Model evaluation with out Fairlearn")  

#feature importance and name the columns
if (classifier == 'Random Forest' or classifier == 'Light GBM') and fairlearn == False:
    X = train_X
    clf = pipe.named_steps['model']
    st.subheader(f'Feature Importance of {classifier} model')

    #convert the values to percentage and round upto 2 decimal places
    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X.columns)), columns=['Value','Feature'])
    feature_imp['Value'] = round((feature_imp['Value']*100),2)
    plt.figure(figsize=(20, 10))
    st.bar_chart(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('Classifier Features')
    plt.tight_layout()
    plt.show()

    # 
    # feature_importance = pipe.named_steps['model'].feature_importances_
    # st.bar_chart(feature_importance)
    # st.text(train_X.columns)
    # st.text(feature_importance)

st.text("Accuracy Score: {}".format(accuracy_score(test_y,y_pred)))
st.text("Precision Score: {}".format(precision_score(test_y,y_pred)))
st.text("Recall Score: {}".format(recall_score(test_y,y_pred)))
st.text("F1 Score: {}".format(f1_score(test_y,y_pred)))
st.text("Classification Report: {}".format(classification_report(test_y,y_pred)))
st.text("Demographic Parity Ratio: {}".format(demographic_parity_ratio(test_y,y_pred,sensitive_features=test_A)))
st.text("Equalized Odds Ratio: {}".format(equalized_odds_ratio(test_y,y_pred,sensitive_features=test_A)))

gm = MetricFrame(metrics=selection_rate, y_true=test_y.values, y_pred=y_pred, sensitive_features=test_A)
gm_acc = MetricFrame(metrics=accuracy_score, y_true=test_y.values, y_pred=y_pred, sensitive_features=test_A)
st.dataframe(gm.by_group)
st.dataframe(gm_acc.by_group)

metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'false positive rate': false_positive_rate,
    'true positive rate': true_positive_rate,
    'selection rate': selection_rate,
    'count': count}
metric_frame = MetricFrame(metrics=metrics,
                           y_true=test_y.values,
                           y_pred=y_pred,
                           sensitive_features=test_A)

summary = metric_frame.by_group
st.dataframe(summary)
