import pandas as pd
import yaml
import joblib
import logging
from sklearn.preprocessing import PowerTransformer
from pathlib import Path
from sklearn.ensemble import StackingRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score           # Accuracy
from sklearn.metrics import classification_report    # Classification report
from sklearn.metrics import confusion_matrix         # Confusion matrix
from sklearn.metrics import roc_auc_score            # ROC AUC Score
from sklearn.metrics import roc_curve, precision_recall_curve  # Curves
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


TARGET = "diagnosis"


#create logger
logger = logging.getLogger("Model_traing_for_cancer")
logger.setLevel(logging.INFO)


#consol handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)


#add handler
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    
    except FileNotFoundError:
        logger.error("The file to load does not exist")
    return df

def read_params(file_path):
    with open(file_path,"r") as f:
        params_file = yaml.safe_load(f)
    
    return params_file


def save_model(model, save_dir: Path, model_name: str):
    # form the save location
    save_location = save_dir / model_name
    # save the model
    joblib.dump(value=model,filename=save_location)
    
    
# def save_transformer(transformer, save_dir: Path, transformer_name: str):
#     # form the save location
#     save_location = save_dir / transformer_name
#     # save the transformer
#     joblib.dump(transformer, save_location)
    
    
def train_model(model, X_train: pd.DataFrame, y_train):
    try:
        # Fit the model on the training data
        model.fit(X_train, y_train)
        logger.info(f"Model {model.__class__.__name__} successfully trained on the data.")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise



def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y



if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # train data load path
    data_path = root_path / "data" / "cleaned_processed"/ "processed_cancer" / "train_trans.csv"
    # parameters file
    params_file_path = root_path / "params.yaml"  #need to set params yamal

    
    # load the training data
    training_data = load_data(data_path)
    logger.info("Training Data read successfully")
    
    # split the data into X and y
    X_train, y_train = make_X_and_y(training_data, TARGET)
    logger.info("Dataset splitting completed")
    
    # model parameters
    model_params = read_params(params_file_path)['Train']

    # rf_params
    rf_params = model_params['Random_Forest']
    logger.info("Random forest parameters read")
        
    # build random forest model
    rf = RandomForestClassifier(**rf_params)
    logger.info("Built random forest model")

    # light gbm params
    lgbm_params = model_params["LightGBM"]
    logger.info("Light GBM parameters read")
    lgbm = LGBMClassifier(**lgbm_params)  # Use LGBMClassifier for classification
    logger.info("Built Light GBM model")
    # meta model
    lr = LogisticRegression()  # Use LogisticRegression for classification tasks
    logger.info("Meta model built")

     # Build the Stacking Classifier
    stacking_clf = StackingClassifier(estimators=[("rf_model", rf),
                                                 ("lgbm_model", lgbm)],
                                     final_estimator=lr,
                                     cv=5, n_jobs=-1)
    logger.info("Stacking Classifier built")
    # Fit the stacking classifier on the transformed target
    stacking_clf.fit(X_train, y_train)
    logger.info("Stacking Classifier trained successfully")


    #fit the model on training data
    model_filename = "model.joblib"
    # directory to save model
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)

    # Save the stacking classifier model directly
    stacking_model = stacking_clf  # Just use the model directly in classification

    # Save the model
    save_model(model=stacking_model,
            save_dir=model_save_dir,
            model_name=model_filename)
    logger.info("Trained stacking classifier model saved to location")

#    # Save the transformer
#     transformer_filename = "power_transformer.joblib"
#     transformer_save_dir = model_save_dir
#     save_transformer(transformer, transformer_save_dir, transformer_filename)
#     logger.info("Transformer saved to location")
