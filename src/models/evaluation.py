import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss



# initialize dagshub
import dagshub
dagshub.init(repo_owner='Adrshp806', repo_name='Health-App', mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/Adrshp806/Health-App.mlflow")
# set mlflow experment name
mlflow.set_experiment("DVC Pipeline")


TARGET = "diagnosis"



# create logger
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
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


def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def load_model(model_path: Path):
    model = joblib.load(model_path)
    # Check if model has been fitted
    if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
        logger.info("Model is already fitted")
    else:
        logger.error("Model is not fitted!")
        # You might want to raise an exception or handle this case
    return model



def save_model_info(save_json_path,run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path,"w") as f:
        json.dump(info_dict,f,indent=4)

if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # train data load path
    train_data_path = root_path / "data" / "cleaned_processed"/ "processed_cancer" / "train_trans.csv"
    test_data_path = root_path / "data" / "cleaned_processed"/ "processed_cancer" / "test_trans.csv"
    # model path
    model_path = root_path / "models" / "model.joblib"

    # load the training data
    train_data = load_data(train_data_path)
    logger.info("Train data loaded successfully")
    # load the test data
    test_data = load_data(test_data_path)
    logger.info("Test data loaded successfully")

      # split the train and test data
    X_train, y_train = make_X_and_y(train_data,TARGET)
    X_test, y_test = make_X_and_y(test_data,TARGET)
    logger.info("Data split completed")
    
    # load the model
    model = load_model(model_path)
    logger.info("Model Loaded successfully")
    
   # Get the predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("Prediction on data complete")

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    logger.info(f"Train Accuracy: {train_accuracy}")
    logger.info(f"Test Accuracy: {test_accuracy}")
    

    # Misclassification Error
    train_misclassification_error = 1 - train_accuracy
    test_misclassification_error = 1 - test_accuracy
    logger.info(f"Train Misclassification Error: {train_misclassification_error}")
    logger.info(f"Test Misclassification Error: {test_misclassification_error}")

    # Precision, Recall, F1-Score (for binary classification)
    train_precision = precision_score(y_train, y_train_pred, average='binary')
    test_precision = precision_score(y_test, y_test_pred, average='binary')
    train_recall = recall_score(y_train, y_train_pred, average='binary')
    test_recall = recall_score(y_test, y_test_pred, average='binary')
    train_f1 = f1_score(y_train, y_train_pred, average='binary')
    test_f1 = f1_score(y_test, y_test_pred, average='binary')
    logger.info(f"Train Precision: {train_precision}")
    logger.info(f"Test Precision: {test_precision}")
    logger.info(f"Train Recall: {train_recall}")
    logger.info(f"Test Recall: {test_recall}")
    logger.info(f"Train F1 Score: {train_f1}")
    logger.info(f"Test F1 Score: {test_f1}")
    # calculate cross val scores
    # Calculate cross-validation scores using accuracy as the scoring metric
    cv_scores = cross_val_score(model,
                                X_train,
                                y_train,
                                cv=5,
                                scoring="accuracy",  # Use accuracy for classification
                                n_jobs=-1)
    logger.info("Cross-validation complete with accuracy scores")
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV Accuracy: {cv_scores.mean()}")
    
    # mean cross-validation score (negate to get positive log loss)
    mean_cv_score = -(cv_scores.mean())

    logger.info(f"Mean CV Log Loss: {mean_cv_score}")
     # log with mlflow
    with mlflow.start_run() as run:
        # set tags
        mlflow.set_tag("model","cancer_prediction_class")

        # log parameters
        mlflow.log_params(model.get_params())

        # Log classification metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.log_metric("train_misclassification_error", train_misclassification_error)
        mlflow.log_metric("test_misclassification_error", test_misclassification_error)

        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("test_precision", test_precision)

        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("test_recall", test_recall)

        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_f1", test_f1)

        # log individual cv scores
        mlflow.log_metrics({f"CV {num}": score for num, score in enumerate(-cv_scores)})
        
        # mlflow dataset input datatype
        train_data_input = mlflow.data.from_pandas(train_data,targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data,targets=TARGET)
        
        # log input
        mlflow.log_input(dataset=train_data_input,context="training")
        mlflow.log_input(dataset=test_data_input,context="validation")
        
        # model signature
        model_signature = mlflow.models.infer_signature(model_input=X_train.sample(20,random_state=42),
                                    model_output=model.predict(X_train.sample(20,random_state=42)))
        
        # log the final model
        mlflow.sklearn.log_model(model,"delivery_time_pred_model",signature=model_signature)

        # log stacking regressor
        mlflow.log_artifact(root_path / "models" / "model.joblib")
        
        # # log the power transformer
        # mlflow.log_artifact(root_path / "models" / "power_transformer.joblib")
        
        # log the preprocessor
        mlflow.log_artifact(root_path / "models" / "cancer_preprocessor.joblib")
        
        # get the current run artifact uri
        artifact_uri = mlflow.get_artifact_uri()
        
        logger.info("Mlflow logging complete and model logged")
     # get the run id 
    run_id = run.info.run_id
    model_name = "cancer_pred_model"
    
    # save the model info
    save_json_path = root_path / "run_information.json"
    save_model_info(save_json_path=save_json_path,
                    run_id=run_id,
                    artifact_path=artifact_uri,
                    model_name=model_name)
    logger.info("Model Information saved")