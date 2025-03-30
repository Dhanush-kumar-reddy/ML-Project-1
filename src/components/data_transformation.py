import sys
from dataclasses import dataclass

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object


from src.exception import CustumeException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsibles for data transformation
        '''
        try:
            num_features = ["reading score","writing score"]
            cat_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            
            num_pipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    
                ]
            )
            
            cat_pipline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                    ]
            )    
            
            logging.info(f"Numerical columns {num_features}")
            logging.info(f"Catogorical columns {cat_features}")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipline,num_features),
                    ("cat_pipeline",cat_pipline,cat_features)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustumeException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing Object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name = "math score"
            numerical_columns = ["reading score","writing score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            terget_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            terget_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying Preprocessing object on training and test dataset")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[input_feature_train_arr,np.array(terget_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(terget_feature_test_df)]
            
            logging.info("Saved Preprocessing Object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustumeException(e,sys)