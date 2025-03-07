
import pandas as pd
import numpy as np
import re
from typing import Union
from .info import process_step_mapping



class DataObfuscator:
    
    def __init__(self, rename_dict: dict[str, str] = None):

        self.rename_dict = rename_dict or {}
        self.process_step_mapping = process_step_mapping
        

    def rename_features(self, feature_names: list[str]) -> list[str]:
        """
        Rename features based on the provided dictionary.
        """
        renamed = []

        for feature in feature_names:
        
            for old, new in self.rename_dict.items():
                feature = feature.replace(old, new)
        
            renamed.append(feature)
        
        return renamed
    

    def map_process_steps(self, feature_names: list[str]) -> list[str]:
        """
        Map process steps to names given by mapping (Start, Ramp, Polish).
        """
        renamed = []
        for feature in feature_names:
            mapped_feature = feature

            for step_num, step_name in self.process_step_mapping.items():
                mapped_feature = re.sub(f'_ps{step_num}(?=_|$)', f' {step_name}', mapped_feature)

            renamed.append(mapped_feature)

        return renamed
    

    def separate_camel_case(self, feature_names: list[str]) -> list[str]:
        """
        Separate camel case, preserve consecutive capital letters (acronyms).
        """
        def separate_with_preserved_acronyms(s: str) -> str:
            # Pattern matches:
            # (?<!^) - not at start of string
            # (?<![\sA-Z]) - not after a space or capital letter
            # (?=[A-Z]) - followed by capital letter
            return re.sub(r'(?<!^)(?<![\sA-Z])(?=[A-Z])', ' ', s)
        
        return [separate_with_preserved_acronyms(feature) for feature in feature_names]
    

    def replace_underscores(self, feature_names: list[str]) -> list[str]:
        """
        Replace underscores with spaces.
        """
        return [feature.replace('_', ' ') for feature in feature_names]
    

    def obfuscate(self, feature_names: list[str]) -> list[str]:
        """
        Apply all transformations in sequence.
        """
        features = self.rename_features(feature_names)
        features = self.map_process_steps(features)
        features = self.separate_camel_case(features)
        features = self.replace_underscores(features)

        return features
    

    def obfuscate_ind(self, feature_name: str) -> str:

        feature_names = [feature_name]

        new_feature_names = self.obfuscate(feature_names = feature_names)
        
        return new_feature_names[0]
    
    
    def obfuscate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Obfuscates dataframe.
        """
        new_columns = self.obfuscate(df.columns)
        return df.rename(columns=dict(zip(df.columns, new_columns)))
    
