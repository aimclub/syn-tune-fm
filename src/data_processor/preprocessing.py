import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from src.data_processor.schema import TabularSchema 

class TabularPreprocessor:
    def __init__(self):
        self.schema = None
        self.scalers = {}
        self.encoders = {}
        self.target_scaler = None

    def transform_global(self, df: pd.DataFrame, schema: TabularSchema) -> pd.DataFrame:
        """
        Stub method for TabularDataModule.
        Tells the module that global transformations before splitting are not needed.
        This prevents data leakage.
        """
        return df

    def fit(self, df: pd.DataFrame, schema: TabularSchema):
        self.schema = schema
        
        # 1. Scalers for continuous and discrete features
        numeric_cols = schema.continuous_cols + schema.discrete_cols
        for col in numeric_cols:
            scaler = StandardScaler()
            scaler.fit(df[[col]].dropna())
            self.scalers[col] = scaler

        # 2. Encoders for categorical features
        for col in schema.categorical_cols:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            enc.fit(df[[col]].dropna().astype(str))
            self.encoders[col] = enc

        # 3. Target scaling
        # TabPFN classification labels are usually integer class codes.
        # Scaling such integer-like targets breaks classification.
        if schema.target_col and pd.api.types.is_numeric_dtype(df[schema.target_col]):
            s = df[schema.target_col].dropna()
            is_integer_like = False
            if pd.api.types.is_integer_dtype(s):
                is_integer_like = True
            else:
                # integer-valued numeric (e.g. 0.0, 1.0, ...)
                is_integer_like = bool((s.astype("float64") % 1 == 0).all())

            if (not is_integer_like) and (df[schema.target_col].nunique() > 20):
                self.target_scaler = StandardScaler()
                self.target_scaler.fit(df[[schema.target_col]].dropna())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.schema is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
            
        df_out = df.copy()
        
        # Scaling
        for col, scaler in self.scalers.items():
            if col in df_out.columns:
                # If the column was categorical, force it to float before scaling
                if isinstance(df_out[col].dtype, pd.CategoricalDtype):
                    df_out[col] = df_out[col].astype(float)
                    
                notna_mask = df_out[col].notna()
                df_out.loc[notna_mask, col] = scaler.transform(df_out.loc[notna_mask, [col]]).flatten()
                
        # Encoding
        for col, enc in self.encoders.items():
            if col in df_out.columns:
                # IMPORTANT: remove pandas.Categorical restrictions by converting to object
                if isinstance(df_out[col].dtype, pd.CategoricalDtype):
                    df_out[col] = df_out[col].astype('object')
                
                notna_mask = df_out[col].notna()
                df_out.loc[notna_mask, col] = enc.transform(df_out.loc[notna_mask, [col]].astype(str)).flatten()
                
                # Convert to numeric type to make it easier for models (TabPFN)
                df_out[col] = pd.to_numeric(df_out[col])
                
        # Target
        if self.target_scaler and self.schema.target_col in df_out.columns:
            if isinstance(df_out[self.schema.target_col].dtype, pd.CategoricalDtype):
                df_out[self.schema.target_col] = df_out[self.schema.target_col].astype('object')
                
            notna_mask = df_out[self.schema.target_col].notna()
            df_out.loc[notna_mask, self.schema.target_col] = self.target_scaler.transform(
                df_out.loc[notna_mask, [self.schema.target_col]]
            ).flatten()
            
        return df_out