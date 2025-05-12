import json
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import sys
import os
from .Wrapper import PySimFin
from ml.reg_model import LogisticRegrModel
from ml.StockPricePredictor import load_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from etl.etl_class import StockETL

class LivePredictor:
    def __init__(self, api_key: str, logger):
        """
        Initializes the predictor by loading the ML model, scaler, and feature list.
        """
        self.simfin = PySimFin(api_key, logger)
        self.logger = logger
        self.etl = StockETL(logger=self.logger)
        
        self.logger.info("Loading ML model, scaler, and selected features...")
        
        self.logger.info("StockPricePredictor initialized successfully.")

    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches and merges stock price data, financial statements, and company info.
        
        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): The start date for fetching financial statements.
            end_date (str): The end date for fetching financial statements.
        """
        try:
            self.logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")

            # Fetch share prices
            prices_df = self.simfin.get_share_prices(ticker, start_date, end_date)
            if prices_df.empty:
                return None
            prices_df = self.rename_columns_to_model(prices_df,"livepredictor/mapping.json")
            pl_df = self.fixing_missing_years(ticker,"pl",start_date,end_date)
            bs_df = self.fixing_missing_years(ticker,"bs",start_date,end_date)
            cf_df = self.fixing_missing_years(ticker,"cf",start_date,end_date)

            # Fetch company info
            company_info_df = self.simfin.get_company_info(ticker)
            company_info_df = self.rename_columns_to_model(company_info_df,"livepredictor/mapping.json")
        
            price_methods = self.etl.config.get("share_prices_cleaning_methods", {})
            income_methods = self.etl.config.get("income_cleaning_methods", {})
            balance_methods = self.etl.config.get("balance_cleaning_methods", {})
            cashflow_methods = self.etl.config.get("cashflow_cleaning_methods", {})
            company_methods = self.etl.config.get("company_cleaning_methods", {})


            df_prices = self.etl.clean_data(prices_df, price_methods)
            df_income = self.etl.clean_data(pl_df, income_methods)
            df_balance = self.etl.clean_data(bs_df, balance_methods)
            df_cashflow = self.etl.clean_data(cf_df, cashflow_methods)
            df_company = self.etl.clean_data(company_info_df, company_methods)
    
            df_merged = self.etl.merge_data(df_prices, df_income, df_balance, df_cashflow, df_company, ticker)
            self.logger.info(f"Data successfully merged for {ticker} from {start_date} to {end_date}.")
            return df_merged
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            raise

    def fixing_missing_years(self,ticker: str,statement:str, start_date: str,end_date) -> pd.DataFrame:
        """
        Fixes missing years in the specified financial statement for the given ticker.

        Args:
            ticker (str): The stock ticker symbol.
            statement (str): The financial statement to fix ("pl", "bs", "cf").
            start_date (str): The start date for fetching financial statements.
            end_date (str): The end date for fetching financial statements.

        Returns:
            pd.DataFrame: The updated DataFrame with fixed missing years.
        """
        
        # Fetch financial statements
        try:
            df = self.simfin.get_financial_statement(ticker, statement, start_date, end_date)
        
        except Exception as e:
            df = self.simfin.get_financial_statement(ticker, statement, f"{int(start_date.split("-")[0])-2}-01-01", end_date)
            self.logger.error(f"Error fetching financial statement for {ticker}: {e}")
           
        if statement == "pl":
            df = df[df["Fiscal Period"] == "FY"]
        else:
            # Ensure Fiscal Year is numeric
            df["Fiscal Year"] = pd.to_numeric(df["Fiscal Year"], errors="coerce")
            # Sort by Fiscal Year (ascending) and Fiscal Period (descending) to get the latest quarter first
            df_sorted = df.sort_values(by=["Fiscal Year", "Fiscal Period"], ascending=[True, False])
            # Extract the last reported fiscal period for each year
            df = df_sorted.drop_duplicates(subset=["Fiscal Year"], keep="first")


        df = self.rename_columns_to_model(df,"livepredictor/mapping.json")
        df = self.ensure_latest_fiscal_year(df,start_date)
        return df



    def ensure_latest_fiscal_year(self,df: pd.DataFrame, start_year:str) -> pd.DataFrame:
        """
        Ensures that the DataFrame has data for at least `start_year`.
        If the last available fiscal year is lower than `start_year`, a new row is added with `start_year`.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing a "Fiscal Year" column.
            start_year (int): The desired starting fiscal year.

        Returns:
            pd.DataFrame: The updated DataFrame.
        """
        if df.empty:
            return df  # Return empty if no data

        # Ensure Fiscal Year is numeric
        df["Fiscal Year"] = pd.to_numeric(df["Fiscal Year"], errors="coerce")

        # Sort by Fiscal Year (ascending) and Fiscal Period (descending)
        df_sorted = df.sort_values(by=["Fiscal Year", "Fiscal Period"], ascending=[True, False])

        # Keep the latest fiscal period for each year
        df_sorted = df_sorted.drop_duplicates(subset=["Fiscal Year"], keep="first")

        # Check the last available fiscal year
        last_fiscal_year = df_sorted["Fiscal Year"].max()

        start_year = int(start_year.split("-")[0])
        if last_fiscal_year < start_year:
            last_row = df_sorted.iloc[-1].copy()
            last_row["Fiscal Year"] = start_year  # Update year to match start date
            df_sorted = pd.concat([df_sorted, last_row.to_frame().T], ignore_index=True)

        return df_sorted

    def predict_next_day(self, ticker: str, start_date: str, end_date: str) -> str:
        """
        Predicts if the stock price will increase or decrease based on the given date range.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): The start date for fetching data (format: "YYYY-MM-DD").
            end_date (str): The end date for fetching data (format: "YYYY-MM-DD").

        Returns:
            str: "Increase" or "Decrease" prediction based on the provided data.
        """
        try:
            # Fetch and preprocess data
            df = self.fetch_data(ticker, start_date, end_date)
            if df is None:
                return None
            config = load_config("ml/config_ml.json")
            predictor = LogisticRegrModel(ticker, config, self.logger)
            predictor.load_model()
            prediction = predictor.predict(df)

            self.logger.info(f"Prediction for {ticker} based on data from {start_date} to {end_date}: {prediction}")
            return prediction
        except Exception as e:
            self.logger.error(f"Error predicting stock price movement for {ticker} from {start_date} to {end_date}: {e}")
            raise

    def load_column_mapping(self,json_path: str) -> dict:
        """
        Load column mapping from a JSON file.
        
        Args:
            json_path (str): Path to the JSON file containing column mappings.
        
        Returns:
            dict: Dictionary mapping model column names to API column names.
        """
        with open(json_path, 'r') as file:
            column_mapping = json.load(file)
        return column_mapping

    def rename_columns_to_model(self,df: pd.DataFrame, json_path: str) -> pd.DataFrame:
        """
        Rename DataFrame columns to match the model's expected column names.
        Only columns present in the mapping will be renamed; others remain unchanged.
        
        Args:
            df (pd.DataFrame): DataFrame whose columns need renaming.
            json_path (str): Path to the JSON file containing column mappings.
        
        Returns:
            pd.DataFrame: DataFrame with renamed columns.
        """
        column_mapping = self.load_column_mapping(json_path)  # Load mapping (API name -> Model name)
        df.rename(columns={v: k for k, v in column_mapping.items() if v in df.columns}, inplace=True)
        return df

   