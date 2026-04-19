import sys
import os
import pandas as pd
from pandas import DataFrame  
from dataclasses import dataclass
import numpy as np 
import glob

from src.utils.logger import logging 
from src.utils.exception import Custom_exception

@dataclass
class DataCleaningConfig:
    # เช็คว่าเป็นสภาพแวดล้อม Airflow หรือไม่
    is_airflow = os.getenv("IS_AIRFLOW", "false").lower() == "true"

    if is_airflow:
        input_path = "/opt/airflow/data/"
        output_path = "/opt/airflow/artifacts/data_cleaned.csv" 
    else:
        input_path = "data"
        output_path = "artifacts/data_cleaned.csv" 

class DataCleaner:
    """
    Class สำหรับทำความสะอาดข้อมูล โดยเน้นไปที่การจัดการค่า Missing values ('na')
    """

    def __init__(self):
        self.data_cleaner_config = DataCleaningConfig()

    def load_data(self, file_path):
        try:
            logging.info(f"กำลังโหลดข้อมูลจาก: {file_path}")
            dfs = []
            file_paths = glob.glob(os.path.join(file_path, "*.csv"))
            
            for f in file_paths:
                try:
                    file = pd.read_csv(f)
                    if not file.empty:
                        dfs.append(file)
                except pd.errors.EmptyDataError:
                    logging.warning(f"ไฟล์ {f} ว่างเปล่า กำลังข้าม...")

            if not dfs:
                raise ValueError(f"ไม่พบไฟล์ CSV ที่ใช้งานได้ใน {file_path}")

            df = pd.concat(dfs, ignore_index=True)
            logging.info("โหลดข้อมูลสำเร็จ")
            return df
        except Exception as e:
            logging.info(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {str(e)}")
            raise Custom_exception(e, sys)

    def check_for_na(self, df: DataFrame):
        """ ตรวจสอบจำนวนค่า 'na' (string) ในข้อมูล """
        try:
            logging.info("กำลังตรวจสอบค่า 'na'")
            # แปลงทุกอย่างเป็น String และ Lowercase เพื่อเปรียบเทียบ
            is_na = df.apply(lambda x: x.astype(str).str.strip().str.lower() == 'na')
            
            total_na_rows = is_na.any(axis=1).sum()
            columns_na_sum = is_na.sum()

            print(f"จำนวนแถวที่มีค่า 'na': {total_na_rows}")
            print(f"\nจำนวน 'na' แยกตามคอลัมน์: \n{columns_na_sum}")

        except Exception as e:
            logging.info(f"เกิดข้อผิดพลาดในการตรวจสอบค่า NA: {str(e)}")
            raise Custom_exception(e, sys)

    def find_mode(self, df: DataFrame):
        """ คำนวณหาค่า Mode (ฐานนิยม) เพื่อเอาไปแทนที่ค่าว่าง """
        try:
            cols = df.select_dtypes(include=['object', 'category']).columns
            modes_dict = {}

            for col in cols:
                # กรองค่าที่ไม่ใช่ 'na' ออกก่อนหา Mode
                valid_data = df[col][df[col].astype(str).str.strip().str.lower() != 'na']
                mode_values = valid_data.mode()
                
                if not mode_values.empty:
                    modes_dict[col] = mode_values[0]
                else:
                    modes_dict[col] = "Unknown"

            return cols, modes_dict
        except Exception as e:
            logging.info(f"เกิดข้อผิดพลาดในการคำนวณค่า Mode: {str(e)}")
            raise Custom_exception(e, sys)

    def handling_na(self, columns, replacement_value, df: DataFrame, path):
        """ จัดการแทนที่ค่า 'na' และบันทึกไฟล์ """
        try:
            logging.info("กำลังแทนที่ค่า 'na' ด้วยค่า Mode")
            
            # แปลง String 'na' ให้เป็นค่าว่างของ Pandas เพื่อใช้ fillna ได้ง่าย
            df = df.replace(to_replace=r'^\s*[Nn][Aa]\s*$', value=pd.NA, regex=True)
            
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].fillna(replacement_value.get(col, "Unknown"))

            logging.info("จัดการค่า NA เรียบร้อย กำลังบันทึกข้อมูล...")

            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # --- CRITICAL FIX HERE ---
            # ต้องใส่ index=False เพื่อไม่ให้ CSV มีคอลัมน์ลำดับที่เกินออกมา
            df.to_csv(path, index=False) 
            # -------------------------

            logging.info(f"บันทึกไฟล์ไปที่: {path} เรียบร้อยแล้ว")
            return df
        
        except Exception as e:
            logging.info(f"เกิดข้อผิดพลาดในการจัดการ NA: {str(e)}")
            raise Custom_exception(e, sys)

    def clean_data(self):
        """ ฟังก์ชันหลักในการรัน Pipeline การคลีนข้อมูล """
        try:
            logging.info("เริ่มกระบวนการ Data Cleaning")
            df = self.load_data(self.data_cleaner_config.input_path)
            
            self.check_for_na(df)
            
            cols, replace_value = self.find_mode(df)
            
            df_cleaned = self.handling_na(
                columns=cols, 
                replacement_value=replace_value, 
                df=df, 
                path=self.data_cleaner_config.output_path
            )

            logging.info("จบกระบวนการ Data Cleaning เรียบร้อย")
            return df_cleaned
        
        except Exception as e:
            logging.error(f"เกิดข้อผิดพลาดรุนแรงในขั้นตอน clean_data: {str(e)}")
            raise Custom_exception(e, sys)