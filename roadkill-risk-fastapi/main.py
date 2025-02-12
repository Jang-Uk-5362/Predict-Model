from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List, Dict
from pydantic import BaseModel
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import joblib
import pickle
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# FastAPI 앱 초기화
app = FastAPI(
    version="1.0.0",
    root_path="/ai"  # ✅ 모든 API의 기본 경로를 `/ai`로 설정
    docs_url="/ai/docs",  # ✅ Swagger UI 경로 변경
    )
# CSV 파일 로드 (VS Code의 src 디렉토리 기준)
df_habitat = pd.read_csv('서식지.csv', encoding='utf-8')  # 서식지 정보
df_corridor = pd.read_csv('생태통로.csv', encoding='utf-8')  # 생태통로 정보
df_river = pd.read_csv('하천.csv', encoding='utf-8')  # 하천 정보
df_road = pd.read_csv('road.csv', encoding='utf-8')  # 도로 정보
data_full = pd.read_csv('data.csv', encoding='utf-8')  # 전체 데이터

# 모델 및 스케일러 로드
with open("./rf_model.pkl_2", "rb") as file:
    rf_model = pickle.load(file)

with open("./scaler.pkl_2", "rb") as file:
    loaded_scaler = pickle.load(file)

# 사분위수 기반 위험도 분류
quantiles = data_full['발생건수(5km)'].quantile([0.25, 0.5, 0.75])

def categorize(value):
    if value <= quantiles[0.25]:
        return '저위험'
    elif value <= quantiles[0.5]:
        return '중위험'
    elif value <= quantiles[0.75]:
        return '고위험'
    else:
        return '매우고위험'

# 거리 계산 함수
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def count_within_radius_vectorized(lat, lon, obj_lat, obj_lon, radius=5.0):
    distances = haversine_vectorized(np.array([lat]), np.array([lon]), obj_lat.values, obj_lon.values)
    return np.sum(distances <= radius)

def calculate_shortest_distance(lat, lon, df):
    distances = haversine_vectorized(np.array([lat]), np.array([lon]), df['위도'].values, df['경도'].values)
    return np.min(distances)

def calculate_river_distance(lat, lon, df_river):
    start_distances = haversine_vectorized(np.array([lat]), np.array([lon]), df_river['시점위치(위도)'].values, df_river['시점위치(경도)'].values)
    end_distances = haversine_vectorized(np.array([lat]), np.array([lon]), df_river['종점위치(위도)'].values, df_river['종점위치(경도)'].values)
    return np.minimum(np.min(start_distances), np.min(end_distances))

def find_nearest_road(lat, lon, df_road):
    distances = haversine_vectorized(np.array([lat]), np.array([lon]), df_road['위도'].values, df_road['경도'].values)
    nearest_idx = np.argmin(distances)
    return (
        df_road.iloc[nearest_idx]['평균 총차량수'],
        df_road.iloc[nearest_idx]['평균 속도'],
        df_road.iloc[nearest_idx]['기울기']
    )

def calculate_all_metrics(lat, lon, df_habitat, df_corridor, df_river, df_road, radius=5.0):
    habitat_distance = calculate_shortest_distance(lat, lon, df_habitat)
    habitat_count = count_within_radius_vectorized(lat, lon, df_habitat['위도'], df_habitat['경도'], radius)
 
    corridor_distance = calculate_shortest_distance(lat, lon, df_corridor)
    corridor_count = count_within_radius_vectorized(lat, lon, df_corridor['위도'], df_corridor['경도'], radius)
 
    river_distance = calculate_river_distance(lat, lon, df_river)
    river_start_count = count_within_radius_vectorized(lat, lon, df_river['시점위치(위도)'], df_river['시점위치(경도)'], radius)
    river_end_count = count_within_radius_vectorized(lat, lon, df_river['종점위치(위도)'], df_river['종점위치(경도)'], radius)
    river_count = river_start_count + river_end_count
 
    avg_traffic, avg_speed, slope = find_nearest_road(lat, lon, df_road)
 
    return {
        '위도': lat,
        '경도': lon,
        '서식지(최단거리)': habitat_distance,
        '생태통로(최단거리)': corridor_distance,
        '하천(최단거리)': river_distance,
        '서식지 개수(5km)': habitat_count,
        '생태통로 개수(5km)': corridor_count,
        '하천 개수(5km)': river_count,
        '평균 총차량수': avg_traffic,
        '평균 속도': avg_speed,
        '기울기': slope
    }


# 예측 API (GET 방식)
@app.get("/predict")
def predict(lat: float = Query(...), lon: float = Query(...)):
    # 거리 및 환경 요소 계산
    result = calculate_all_metrics(lat, lon,df_habitat=df_habitat, df_corridor=df_corridor,df_river=df_river,df_road=df_road)

    # 입력 데이터 변환
    input_data = np.array([list(result.values())]).reshape(1, -1)
    feature_names = ['위도', '경도', '서식지(최단거리)', '생태통로(최단거리)', '하천(최단거리)', 
                     '서식지 개수(5km)', '생태통로 개수(5km)', '하천 개수(5km)', '평균 총차량수', 
                     '평균 속도', '기울기']
    
    input_scaled = loaded_scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

    # 예측 수행
    predicted_log_value = rf_model.predict(input_scaled_df)[0]
    predicted_value = np.expm1(predicted_log_value)  # 로그 변환 복원

    category = categorize(predicted_value)

    return {
        "예측된 발생건수(5km)": format(predicted_value, ".4f"),
        "위험도": category
    }

# 기본 페이지
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Wildlife Risk Prediction API!"}
