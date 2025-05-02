import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def cargar_dataset(ruta: str) -> pd.DataFrame:
  df = pd.read_csv(ruta)
  return df

def preparar_datos(df: pd.DataFrame):
  df.drop_duplicates(inplace=True)
  df.dropna(inplace=True)
  df.rename(columns={'weight (kg)': 'weight_kg', 'price (USD)': 'price_usd'}, inplace=True)

  X = df.drop(['id', 'price_usd'], axis=1)
  y = df['price_usd']
  return X, y

def crear_preprocesador():

  num_features = ["gear_count", "wheel_size", "weight_kg"]
  cat_features = ["brand", "type", "frame_material"]

  num_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
  ])

  cat_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop='first'))
  ])

  preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
  ])

  return preprocessor

def crear_pipeline(model, preprocessor):

  pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", model)
  ])
  return pipeline

def definir_param_grid(model_name: str):

  if model_name == "LinearRegression":
    param_grid = {
    }
  elif model_name == "RandomForest":
    param_grid = {
      "regressor__n_estimators": [50, 100],
      "regressor__max_depth": [None, 5, 10],
      "regressor__min_samples_split": [2, 5]
    }
  elif model_name == "KNN":
    param_grid = {
      "regressor__n_neighbors": [3, 5, 7],
      "regressor__weights": ["uniform", "distance"]
    }
  else:
    param_grid = {}
  return param_grid

def buscar_mejor_modelo(X, y, preprocessor):

  modelos = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
  }

  resultados = {}
  best_score = -np.inf
  best_pipeline = None
  best_model_name = None

  for name, model in modelos.items():
    print(f"\nBuscando mejores hiperparámetros para {name}...")
    pipeline = crear_pipeline(model, preprocessor)
    param_grid = definir_param_grid(name)

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X, y)

    score = grid_search.best_score_
    print(f"Mejor R² para {name}: {score:.4f}")
    print(f"Mejores parámetros: {grid_search.best_params_}")

    resultados[name] = {
      "best_score": score,
      "best_params": grid_search.best_params_
    }

    # Mejor modelo según R²
    if score > best_score:
      best_score = score
      best_pipeline = grid_search.best_estimator_
      best_model_name = name

  return best_pipeline, best_model_name, resultados

def evaluar_modelo(pipeline, X_test, y_test):
  """Evalúa el modelo en el conjunto de prueba utilizando las métricas MAE, RMSE y R².
  """
  y_pred = pipeline.predict(X_test)

  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2 = r2_score(y_test, y_pred)

  metrics = {"MAE": mae, "RMSE": rmse, "R²": r2}
  return metrics

def guardar_modelo(pipeline, directorio="models"):

  if not os.path.exists(directorio):
    os.makedirs(directorio)

  modelo_path = os.path.join(directorio, "model.pkl")
  preprocessor_path = os.path.join(directorio, "preprocessor.pkl")

  with open(modelo_path, "wb") as f:
    pickle.dump(pipeline, f)

  preprocessor = pipeline.named_steps["preprocessor"]
  with open(preprocessor_path, "wb") as f:
    pickle.dump(preprocessor, f)

  print(f"Modelo guardado en {modelo_path}")
  print(f"Preprocesador guardado en {preprocessor_path}")

def graficar_resultados_modelos(resultados: dict, save_path: str = None):
  """Genera una gráfica de barras comparando el mejor R² obtenido para cada modelo.
  """
  modelos = list(resultados.keys())
  scores = [resultados[m]["best_score"] for m in modelos]

  plt.figure(figsize=(8, 6))
  bars = plt.bar(modelos, scores, color="skyblue")
  plt.xlabel("Modelos")
  plt.ylabel("Mejor R²")
  plt.title("Comparación de Modelos (R²)")

  # Mostrar valor sobre cada barra
  for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.3f}",
      xy=(bar.get_x() + bar.get_width() / 2, height),
      xytext=(0, 3),
      textcoords="offset points",
      ha='center', va='bottom')

  plt.ylim(0, 1)
  plt.tight_layout()
  if save_path:
    plt.savefig(save_path)
    print(f"Gráfica de resultados guardada en {save_path}")
  plt.show()


def graficar_metricas(metrics: dict, save_path: str = None):
  """Genera una gráfica de barras para visualizar las métricas del modelo evaluado.
  """
  nombres = list(metrics.keys())
  valores = list(metrics.values())

  plt.figure(figsize=(8, 6))
  bars = plt.bar(nombres, valores, color="lightgreen")
  plt.xlabel("Métricas")
  plt.ylabel("Valor")
  plt.title("Evaluación del Mejor Modelo en Test")

  for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.3f}",
      xy=(bar.get_x() + bar.get_width() / 2, height),
      xytext=(0, 3),
      textcoords="offset points",
      ha='center', va='bottom')

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path)
    print(f"Gráfica de métricas guardada en {save_path}")
  plt.show()


def main():
  """Función principal
  """
  # Cargar dataset
  print("Cargando dataset...")
  df = cargar_dataset("data/bike_prices.csv")

  # Preparar datos
  X, y = preparar_datos(df)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Crear preprocesador
  preprocessor = crear_preprocesador()

  # Buscar el mejor modelo con GridSearchCV
  best_pipeline, best_model_name, resultados = buscar_mejor_modelo(X_train, y_train, preprocessor)
  print(f"\nEl mejor modelo es: {best_model_name}")

  # Evaluar el modelo en el conjunto de prueba
  metrics = evaluar_modelo(best_pipeline, X_test, y_test)
  print("\nResultados en conjunto de prueba:")
  for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

  # Realizar gráficas
  graficar_resultados_modelos(resultados, save_path="resultados_modelos.png")
  graficar_metricas(metrics, save_path="metricas_test.png")

  # Guardar el modelo final y el preprocesador
  guardar_modelo(best_pipeline)

if __name__ == "__main__":
    main()


import os
import pickle

def load_pipeline(model_path: str = os.path.join("model", "model.pkl")):
    """
    Carga el pipeline (modelo + preprocesador) desde un archivo pickle.
    """
    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    except Exception as e:
        raise Exception("Error al cargar el pipeline: " + str(e))


loaded_pipeline = load_pipeline()

new_data = pd.DataFrame({
    'brand': ['Trek'],
    'type': ['Mountain'],
    'frame_material': ['Aluminum'],
    'gear_count': [21],
    'wheel_size': [27.5],
    'weight_kg': [13.5]
})

prediction = loaded_pipeline.predict(new_data)

# Mostrar la predicción
print(f"Predicción del precio: {prediction[0]}")

df = cargar_dataset("bike_prices.csv")

df.frame_material.unique()