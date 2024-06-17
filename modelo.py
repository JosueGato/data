
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import xgboost as xgb



# Cargar datos
data = pd.read_csv("AIDS_Classification_5000.csv")
print(data.shape)

print(data.head(5))

print(data.info())

print(data.isnull().sum())

# Calcular matriz de correlaciones
corr_ds = data.corr()

# Ajustar el tamaño de la figura y rotar etiquetas
plt.figure(figsize=(14, 12))
sns.heatmap(corr_ds, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"size": 10})
plt.title('Mapa de Calor de Correlaciones')

# Rotar etiquetas de los ejes
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()



# Dividir datos en características y etiqueta
X = data.drop('infected', axis=1)
y = data['infected']

# Identificar variables numéricas y categóricas
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Preprocesamiento para variables numéricas
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocesamiento para variables categóricas
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Aplicar transformaciones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar preprocesador
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Entrenar modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predicciones Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluación del modelo Random Forest
acc_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy Random Forest:", acc_rf)
print("Classification Report RF:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix RF:\n", confusion_matrix(y_test, y_pred_rf))

# Visualización de la matriz de confusión Random Forest
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión - Random Forest')
plt.show()

# Entrenar modelo XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predicciones XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# Evaluación del modelo XGBoost
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print("Accuracy XGBoost:", acc_xgb)
print("Classification Report XGB:\n", classification_report(y_test, y_pred_xgb))
print("Confusion Matrix XGB:\n", confusion_matrix(y_test, y_pred_xgb))

# Visualización de la matriz de confusión XGBoost
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión - XGBoost')
plt.show()




# Streamlit application
st.title("Clasificación de Datos de AIDS")
st.write("Aplicación para predecir si un paciente está infectado basado en características clínicas.")

# Mostrar la precisión del modelo
st.write(f"Precisión del modelo XGBoost: {acc_xgb:.2f}")

# Subir datos
uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    
    # Preprocesamiento de nuevos datos
    new_data_processed = preprocessor.transform(new_data)
    
    # Predicciones con el mejor modelo
    predictions = xgb_model.predict(new_data_processed)
    
    # Mostrar predicciones
    st.write("Predicciones:")
    st.write(predictions)
    
    # Añadir las predicciones al dataframe original y mostrar
    new_data['Predicciones'] = predictions
    st.write(new_data)
    
    # Visualización de los datos cargados
    st.write("Mapa de Calor de Correlaciones - Nuevos Datos")
    corr_new_data = new_data.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_new_data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"size": 10})
    plt.title('Mapa de Calor de Correlaciones - Nuevos Datos')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(plt)
    
    # Importancia de características para los nuevos datos
    st.write("Importancia de Características - Nuevos Datos")
    importances_xgb = xgb_model.feature_importances_
    features_xgb = pd.Series(importances_xgb, index=preprocessor.get_feature_names_out()).sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=features_xgb, y=features_xgb.index)
    plt.title('Importancia de Características - Nuevos Datos')
    st.pyplot(plt)
    
    # Matriz de Confusión para los nuevos datos
    st.write("Matriz de Confusión - Nuevos Datos")
    if 'infected' in new_data.columns:
        conf_matrix_new_data = confusion_matrix(new_data['infected'], predictions)
        plt.figure(figsize=(10, 6))
        sns.heatmap(conf_matrix_new_data, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Matriz de Confusión - Nuevos Datos')
        st.pyplot(plt)
    else:
        st.write("No se puede mostrar la matriz de confusión porque los datos nuevos no contienen la columna 'infected'.")

    # Distribución de las Predicciones
    st.write("Distribución de las Predicciones")
    plt.figure(figsize=(8, 6))
    sns.countplot(predictions)
    plt.title('Distribución de las Predicciones')
    st.pyplot(plt)
    
    # Comparación de Predicciones vs Real
    if 'infected' in new_data.columns:
        st.write("Comparación de Predicciones vs Real")
        comparison_df = pd.DataFrame({'Real': new_data['infected'], 'Predicciones': predictions})
        plt.figure(figsize=(10, 6))
        sns.histplot(comparison_df, kde=True, multiple='dodge', palette='Set1')
        plt.title('Comparación de Predicciones vs Real')
        st.pyplot(plt)
    
    # Histograma de Características
    st.write("Histogramas de Características")
    num_features_new_data = new_data.select_dtypes(include=['int64', 'float64']).columns
    for feature in num_features_new_data:
        plt.figure(figsize=(8, 6))
        sns.histplot(new_data[feature], kde=True)
        plt.title(f'Histograma de {feature}')
        st.pyplot(plt)
    
    # Boxplot de Características
    st.write("Boxplots de Características")
    for feature in num_features_new_data:
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=new_data[feature])
        plt.title(f'Boxplot de {feature}')
        st.pyplot(plt)

    # Distribución de Edad por Infección
    if 'infected' in new_data.columns:
        st.write("Distribución de Edad por Infección")
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Predicciones', y='age', data=new_data)
        plt.title('Distribución de Edad por Infección')
        st.pyplot(plt)

    # PCA (Análisis de Componentes Principales)
    st.write("PCA (Análisis de Componentes Principales)")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(new_data_processed)
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Predicciones'] = predictions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Predicciones', palette='Set1', data=pca_df)
    plt.title('PCA (Análisis de Componentes Principales)')
    st.pyplot(plt)

    # Distribución de Infecciones por Género
    if 'gender' in new_data.columns:
        st.write("Distribución de Infecciones por Género")
        plt.figure(figsize=(8, 6))
        sns.countplot(x='gender', hue='Predicciones', data=new_data, palette='Set1')
        plt.title('Distribución de Infecciones por Género')
        st.pyplot(plt)