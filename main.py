#Necesitamos instalar las bibliotecas necesarias para que funcione el programa#
#pip install pandas, numpy, matplotlib, seaborn, scikit-learn, imblearn, xgboost#
#pip install PyQt5 pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost pyqt5-tools PySide5#
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
import sys
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        uic.loadUi('design.ui', self)  # Asegúrate de que el archivo UI esté en el mismo directorio

        # Configuraciones de la ventana
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowOpacity(1)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Configuraciones de los botones
        self.btnclose.clicked.connect(lambda: self.close()) 
        self.btnLoadData.clicked.connect(self.cargar_datos_en_tabla)
        self.btnXG.clicked.connect(self.entrenar_modelo_xgboost)
        self.btnRF.clicked.connect(self.entrenar_modelo_rf)
        self.btncleardata.clicked.connect(self.clear_data)
    def mostrar_error(self, mensaje):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText(mensaje)
        error_dialog.setWindowTitle("Error")
        error_dialog.exec_()

    def cargar_datos(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'Dataset_de_Predicci_n_de_Abandono_Escolar.csv')
        df = pd.read_csv(csv_path)

        # Agregar nuevas variables
        df["Historial_Notas"] = np.random.uniform(5, 20, df.shape[0])
        df["Reportes_Disciplinarios"] = np.random.randint(0, 5, df.shape[0])
        df["Faltas_Acumuladas"] = df["Faltas"] * np.random.uniform(0.9, 1.1, df.shape[0])
        df["Acceso_a_Recursos"] = np.random.randint(0, 2, df.shape[0])
        df["Horas_Estudio_Semanal"] = np.random.uniform(0, 20, df.shape[0])
        df["Estrés_Académico"] = df["Nivel_Estrés"] * np.random.uniform(0.8, 1.2, df.shape[0])
        return df

    def clear_data(self):
        self.textArea.clear()
        
    def cargar_datos_en_tabla(self):
        try:
            # Cargar los datos
            df = self.cargar_datos()
            
            # Limpiar la tabla antes de cargar nuevos datos
            self.table.setRowCount(0)  # Eliminar todas las filas de la tabla
            self.table.setColumnCount(len(df.columns))  # Establecer la cantidad de columnas según el DataFrame
            self.table.setHorizontalHeaderLabels(df.columns)  # Establecer los nombres de las columnas en la tabla
            
            # Rellenar la tabla con los datos
            for row in df.itertuples():
                row_position = self.table.rowCount()
                self.table.insertRow(row_position)  # Agregar una nueva fila
                for col_num, value in enumerate(row[1:]):  # Omite el índice
                    self.table.setItem(row_position, col_num, QTableWidgetItem(str(value)))
        except Exception as e:
            self.mostrar_error(f"Error al cargar los datos: {str(e)}")

    def preprocesar_datos(self, df):
        X = df[["Reportes_Disciplinarios", "Nivel_Estrés", "Tareas_No_Entregadas", "Faltas_Acumuladas", "Acceso_a_Recursos", "Horas_Estudio_Semanal", "Estrés_Académico"]]
        y = df["Abandono"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def balancear_datos(self, X_train, y_train):
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled

    def entrenar_random_forest_modelo(self, X_train, y_train):
        modelo = RandomForestClassifier(n_estimators=350, max_depth=10, class_weight="balanced_subsample", min_samples_split=2, min_samples_leaf=2, random_state=42)
        modelo.fit(X_train, y_train)
        return modelo

    def entrenar_xgboost_modelo(self, X_train, y_train):
        modelo = XGBClassifier(
            n_estimators=300, learning_rate=0.02, max_depth=5, scale_pos_weight=4.8, 
            colsample_bytree=0.7, subsample=0.9, gamma=0.3, reg_lambda=1.5, reg_alpha=0.7, random_state=42
        )
        modelo.fit(X_train, y_train)
        return modelo

    def evaluar_random_forest(self, modelo, X_test, y_test):
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        return accuracy, conf_matrix, class_report

    def evaluar_xgboost(self, modelo, X_test, y_test):
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        return accuracy, conf_matrix, class_report

    def plot_confusion_matrix(self, matrix, title):
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=["No Abandona", "Abandona"], yticklabels=["No Abandona", "Abandona"])
        plt.xlabel("Predicción")
        plt.ylabel("Realidad")
        plt.title(title)
        plt.show()

    def entrenar_modelo_rf(self):
        try:
            # Cargar los datos y preprocesarlos
            df = self.cargar_datos()
            X_train, X_test, y_train, y_test = self.preprocesar_datos(df)
            
            # Balancear los datos usando SMOTE
            X_train_resampled, y_train_resampled = self.balancear_datos(X_train, y_train)

            # Entrenar el modelo Random Forest
            modelo_rf = self.entrenar_random_forest_modelo(X_train_resampled, y_train_resampled)

            # Evaluar el modelo y obtener métricas
            accuracy_rf, conf_matrix_rf, class_report_rf = self.evaluar_random_forest(modelo_rf, X_test, y_test)

            # Crear el texto con los resultados del modelo
            resultado_rf = "\n🔹 Evaluación del Modelo: Random Forest 🔹\n\n"
            resultado_rf += f"✅ Precisión: {accuracy_rf:.2%}\n\n"
            resultado_rf += "📊 Matriz de Confusión:\n"
            resultado_rf += f"{conf_matrix_rf}\n\n"
            resultado_rf += "📌 Reporte de Clasificación:\n"
            resultado_rf += f"{class_report_rf}"

            # Mostrar los resultados en el QTextBrowser
            self.textArea.setText(resultado_rf)

            # Mostrar la matriz de confusión en un gráfico
            self.plot_confusion_matrix(conf_matrix_rf, "Matriz de Confusión - Random Forest")

        except Exception as e:
            # Mostrar un mensaje de error si algo falla
            self.mostrar_error(f"Error al entrenar el modelo Random Forest: {str(e)}")


    def entrenar_modelo_xgboost(self):
        try:
            # Cargar los datos y preprocesarlos
            df = self.cargar_datos()
            X_train, X_test, y_train, y_test = self.preprocesar_datos(df)
            
            # Balancear los datos usando SMOTE
            X_train_resampled, y_train_resampled = self.balancear_datos(X_train, y_train)
            
            # Entrenar el modelo XGBoost
            modelo_xgb = self.entrenar_xgboost_modelo(X_train_resampled, y_train_resampled)
            
            # Evaluar el modelo y obtener métricas
            accuracy_xgb, conf_matrix_xgb, class_report_xgb = self.evaluar_xgboost(modelo_xgb, X_test, y_test)

            # Crear el texto con los resultados del modelo
            resultado_xgb = "\n🔹 Evaluación del Modelo: XGBoost 🔹\n\n"
            resultado_xgb += f"✅ Precisión: {accuracy_xgb:.2%}\n\n"
            resultado_xgb += "📊 Matriz de Confusión:\n"
            resultado_xgb += f"{conf_matrix_xgb}\n\n"
            resultado_xgb += "📌 Reporte de Clasificación:\n"
            resultado_xgb += f"{class_report_xgb}"

            # Mostrar los resultados en el QTextBrowser
            self.textArea.setText(resultado_xgb)

            # Mostrar la matriz de confusión en un gráfico
            self.plot_confusion_matrix(conf_matrix_xgb, "Matriz de Confusión - XGBoost")

        except Exception as e:
            # Mostrar un mensaje de error si algo falla
            self.mostrar_error(f"Error al entrenar el modelo XGBoost: {str(e)}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec())
