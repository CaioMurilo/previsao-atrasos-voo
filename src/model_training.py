import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

def train_and_evaluate_model(dataset_path: str, model_save_path: str):
    """
    Carrega os dados processados, treina o modelo XGBoost e salva o arquivo final.
    """
    # 1. Carregando os dados
    df = pd.read_csv(dataset_path)
    
    # 2. Engenharia de Features (One-Hot Encoding)
    # Modelos matemáticos não entendem texto (ex: "LATAM", "Gol"). 
    # get_dummies transforma essas categorias em colunas binárias (0 ou 1).
    features_categoricas = ["cia_aerea", "origem", "destino"]
    df_encoded = pd.get_dummies(df, columns=features_categoricas, drop_first=True)
    
    # Removemos colunas que não ajudam na previsão (como ID do voo) ou que vazam a resposta
    X = df_encoded.drop(columns=["id_voo", "chegada_prevista", "target_atraso"])
    y = df_encoded["target_atraso"] # Nosso alvo: 0 (No horário) ou 1 (Atrasado)
    
    # 3. Divisão de Treino e Teste
    # Separados 80% dos dados para o modelo estudar e 20% para aplicarmos uma "prova" cega
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Configuração e Treinamento do XGBoost
    print("Treinando o modelo XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,      # Número de "árvores de decisão" que ele vai criar
        learning_rate=0.1,     # Velocidade de aprendizado
        max_depth=5,           # Profundidade máxima de cada árvore
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 5. Avaliação do Modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia do Modelo: {accuracy * 100:.2f}%\n")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 6. Salvando o Modelo Treinado para o Deploy na AWS
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path)
    print(f"Modelo salvo com sucesso em: {model_save_path}")

if __name__ == "__main__":
    caminho_dados = "data/processed/dataset_modelo.csv"
    caminho_modelo = "models/xgboost_atrasos.json"
    
    try:
        train_and_evaluate_model(caminho_dados, caminho_modelo)
    except FileNotFoundError:
        print("Erro: O arquivo de dados processados não foi encontrado. Execute a Etapa 3 primeiro.")