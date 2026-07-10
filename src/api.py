from fastapi import FastAPI
import xgboost as xgb

# 1. Inicializando a aplicação web
app = FastAPI(
    title="API de Previsão de Atrasos na Aviação",
    description="API para prever atrasos de voos usando XGBoost. Preparação para AWS Lambda.",
    version="1.0.0"
)

# 2. Carregando o modelo treinado globalmente (para não carregar a cada requisição)
model = xgb.XGBClassifier()
model.load_model("models/xgboost_atrasos.json")

# 3. Criando o Endpoint de "Health Check" (Teste de Saúde da API)
@app.get("/")
def health_check():
    """
    Rota principal para verificar se a API está no ar e o modelo carregado.
    """
    return {
        "status": "Operacional",
        "mensagem": "Sistema de previsão de atrasos online.",
        "modelo_carregado": True
    }

# 4. Criando o Endpoint de Previsão (Mock inicial)
@app.post("/predict")
def prever_atraso():
    """
    Rota que receberá os dados do voo no futuro. 
    Por enquanto, retorna uma estrutura validando a conexão.
    """
    return {
        "status_voo": "No Horario",
        "probabilidade_atraso": 0.15,
        "alerta": "Condicoes climaticas favoraveis."
    }