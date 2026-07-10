import requests
import pandas as pd

def fetch_weather_data(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Busca dados climáticos históricos da API Open-Meteo e converte em um DataFrame Pandas.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Parâmetros exigidos pela documentação da API para buscar dados horários
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation,wind_speed_10m,visibility",
        "timezone": "America/Sao_Paulo"
    }
    
    try:
        # Fazendo a requisição HTTP GET
        response = requests.get(url, params=params)
        response.raise_for_status() 
        
        # Parseando a resposta de JSON para um dicionário Python
        data = response.json()
        
        # Convertendo o nó 'hourly' diretamente para um formato tabular (DataFrame)
        df = pd.DataFrame(data["hourly"])
        
        # Transformando a coluna de tempo em um objeto datetime real
        df["time"] = pd.to_datetime(df["time"])
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Erro crítico na conexão com a API de clima: {e}")
        return pd.DataFrame() # Retorna um Dataframe vazio para nao quebrar a pipeline

if __name__ == "__main__":
    print("Iniciando rotina de extração climática...")
    
    # Coordenadas do Aeroporto Internacional de Brasília (SBBR / BSB)
    lat_bsb = -15.8697
    lon_bsb = -47.9208
    
    # Janela de dados de exemplo para validação do script
    df_clima = fetch_weather_data(lat_bsb, lon_bsb, "2024-01-01", "2024-01-05")
    
    if not df_clima.empty:
        print("\nVisualizando as primeiras 5 linhas dos dados coletados:")
        print(df_clima.head())
        
        # Salvando o teste fisicamente na pasta de dados brutos
        caminho_salvamento = "data/raw/clima_bsb_teste.csv"
        df_clima.to_csv(caminho_salvamento, index=False)
        print(f"\nArquivo salvo com sucesso em: {caminho_salvamento}")
    else:
        print("Falha na coleta. DataFrame vazio retornado.")