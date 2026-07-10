import pandas as pd

def create_mock_anac_data() -> pd.DataFrame:
    """
    Gera uma amostra espelhando o formato VRA da ANAC para o Aeroporto de Brasília (BSB).
    """
    data = {
        "id_voo": ["G3-1001", "LA-2002", "AD-3003", "G3-1004", "LA-2005"],
        "cia_aerea": ["Gol", "LATAM", "Azul", "Gol", "LATAM"],
        "origem": ["SBRJ", "CGH", "CNF", "SBRJ", "CGH"],
        "destino": ["SBBR", "SBBR", "SBBR", "SBBR", "SBBR"], # Chegando em Brasília
        "chegada_prevista": ["2024-01-01 01:15:00", "2024-01-01 02:45:00", "2024-01-01 03:10:00", "2024-01-01 04:05:00", "2024-01-01 04:50:00"],
        "chegada_real": ["2024-01-01 01:20:00", "2024-01-01 03:45:00", "2024-01-01 03:12:00", "2024-01-01 04:55:00", "2024-01-01 04:50:00"]
    }
    df = pd.DataFrame(data)
    df["chegada_prevista"] = pd.to_datetime(df["chegada_prevista"])
    df["chegada_real"] = pd.to_datetime(df["chegada_real"])
    return df

def process_and_merge_data(df_voos: pd.DataFrame, df_clima: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a variável Target (Atraso) e cruza as tabelas de voo e clima.
    """
    # 1. Engenharia do Target: Calculando o atraso em minutos
    df_voos["atraso_minutos"] = (df_voos["chegada_real"] - df_voos["chegada_prevista"]).dt.total_seconds() / 60
    
    # 2. Regra de Negócio: Consideramos "Atraso" (1) se for maior que 15 minutos. Caso contrário, (0).
    df_voos["target_atraso"] = (df_voos["atraso_minutos"] > 15).astype(int)
    
    # 3. Preparando a chave do JOIN: Arredondamos a hora do voo para cruzar com a hora cheia do clima
    df_voos["hora_arredondada"] = df_voos["chegada_prevista"].dt.floor("h")
    
    # 4. Cruzamento dos dados (Left Join)
    df_merged = pd.merge(
        df_voos, 
        df_clima, 
        left_on="hora_arredondada", 
        right_on="time", 
        how="left"
    )
    
    # Removendo colunas redundantes após o join para deixar o dataset limpo para o XGBoost
    df_merged.drop(columns=["hora_arredondada", "time", "atraso_minutos", "chegada_real"], inplace=True)
    
    return df_merged

if __name__ == "__main__":
    print("Iniciando pipeline de processamento...")
    
    try:
        # Carregando o arquivo gerado na Etapa 2
        df_clima = pd.read_csv("data/raw/clima_bsb_teste.csv")
        df_clima["time"] = pd.to_datetime(df_clima["time"])
    except FileNotFoundError:
        print("Erro: clima_bsb_teste.csv não encontrado. Rode a Etapa 2 primeiro.")
        exit()
        
    df_voos = create_mock_anac_data()
    df_final = process_and_merge_data(df_voos, df_clima)
    
    print("\nVisualizando o Dataset Final Preparado:")
    print(df_final[["id_voo", "target_atraso", "precipitation", "wind_speed_10m"]].head())
    
    # Salvando na pasta de dados processados
    caminho_salvo = "data/processed/dataset_modelo.csv"
    df_final.to_csv(caminho_salvo, index=False)
    print(f"\nDataset pronto para Machine Learning salvo em: {caminho_salvo}")