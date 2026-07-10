# Usa a imagem oficial do AWS Lambda para Python 3.12 (Baseada em Amazon Linux)
FROM public.ecr.aws/lambda/python:3.12

# Copia o arquivo de dependências para dentro do container
COPY requirements.txt .

# Instala as bibliotecas exatamente como especificado, forçando compatibilidade com Linux
RUN pip install --no-cache-dir -r requirements.txt

# Copia a pasta do código fonte e a pasta do modelo preditivo
COPY src/ ./src/
COPY models/ ./models/

# Define a variável handler do Mangum como o gatilho de execução da AWS
CMD ["src.api.handler"]