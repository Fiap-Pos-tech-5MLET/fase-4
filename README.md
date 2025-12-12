# Previsão de Preços de Ações com LSTM

## Visão e Estratégia
Este projeto foca no uso de Deep Learning para prever tendências do mercado de ações, especificamente preços de fechamento. Utilizando redes neurais LSTM (Long Short-Term Memory), eficazes para dados sequenciais, buscamos capturar padrões temporais muitas vezes perdidos por modelos tradicionais. A solução é completa (end-to-end), cobrindo desde a ingestão de dados e treinamento até o deploy via uma API escalável.

## Arquitetura do Projeto
O sistema é construído sobre uma arquitetura modular:

1.  **Camada de Dados**: Recuperação automática de dados históricos do Yahoo Finance.
2.  **Camada de Processamento**: Normalização e criação de sequências para previsão de séries temporais.
3.  **Camada de Modelo**: Rede neural LSTM baseada em PyTorch.
4.  **Camada de Serviço**: Aplicação FastAPI expondo endpoints de previsão e treinamento.
5.  **Infraestrutura**: Ambiente dockerizado para deploys reprodutíveis.

## Estrutura de Diretórios
```
c:\Projetos\Leonardo\PosTech\Fase4\TechChallenge\Antigravity\
│
├── app/
│   ├── api/
│   │   └── main.py          # Ponto de entrada da aplicação FastAPI
│   ├── data/
│   │   └── data_loader.py   # Lógica de busca e pré-processamento de dados
│   ├── model/
│   │   └── lstm_model.py    # Definição da Rede Neural LSTM
│   └── train.py             # Orquestração do pipeline de treinamento
│
├── docker/                  # Configurações Docker (separação opcional)
├── docker-compose.yml       # Orquestração de contêineres
├── Dockerfile               # Definição da imagem da aplicação
├── requirements.txt         # Dependências Python
└── README.md                # Documentação do projeto
```

## Pré-requisitos
- **Python**: 3.9 ou superior
- **Docker** (opcional para execução em contêiner)
- **Git**

## Configuração e Instalação

### Opção A: Ambiente Python Local
1.  **Clone o repositório** (se aplicável).
2.  **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Treine o Modelo**:
    Antes de rodar a API, você deve treinar o modelo inicial para gerar os artefatos.
    ```bash
    python -m app.train
    ```
    *Isso salvará `lstm_model.pth` e `scaler.pkl` em `app/artifacts/`.*

4.  **Execute a API**:
    ```bash
    python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
    ```

### Opção B: Contêineres Docker
1.  **Construa e Execute**:
    ```bash
    docker-compose up --build
    ```
    A API estará disponível em `http://localhost:8000`.

## Documentação da API

### 1. Verificação de Saúde (Health Check)
*   **GET** `/health`
*   **Descrição**: Verifica se a API está rodando e se o modelo está carregado.
*   **Resposta**: `{"status": "ok", "model_loaded": true}`

### 2. Prever Preço de Ação
*   **POST** `/predict`
*   **Descrição**: Prevê o próximo preço de fechamento com base nos últimos 60 dias de dados.
*   **Corpo (Body)**:
    ```json
    {
      "last_60_days_prices": [150.1, 151.0, ..., 155.4] // Deve conter exatamente 60 números (floats)
    }
    ```
*   **Resposta**: `{"predicted_price": 156.2}`

### 3. Disparar Treinamento
*   **POST** `/train`
*   **Descrição**: Dispara um job de treinamento em segundo plano para um símbolo de ação específico.
*   **Corpo (Body)** (valores padrão opcionais mostrados):
    ```json
    {
      "symbol": "AAPL",
      "start_date": "2018-01-01",
      "end_date": "2024-07-20",
      "epochs": 50
    }
    ```
*   **Resposta**: `{"message": "Training started in background", ...}`

## Detalhes Técnicos
- **Framework**: PyTorch
- **Arquitetura do Modelo**:
    - **Entrada**: Sequência de 60 dias (Preços de Fechamento).
    - **Camada Oculta**: LSTM com 50 unidades.
    - **Saída**: Camada linear projetando para 1 valor (Fechamento Previsto).
- **Scaler**: MinMaxScaler (0, 1) para normalizar os dados de entrada para um gradiente descendente estável.

## Monitoramento e MLflow
O projeto utiliza **MLflow** para rastreamento de experimentos. Todas as execuções de treinamento (parâmetros, métricas e modelos) são registradas automaticamente.

Para visualizar o painel do MLflow:
```bash

mlflow ui --host 0.0.0.0 --port 5001

```
Acesse `http://localhost:5001` no seu navegador.

A aplicação também registra o progresso do treinamento (Perda/Loss por época) e requisições da API na saída padrão (stdout).
