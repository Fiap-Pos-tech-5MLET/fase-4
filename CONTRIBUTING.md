# Guia de Contribuição

Obrigado pelo interesse em contribuir para o projeto **Tech Challenge Fase 4 - Stock Prediction API**!

Este projeto faz parte do **Tech Challenge da Pós-Graduação em Machine Learning Engineering da FIAP (Pos Tech)**.
Este documento estabelece as diretrizes detalhadas para garantir que o projeto se mantenha organizado, seguro, testado e com alta qualidade de código.

## � Índice

1.  [�🚀 Como Começar](#-como-começar)
2.  [🛠 Padrões de Código](#-padrões-de-código)
3.  [🔒 Segurança e Boas Práticas](#-segurança-e-boas-práticas)
4.  [🧠 Desenvolvimento com PyTorch](#-desenvolvimento-com-pytorch)
5.  [✅ Testes](#-testes)
6.  [📦 Workflow de Submissão](#-workflow-de-submissão)
7.  [⌨️ Comandos do Makefile](#️-comandos-do-makefile)

---

## 🚀 Como Começar

### 1. Configuração do Ambiente

O projeto utiliza **Python 3.11+**. Recomendamos fortemente o uso de ambientes virtuais (`venv`).

```bash
# 1. Clone o repositório
git clone https://github.com/Fiap-Pos-tech-5MLET/fase-4.git
cd fase-4

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instale todas as dependências (Dev + Prod)
make install-dev
```

> **Nota**: O comando `make install-dev` é crucial para ter acesso a ferramentas como `pylint`, `mypy` e `pytest`.

### 2. Estrutura de Branches

Utilizamos um fluxo baseado no **Git Flow**:

*   `main`: Código de produção estável.
*   `develop`: Branch principal de desenvolvimento. Todos os PRs devem apontar para cá.
*   `feature/nome-da-feature`: Para novas funcionalidades.
*   `fix/nome-do-bug`: Para correções de bugs.
*   `hotfix/nome-do-erro`: Para correções críticas diretas na main (raro).

---

## 🛠 Padrões de Código

Mantemos um alto padrão de qualidade automatizado. Consulte também [.github/copilot-instructions.md](.github/copilot-instructions.md) para ver as regras que nossa IA de Code Review segue.

### 1. Estilo e Formatação
*   **Formatador**: Utilizamos `Black` e `isort`.
*   **Linter**: `Flake8` e `Pylint`.
*   **Comprimento de Linha**: Máximo de **100 caracteres**.

### 2. Nomenclatura
Siga as convenções do Python (PEP 8):
*   `snake_case`: Variáveis, funções, métodos, módulos (`learning_rate`, `train_model`).
*   `PascalCase`: Classes (`LSTMModel`, `DataProcessor`).
*   `UPPER_SNAKE_CASE`: Constantes (`BATCH_SIZE`, `MAX_EPOCHS`).

### 3. Type Hints (Tipagem Estática)
**Obrigatório**. Todas as assinaturas de funções e métodos devem ter anotações de tipo para argumentos e retorno.

```python
# ✅ Correto
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    ...

# ❌ Incorreto
def calculate_metrics(y_true, y_pred):
    ...
```

### 4. Docstrings
**Obrigatório**. Utilizamos o **Google Style** em **Português**.
Toda classe, módulo e função pública deve ter docstring explicando:
*   O que faz.
*   Argumentos (`Args`).
*   Retorno (`Returns`).
*   Exceções (`Raises`).

```python
def train_model(data: DataLoader, epochs: int = 10) -> List[float]:
    """
    Treina o modelo LSTM com os dados fornecidos.

    Args:
        data (DataLoader): Loader contendo os batches de treinamento.
        epochs (int): Número de épocas para treinar. Padrão: 10.

    Returns:
        List[float]: Lista contendo o valor da perda (loss) por época.
    
    Raises:
        ValueError: Se o número de épocas for menor que 1.
    """
    if epochs < 1:
        raise ValueError("Epochs deve ser >= 1")
    ...
```

---

## 🔒 Segurança e Boas Práticas

### 1. Gestão de Segredos
*   **NUNCA** commite senhas, chaves de API ou tokens.
*   Utilize variáveis de ambiente carregadas via `python-dotenv`.
*   Verifique se o `.gitignore` contém `.env`.

### 2. Logging vs Print
*   Em código de produção (`src/`, `app/`), **evite `print()`**.
*   Utilize o módulo `logging` padrão do Python.
*   `print()` é aceitável apenas em scripts de CLI ou notebooks de teste.

```python
import logging
logger = logging.getLogger(__name__)

# ✅ Correto
logger.info("Iniciando treinamento...")
logger.error(f"Erro ao carregar dados: {e}")
```

### 3. Tratamento de Erros
*   Use `try/except` com exceções específicas (`ValueError`, `FileNotFoundError`).
*   Evite `except Exception:` genérico, pois mascara erros inesperados.

---

## 🧠 Desenvolvimento com PyTorch

Ao trabalhar com modelos de Deep Learning (`src/lstm_model.py`):

1.  **Gerenciamento de Dispositivo**: O código deve ser agnóstico a CPU/GPU.
    ```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    ```

2.  **Modos de Operação**:
    *   Use `model.train()` antes de loops de treinamento.
    *   Use `model.eval()` antes de avaliação/inferência.

3.  **Gerenciamento de Memória**: Use `with torch.no_grad():` durante inferência para economizar memória.

---

## ✅ Testes

Para detalhes completos, consulte [TESTING.md](TESTING.md).

**Regra de Ouro**: Não aceitamos PRs que diminuam a cobertura de testes para menos de **90%**.

### Executando Testes

```bash
# Rodar todos os testes
make test

# Rodar com relatório de cobertura (HTML e Terminal)
make coverage

# Rodar teste específico
pytest tests/test_lstm_model.py
```

---

## � Workflow de Submissão

1.  **Crie sua branch** a partir de `develop`:
    `git checkout -b feature/minha-nova-feature`
2.  **Desenvolva e Code**.
3.  **Rode a Verificação de Qualidade Local**:
    Este comando executa Linting, Type Checking, Security Check e Testes.
    ```bash
    make quality
    ```
    > **Importante**: Se este comando falhar, seu PR provavelmente será rejeitado pelo CI.
4.  **Commit e Push**:
    `git push origin feature/minha-nova-feature`
5.  **Abra o Pull Request**:
    *   Descreva suas mudanças detalhadamente.
    *   Vincule a Issues se houver.
6.  **Code Review**:
    *   Aguarde a revisão automática do Copilot.
    *   Aguarde a aprovação de um mantenedor.

---

## ⌨️ Comandos do Makefile

Use o `make` para automatizar tarefas repetitivas.

| Comando | Descrição |
|---------|-----------|
| `make install-dev` | Instala dependências completas |
| `make format` | Formata o código (Black + Isort) |
| `make lint` | Verifica problemas de estilo (Pylint + Flake8) |
| `make type-check` | Verifica tipos estáticos (MyPy) |
| `make security` | Verifica vulnerabilidades (Bandit) |
| `make test` | Roda testes unitários |
| `make coverage` | Roda testes com relatório de cobertura |
| `make quality` | **Verificação completa** (Lint + Type + Security + Test + Cover) |
| `make run-api` | Inicia a API FastAPI (porta 8000) |
| `make run-streamlit` | Inicia o Dashboard (porta 8501) |
| `make clean` | Limpa arquivos temporários e cache |

---

Dúvidas? Consulte a [Wiki](https://github.com/Fiap-Pos-tech-5MLET/fase-4/wiki) ou abra uma Issue!
