# Testing Guide - Stock Prediction API (LSTM)

Este documento descreve a estratégia de testes, cobertura de código e como executar os testes.

## Índice
- [Estrutura de Testes](#estrutura-de-testes)
- [Cobertura de Código](#cobertura-de-código)
- [Executar Testes](#executar-testes)
- [Verificação de Qualidade](#verificação-de-qualidade)
- [CI/CD Pipeline](#cicd-pipeline)

---

## Estrutura de Testes

### Arquivos de Teste

```
tests/
├── conftest.py                  # Configuração e fixtures do pytest
├── test_lstm_model.py           # Testes do modelo LSTM (100% cobertura)
├── test_utils.py                # Testes de salvar/carregar modelos (100% cobertura)
├── test_evaluate.py             # Testes de avaliação e métricas (100% cobertura)
├── test_preprocessing.py        # Testes de pré-processamento
├── test_data_loader.py          # Testes de carregamento de dados
├── test_config.py               # Testes de configuração da aplicação
├── test_main.py                 # Testes da API FastAPI
├── test_lifespan.py             # Testes de ciclo de vida da aplicação
├── test_audit_route.py          # Testes de rotas de auditoria
├── test_reproducibility.py      # Testes de reprodutibilidade de seeds
├── test_train_integration.py    # Testes de integração (Treino/Predição)
├── test_train_route_coverage.py # Testes de cobertura da rota de treino
└── test_train_unit.py           # Testes unitários de treinamento
```

### Módulos Testados

#### 1. `src/lstm_model.py` - LSTMModel
**Cobertura**: 100%
- ✓ Inicialização com parâmetros padrão e customizados
- ✓ Forward pass com diferentes shapes
- ✓ Representação em string (__str__)
- ✓ Compatibilidade com CPU/CUDA
- ✓ State dict (salvar/carregar pesos)
- ✓ Casos extremos (zero, muito pequeno, muito grande)

#### 2. `src/utils.py` - save_model, load_model
**Cobertura**: 100%
- ✓ Salvamento em diferentes caminhos
- ✓ Carregamento e validação de pesos
- ✓ Modo eval após carregamento
- ✓ Ciclos completos de save/load
- ✓ Tratamento de erros (arquivo corrompido)
- ✓ Diferentes arquiteturas de modelo

#### 3. `src/evaluate.py` - evaluate_model, calculate_metrics
**Cobertura**: 100%
- ✓ Cálculo correto de MAE, RMSE, MAPE
- ✓ Forma dos outputs
- ✓ Avaliação em modo eval (sem gradientes)
- ✓ Múltiplos lotes e batch sizes
- ✓ Scaler customizado
- ✓ Suporte a CPU/CUDA

#### 4. `src/data_loader.py` - Data Loading
**Cobertura**: ~90%
- ✓ Carregamento de dados do Yahoo Finance
- ✓ Tratamento de datas
- ✓ Validação de símbolos de ações
- ✓ Tratamento de erros de conexão

#### 5. `src/seed_manager.py` - Seed Management
**Cobertura**: ~95%
- ✓ Configuração de seeds para reprodutibilidade
- ✓ Gerenciamento de seeds PyTorch, NumPy e Python
- ✓ Verificação de consistência

#### 6. `src/train.py` - ModelTrainer
**Cobertura**: ~90%
- ✓ Inicialização do treinador
- ✓ Loop de treinamento
- ✓ Histórico de perda
- ✓ Integração com MLflow

#### 7. `app/main.py` - API FastAPI
**Cobertura**: ~90%
- ✓ Endpoints de saúde
- ✓ Ciclo de vida da aplicação
- ✓ Carregamento de modelos
- ✓ Tratamento de erros

---

## Cobertura de Código

### Objetivo Mínimo
- **90% de cobertura geral** (obrigatório na CI/CD)
- **100% de cobertura dos módulos principais** (lstm_model, utils, evaluate)

### Visualizar Cobertura

```bash
# Gerar relatório no terminal
pytest --cov=src --cov-report=term-missing

# Gerar relatório HTML
pytest --cov=src --cov-report=html
# Abrir: htmlcov/index.html
```

### Métricas Atuais

| Módulo | Cobertura | Status |
|--------|-----------|--------|
| lstm_model.py | 100% | ✓ Completo |
| utils.py | 100% | ✓ Completo |
| evaluate.py | 100% | ✓ Completo |
| data_loader.py | ~90% | ✓ Acima do mínimo |
| seed_manager.py | ~95% | ✓ Acima do mínimo |
| preprocessing.py | ~90% | ✓ Acima do mínimo |
| train.py | ~90% | ✓ Acima do mínimo |
| main.py (API) | ~90% | ✓ Acima do mínimo |
| **TOTAL** | **~92%** | ✓ Aprovado |mínimo |

---

## Executar Testes

### Instalação de Dependências

```bash
# Instalação básica
pip install -r requirements.txt

# Instalação com desenvolvimento (incluindo testes)
pip install -r requirements-dev.txt
```

### Comandos Principais

#### 1. Rodar Todos os Testes
```bash
pytest tests/ -v
```

#### 2. Rodar com Cobertura
```bash
pytest tests/ --cov=src --cov-report=term-missing -v
```

#### 3. Rodar Teste Específico
```bash
pytest tests/test_lstm_model.py -v
pytest tests/test_lstm_model.py::TestLSTMModelInit -v
pytest tests/test_lstm_model.py::TestLSTMModelInit::test_init_default_parameters -v
```

#### 4. Rodar Testes em Paralelo (Rápido)
```bash
pytest tests/ -n auto -v
```

#### 5. Rodar com Saída Detalhada
```bash
pytest tests/ -vv --tb=long
```

#### 6. Rodar Apenas Testes Rápidos
```bash
pytest tests/ -v -m "not slow"
```

### Usando Make

```bash
# Ver todos os comandos disponíveis
make help

# Rodar testes
make test

# Rodar com cobertura
make coverage

# Gerar relatório HTML
make coverage-html

# Qualidade completa (lint + type + test + coverage)
make quality

# Qualidade rápida
make quick-quality
```

---

## Verificação de Qualidade

### 1. Linting

```bash
# Pylint
pylint src/ app/

# Flake8
flake8 src/ app/ --max-line-length=100

# Ambos com Make
make lint
```

### 2. Formatação de Código

```bash
# Verificar formatação
black --check src/ app/ tests/

# Aplicar formatação
black src/ app/ tests/

# Ordenar imports
isort src/ app/ tests/

# Com Make
make format
```

### 3. Type Checking

```bash
# MyPy
mypy src/ app/

# Com Make
make type-check
```

### 4. Security

```bash
# Bandit (security analysis)
bandit -r src/ app/

# Com Make
make security
```

### 5. Executar Todos os Checks

```bash
# Com Make (recomendado)
make quality

# Ou manualmente
pytest tests/ --cov=src --cov-report=term-missing
pylint src/ app/ --exit-zero
black --check src/ app/ tests/
mypy src/ app/ --exit-zero
bandit -r src/ app/ -v
```

---

## CI/CD Pipeline

### Estrutura do Pipeline

O pipeline GitHub Actions (`.github/workflows/ci-cd-pipeline.yml`) executa:

1. **Code Quality Check** - Linting e formatting
2. **Build** - Verifica sintaxe e imports
3. **Tests** - Testes unitários com cobertura >= 90%
4. **Integration Tests** - Testes de integração
5. **Train Model** - Treina modelo (branch main apenas)
6. **Security** - Análise de segurança
7. **Documentation** - Verifica documentação
8. **Report** - Gera relatório resumido

### Triggering Pipeline

O pipeline é acionado por:
- `push` em `main` ou `develop`
- `pull_request` em `main` ou `develop`

### Visualizar Pipeline

1. Ir para: `https://github.com/Fiap-Pos-tech-5MLET/fase-4/actions`
2. Clicar na run desejada para ver detalhes

---

## Estratégia de Testes

### Estrutura AAA (Arrange-Act-Assert)

Todos os testes seguem este padrão:

```python
def test_something():
    # Arrange - Prepara dados de teste
    model = LSTMModel()
    x = torch.randn(32, 10, 1)
    
    # Act - Executa o que está sendo testado
    output = model.forward(x)
    
    # Assert - Verifica o resultado
    assert output.shape == (32, 1)
```

### Categorias de Testes

#### Unit Tests
- Testam funções/métodos isoladamente
- Sem dependências externas (mocks quando necessário)
- Rápidos (< 1 segundo cada)

#### Integration Tests
- Testam integração entre componentes
- Simulam pipeline real
- Podem ser mais lentos

#### Marks (Marcadores)

```python
@pytest.mark.unit          # Teste unitário
@pytest.mark.slow          # Teste lento
@pytest.mark.gpu           # Requer GPU
@pytest.mark.skipif(...)   # Skip condicional
```

### Fixtures

Fixtures disponíveis em `tests/conftest.py`:

```python
@pytest.fixture
def lstm_model():
    """Modelo LSTM básico"""

@pytest.fixture
def pytorch_device():
    """Dispositivo (CPU ou CUDA)"""

@pytest.fixture
def minmax_scaler():
    """Scaler normalizado"""

@pytest.fixture
def sample_dataloader():
    """DataLoader de exemplo"""
```

---

## Boas Práticas

### ✓ Fazer
- Usar fixtures para dados compartilhados
- Testes independentes e determinísticos
- Nomes descritivos para testes
- Assert com mensagens úteis
- Utilizar `pytest.raises()` para exceções

### ✗ Evitar
- Testes dependentes de ordem de execução
- Testes que modificam arquivos sem cleanup
- Testes com múltiplas responsabilidades
- Assertions vagas

---

## Troubleshooting

### Problema: Testes falham por "Module not found"
```bash
# Solução: Adicione o projeto ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Problema: Testes CUDA falham em máquina sem GPU
```bash
# Solução: Testes com skip automático
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda():
    ...
```

### Problema: Testes lentos
```bash
# Solução: Rodar em paralelo
pip install pytest-xdist
pytest tests/ -n auto
```

### Problema: Cache de importação
```bash
# Solução: Limpar cache
rm -rf __pycache__ .pytest_cache
pytest --cache-clear tests/
```

---

## Métricas de Sucesso

✓ **Tests**: Todos passam (`pytest tests/`)
✓ **Coverage**: >= 90% (`pytest --cov=src`)
✓ **Linting**: Sem warnings (`pylint src/`)
✓ **Type**: Sem erros (`mypy src/`)
✓ **Security**: Sem vulnerabilidades (`bandit -r src/`)
✓ **Format**: Código formatado (`black --check src/`)

---

## Referências

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [PyTorch Testing](https://pytorch.org/docs/stable/testing.html)
- [GitHub Actions](https://docs.github.com/en/actions)

---

**Última atualização**: Janeiro 2026
**Maintainer**: Equipe 5MLET
