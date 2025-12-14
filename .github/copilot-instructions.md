# Instruções de Code Review para Copilot

## Objetivo Geral
Garantir qualidade, segurança, performance, clareza e limpeza de código através de revisões automáticas com 100% de cobertura.

---

## 1. PADRÕES DE QUALIDADE

### 1.1 Type Hints e Anotações
- **OBRIGATÓRIO**: Todos os parâmetros e retornos devem ter type hints
- **PADRÃO**: Usar `from typing import` para tipos complexos (Dict, List, Tuple, Optional, Union)
- **VERIFICAR**: Compatibilidade com Python 3.9+
- **EXEMPLO CORRETO**:
```python
def process_data(
    data: List[float],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Processa dados com threshold."""
    pass
```

### 1.2 Docstrings
- **PADRÃO**: Google Style docstrings em português
- **CONTEÚDO OBRIGATÓRIO**:
  - Descrição clara da função/classe
  - Args: com tipos e descrição
  - Returns: com tipo e descrição
  - Raises: exceções que podem ser levantadas
- **EXEMPLO CORRETO**:
```python
def train_model(
    data: np.ndarray,
    epochs: int = 10
) -> Dict[str, float]:
    """
    Treina o modelo LSTM com os dados fornecidos.
    
    Args:
        data (np.ndarray): Dados de treinamento com shape (n_samples, seq_length, features).
        epochs (int): Número de épocas. Padrão: 10
    
    Returns:
        Dict[str, float]: Dicionário com histórico de perda por época.
    
    Raises:
        ValueError: Se data estiver vazia ou epochs <= 0.
    """
    pass
```

### 1.3 Nomes e Convenções
- **VARIÁVEIS**: snake_case (ex: `model_weights`, `learning_rate`)
- **CLASSES**: PascalCase (ex: `LSTMModel`, `ModelTrainer`)
- **CONSTANTES**: UPPER_SNAKE_CASE (ex: `MAX_EPOCHS`, `BATCH_SIZE`)
- **PRIVADAS**: Prefixo underscore (ex: `_internal_method`)
- **DESCRITIVOS**: Nomes claros que indicam propósito

### 1.4 Comprimento de Linhas
- **MÁXIMO**: 100 caracteres
- **EXCEÇÕES**: URLs, paths, strings longas (comentar por quê)

---

## 2. SEGURANÇA

### 2.1 Tratamento de Erros
- **OBRIGATÓRIO**: Try/except com exceções específicas
- **PROIBIDO**: Bare `except:` ou `except Exception:`
- **PADRÃO**:
```python
try:
    result = expensive_operation()
except FileNotFoundError as e:
    logger.error(f"Arquivo não encontrado: {e}")
    raise
except ValueError as e:
    logger.warning(f"Valor inválido: {e}")
    return None
```

### 2.2 Validação de Entrada
- **OBRIGATÓRIO**: Validar todos os inputs de usuário
- **VERIFICAR**: Ranges, tipos, valores nulos
- **EXEMPLO**:
```python
def save_model(model: torch.nn.Module, path: str) -> None:
    """Salva o modelo."""
    if not path:
        raise ValueError("Path não pode estar vazio")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model deve ser nn.Module")
```

### 2.3 Secrets e Credenciais
- **PROIBIDO**: Hardcoded secrets, passwords, API keys
- **OBRIGATÓRIO**: Usar variáveis de ambiente ou .env
- **VERIFICAR**: Arquivo `.gitignore` contém `.env`

### 2.4 Dependências
- **VERIFICAR**: Bibliotecas desatualizadas com vulnerabilidades
- **PADRÃO**: Usar versões pinadas no requirements.txt
- **EXEMPLO**:
```
torch==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
```

---

## 3. PERFORMANCE

### 3.1 Operações Vectorizadas
- **PREFERIR**: NumPy/PyTorch operations over loops
- **EVITAR**: Loops Python em operações numéricas
- **EXEMPLO RUIM**: `[x * 2 for x in array]`
- **EXEMPLO BOM**: `array * 2`

### 3.2 Gerenciamento de Memória
- **VERIFICAR**: Leaks de memória em loops
- **USAR**: Context managers (`with` statement)
- **EXEMPLO**:
```python
with torch.no_grad():
    predictions = model(data)
# Libera memória do gradiente
```

### 3.3 Operações em GPU
- **PADRÃO**: Mover tensores para device corretamente
- **VERIFICAR**: Mixing CPU/GPU tensors
- **EXEMPLO**:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

### 3.4 Caching e Lazy Evaluation
- **USAR**: `@lru_cache` para funções puras
- **VERIFICAR**: Computações desnecessárias repetidas

---

## 4. CLAREZA DO CÓDIGO

### 4.1 Estrutura e Organização
- **ORDEM**: Imports → Constants → Classes → Functions → Main
- **MODULES**: Um responsabilidade por arquivo
- **VERIFICAR**: Máximo 500 linhas por arquivo (considerar refactor)

### 4.2 Comentários
- **POR QUÊ**: Explicar decisões de design, não o óbvio
- **EVITAR**: Comentários que duplicam o código
- **EXEMPLO BOM**: `# Reshape necessário pois LSTM espera (batch, seq, features)`
- **EXEMPLO RUIM**: `# Incrementa x em 1` (para `x += 1`)

### 4.3 Funções Pequenas
- **MÁXIMO**: 20 linhas por função (complexas: máximo 30)
- **PADRÃO**: Função = uma responsabilidade
- **VERIFICAR**: Funções testáveis isoladamente

### 4.4 Magic Numbers
- **PROIBIDO**: Números sem contexto no código
- **OBRIGATÓRIO**: Definir como constantes nomeadas
- **EXEMPLO**:
```python
# Ruim
model = LSTMModel(50, 100)

# Bom
HIDDEN_SIZE = 50
OUTPUT_SIZE = 100
model = LSTMModel(HIDDEN_SIZE, OUTPUT_SIZE)
```

---

## 5. LIMPEZA DO CÓDIGO

### 5.1 Imports
- **ORDEM**: Standard library → Third-party → Local
- **REGRA**: Imports não utilizados devem ser removidos
- **VERIFICAR**: Imports duplicados
- **EXEMPLO**:
```python
import os
import sys
from typing import List

import torch
import numpy as np

from src.models import LSTMModel
```

### 5.2 Variáveis Não Utilizadas
- **PROIBIDO**: Variáveis declaradas e nunca usadas
- **VERIFICAR**: Parametros não usados em funções
- **SOLUÇÃO**: Usar `_` para ignorar explicitamente
```python
for _, batch in enumerate(dataloader):  # Ignora índice
    pass
```

### 5.3 Código Duplicado
- **REGRA DRY**: Don't Repeat Yourself
- **VERIFICAR**: Blocos de código similares
- **SOLUÇÃO**: Extrair para função reutilizável

### 5.4 Formatação
- **PADRÃO**: Black formatter ou autopep8
- **VERIFICAR**: Espaçamento, indentação (4 spaces)
- **LINHAS EM BRANCO**: Máximo 2 consecutivas

### 5.5 Trailing Whitespace
- **PROIBIDO**: Espaços/tabs no final de linhas
- **FERRAMENTAS**: Pre-commit hooks

---

## 6. TESTES

### 6.1 Cobertura
- **MÍNIMO**: 90% code coverage (ideal: 100%)
- **FERRAMENTA**: pytest-cov
- **COMANDO**: `pytest --cov=src tests/`

### 6.2 Nomenclatura de Testes
- **PADRÃO**: `test_<função>_<cenário>`
- **EXEMPLO**: `test_calculate_metrics_with_perfect_prediction`

### 6.3 Estrutura AAA
```python
def test_something():
    # Arrange - prepara dados
    data = prepare_test_data()
    
    # Act - executa função
    result = function_under_test(data)
    
    # Assert - verifica resultado
    assert result == expected_value
```

### 6.4 Testes Devem Ser
- **INDEPENDENTES**: Não depender de ordem de execução
- **DETERMINÍSTICOS**: Mesmo resultado sempre
- **RÁPIDOS**: Completar em < 1 segundo
- **ISOLADOS**: Usar mocks/fixtures para dependências externas

---

## 7. DOCUMENTAÇÃO

### 7.1 README
- **CONTEÚDO**: Descrição, instalação, uso, exemplos
- **ATUALIZADO**: Sincronizado com código atual

### 7.2 Exemplos
- **CÓDIGO**: Exemplos funcionais no `main` ou docstrings
- **ATUALIZADOS**: Testados e funcionando

### 7.3 CHANGELOG
- **PADRÃO**: Keep a Changelog (keepachangelog.com)
- **SEÇÕES**: Added, Changed, Deprecated, Removed, Fixed, Security

---

## 8. CHECKLIST DE REVIEW

### Antes de Merge
- [ ] Todos os testes passam (`pytest`)
- [ ] Coverage >= 90% (`pytest --cov`)
- [ ] Sem warnings de linter (`pylint`, `flake8`)
- [ ] Code formatted (`black --check`)
- [ ] Type hints presentes (`mypy`)
- [ ] Docstrings completas
- [ ] Sem secrets/credentials
- [ ] Performance aceitável
- [ ] Documentação atualizada
- [ ] CHANGELOG atualizado

### Commands
```bash
# Executar todos os testes
pytest tests/ -v

# Coverage completo
pytest tests/ --cov=src --cov-report=html

# Linting
pylint src/ tests/
flake8 src/ tests/

# Type checking
mypy src/

# Formatação
black src/ tests/
```

---

## 9. BOAS PRÁTICAS ESPECÍFICAS DO PROJETO

### 9.1 Modelos PyTorch
- **SEMPRE**: Incluir device management
- **VERIFICAR**: Modes (train/eval)
- **EXEMPLO**:
```python
model.train()  # Para treinamento
model.eval()   # Para avaliação
with torch.no_grad():  # Sem gradientes em avaliação
    predictions = model(data)
```

### 9.2 Data Loading
- **USAR**: torch.utils.data.DataLoader
- **VERIFICAR**: Batch sizes, shuffling, num_workers
- **EXEMPLO**:
```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### 9.3 Logging
- **NÃO USAR**: `print()` em produção
- **USAR**: `logging` module
- **EXEMPLO**:
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Treinamento iniciado para {epochs} épocas")
```

### 9.4 MLflow Integration
- **VERIFICAR**: Todos os experimentos logados
- **PADRÃO**: Parameters, metrics e artifacts
- **EXEMPLO**:
```python
mlflow.log_params({"epochs": 50, "lr": 0.001})
mlflow.log_metrics({"loss": 0.5})
mlflow.log_artifact("model.pth")
```

---

## 10. ROTINA DE REVIEW

1. **Executar testes automaticamente** na CI/CD
2. **Verificar coverage** - deve estar >= 90%
3. **Rodar linters** - sem warnings
4. **Type checking** - sem erros
5. **Code review manual** - seguindo checklist acima
6. **Performance** - benchmarks se relevante
7. **Documentação** - atualizada e clara
8. **Security** - sem vulnerabilidades conhecidas

---

## Referências
- [PEP 8 - Style Guide for Python](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PyTorch Best Practices](https://pytorch.org/docs/stable/notes/programming_practices.html)
- [The Twelve-Factor App](https://12factor.net/)
