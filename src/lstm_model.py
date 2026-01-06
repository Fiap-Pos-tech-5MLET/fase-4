import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Modelo de Rede Neural LSTM para predição de sequências.

    Esta classe implementa uma rede neural com camada LSTM seguida por uma camada linear
    para realizar predições em dados sequenciais.

    Atributos:
        hidden_layer_size (int): Número de unidades na camada LSTM.
        lstm (nn.LSTM): Camada LSTM da rede neural.
        linear (nn.Linear): Camada linear para transformação da saída LSTM.
    """

    def __init__(self, input_size: int = 1, hidden_layer_size: int = 50, output_size: int = 1, num_layers: int = 2, dropout: float = 0.2) -> None:
        """
        Inicializa o modelo LSTM.

        Args:
            input_size (int): Número de características de entrada. Padrão: 1
            hidden_layer_size (int): Número de unidades na camada LSTM. Padrão: 50
            output_size (int): Número de características de saída. Padrão: 1
            num_layers (int): Número de camadas LSTM empilhadas. Padrão: 2
            dropout (float): Probabilidade de dropout (se num_layers > 1). Padrão: 0.2
        """
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Executa a propagação forward da rede neural.

        Args:
            input_seq (torch.Tensor): Tensor de entrada com shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Tensor de predições com shape (batch_size, output_size).
        """
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

    def __str__(self) -> str:
        """
        Retorna uma representação em string do modelo.

        Returns:
            str: Descrição do modelo LSTM com seus componentes.
        """
        return (
            f"LSTMModel(\n"
            f"  hidden_layer_size={self.hidden_layer_size},\n"
            f"  lstm={self.lstm},\n"
            f"  linear={self.linear}\n"
            f")"
        )


if __name__ == "__main__":
    # Teste de instanciação do modelo
    model = LSTMModel()
    print(model)