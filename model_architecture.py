#!/usr/bin/env python3
"""
Arquitectura de Transformer Espacio-Temporal para predicción de deformación sísmica.
Utiliza InSAR data para modelar patrones espacio-temporales de deformación.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

class SpatialTemporalTransformer(nn.Module):
    """
    Transformer Espacio-Temporal para predicción de deformación del terreno.
    Combina atención espacial y temporal con mecanismo de atención dispersa.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 seq_length: int = 30, grid_size: Tuple[int, int] = (50, 50),
                 num_classes: int = 3, task_type: str = 'regression'):
        """
        Inicializa el modelo Transformer Espacio-Temporal.

        Args:
            d_model: Dimensión del modelo
            nhead: Número de cabezas de atención
            num_encoder_layers: Número de capas del encoder
            dim_feedforward: Dimensión de la feedforward network
            dropout: Tasa de dropout
            seq_length: Longitud de la secuencia temporal
            grid_size: Tamaño del grid espacial (H, W)
            num_classes: Número de clases para clasificación
            task_type: 'regression' o 'classification'
        """
        super().__init__()

        self.d_model = d_model
        self.seq_length = seq_length
        self.grid_size = grid_size
        self.task_type = task_type
        self.num_patches = grid_size[0] * grid_size[1]

        # Embedding espacial: convertir cada píxel en un vector
        self.spatial_embedding = nn.Linear(1, d_model)  # Deformación escalar -> vector

        # Positional encoding para dimensión espacial
        self.spatial_pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.num_patches)

        # Positional encoding para dimensión temporal
        self.temporal_pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_length)

        # Encoder espacial: procesa cada fotograma independientemente
        spatial_encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.spatial_encoder = TransformerEncoder(spatial_encoder_layer, num_layers=2)

        # Encoder temporal: procesa la secuencia de fotogramas
        temporal_encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.temporal_encoder = TransformerEncoder(temporal_encoder_layer, num_layers=num_encoder_layers)

        # Mecanismo de atención dispersa (Lightning Indexer)
        self.sparse_attention = SparseAttention(d_model, nhead, dropout)

        # Cabeza de predicción
        if task_type == 'regression':
            # Para predicción auto-regresiva: predecir el siguiente fotograma
            self.regression_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, self.num_patches)  # Salida: siguiente fotograma aplanado
            )
        elif task_type == 'classification':
            # Para clasificación: estado de la falla
            self.classification_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )

        # Inicializar pesos
        self._init_weights()

    def _init_weights(self):
        """Inicializa los pesos del modelo."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass del modelo.

        Args:
            x: Tensor de entrada [batch_size, seq_length, height, width]
            mask: Máscara de atención opcional

        Returns:
            Salida del modelo [batch_size, num_classes] o [batch_size, height, width]
        """
        batch_size, seq_len, height, width = x.shape

        # Aplanar espacialmente cada fotograma: [batch, seq_len, num_patches]
        num_patches = height * width
        x_flat = x.view(batch_size, seq_len, num_patches, 1)  # Añadir dimensión para el escalar

        # Embedding lineal para convertir a d_model
        x_embed = self.spatial_embedding(x_flat)  # [batch, seq_len, num_patches, d_model]

        # Reorganizar para encoder espacial: tratar cada patch como token
        # [batch, seq_len, num_patches, d_model] -> [batch * seq_len, num_patches, d_model]
        x_spatial = x_embed.view(batch_size * seq_len, num_patches, self.d_model)

        # Positional encoding espacial
        x_spatial = self.spatial_pos_encoder(x_spatial)

        # Encoder espacial: procesa cada fotograma independientemente
        spatial_output = self.spatial_encoder(x_spatial)  # [batch * seq_len, num_patches, d_model]

        # Reorganizar para encoder temporal: [batch, seq_len, num_patches, d_model]
        x_temporal = spatial_output.view(batch_size, seq_len, num_patches, self.d_model)

        # Para encoder temporal, necesitamos [batch, num_patches, seq_len, d_model]
        # Transponer seq_len y num_patches
        x_temporal = x_temporal.transpose(1, 2)  # [batch, num_patches, seq_len, d_model]

        # Aplanar para encoder temporal: [batch * num_patches, seq_len, d_model]
        x_temporal_flat = x_temporal.contiguous().view(batch_size * num_patches, seq_len, self.d_model)

        # Positional encoding temporal
        x_temporal_encoded = self.temporal_pos_encoder(x_temporal_flat)

        # Encoder temporal
        temporal_output = self.temporal_encoder(x_temporal_encoded)  # [batch * num_patches, seq_len, d_model]

        # Tomar el último timestep para predicción: [batch * num_patches, d_model]
        last_timestep = temporal_output[:, -1, :]

        # Reorganizar: [batch, num_patches, d_model]
        temporal_features = last_timestep.view(batch_size, num_patches, self.d_model)

        # Atención dispersa global
        sparse_out = self.sparse_attention(temporal_features)  # [batch, num_patches, d_model]

        # Cabeza de predicción
        if self.task_type == 'regression':
            # Predecir el siguiente fotograma: [batch, num_patches]
            output = self.regression_head(sparse_out.view(batch_size * num_patches, self.d_model))
            output = output.view(batch_size, num_patches)
            # Reorganizar a [batch, height, width]
            output = output.view(batch_size, height, width)
        elif self.task_type == 'classification':
            # Promediar sobre todos los patches para clasificación global
            pooled = sparse_out.mean(dim=1)  # [batch, d_model]
            output = self.classification_head(pooled)  # [batch, num_classes]

        return output

class PositionalEncoding(nn.Module):
    """Positional encoding para secuencias."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [seq_len, batch_size, embedding_dim] o [batch_size, seq_len, embedding_dim]
        """
        if x.dim() == 3 and x.shape[0] != self.pe.shape[0]:
            # Caso: [batch_size, seq_len, embedding_dim] -> [seq_len, batch_size, embedding_dim]
            x = x.transpose(0, 1)

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SparseAttention(nn.Module):
    """
    Mecanismo de atención dispersa (Lightning Indexer).
    Identifica y enfoca atención en las regiones más predictivas.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, top_k: int = 16):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.top_k = top_k

        # Proyecciones para queries, keys, values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

        # Capa para seleccionar top-k posiciones
        self.selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass de atención dispersa.

        Args:
            x: Tensor de entrada [batch_size, seq_len, d_model]
            mask: Máscara opcional

        Returns:
            Tensor de salida [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Proyectar queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Calcular scores de atención
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Aplicar máscara si existe
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Seleccionar top-k posiciones más importantes
        importance_scores = self.selector(x)  # [batch, seq_len, 1]

        # Para cada cabeza y batch, seleccionar top-k
        attention_weights = []
        for b in range(batch_size):
            for h in range(self.nhead):
                # Obtener importancia para esta cabeza
                imp = importance_scores[b, :, 0]  # [seq_len]

                # Seleccionar top-k índices
                _, top_k_indices = torch.topk(imp, min(self.top_k, seq_len), dim=-1)

                # Crear máscara dispersa
                sparse_mask = torch.zeros_like(scores[b, h])  # [seq_len, seq_len]
                sparse_mask[top_k_indices.unsqueeze(1), :] = 1
                sparse_mask[:, top_k_indices] = 1

                # Aplicar softmax solo en posiciones seleccionadas
                masked_scores = scores[b, h] * sparse_mask
                masked_scores = masked_scores.masked_fill(sparse_mask == 0, float('-inf'))
                weights = F.softmax(masked_scores, dim=-1)

                attention_weights.append(weights)

        # Stack de nuevo
        attention_weights = torch.stack(attention_weights, dim=0).view(
            batch_size, self.nhead, seq_len, seq_len
        )

        # Aplicar atención
        attended = torch.matmul(attention_weights, v)

        # Concatenar cabezas y proyectar
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(attended)

        return self.dropout(output)

def create_model(task_type: str = 'regression', **kwargs) -> SpatialTemporalTransformer:
    """
    Factory function para crear modelos.

    Args:
        task_type: Tipo de tarea ('regression' o 'classification')
        **kwargs: Parámetros adicionales para el modelo

    Returns:
        Modelo inicializado
    """
    default_params = {
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'seq_length': 30,
        'grid_size': (50, 50),
        'num_classes': 3,
        'task_type': task_type
    }

    default_params.update(kwargs)

    model = SpatialTemporalTransformer(**default_params)
    return model

def load_model_checkpoint(checkpoint_path: str, device: str = 'cpu') -> SpatialTemporalTransformer:
    """
    Carga un modelo desde un checkpoint.

    Args:
        checkpoint_path: Ruta del archivo de checkpoint
        device: Dispositivo para cargar el modelo

    Returns:
        Modelo cargado
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_params = checkpoint['model_params']
    model = create_model(**model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model

def save_model_checkpoint(model: SpatialTemporalTransformer, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, checkpoint_path: str) -> None:
    """
    Guarda un checkpoint del modelo.

    Args:
        model: Modelo a guardar
        optimizer: Optimizador
        epoch: Época actual
        loss: Pérdida actual
        checkpoint_path: Ruta donde guardar
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_params': {
            'd_model': getattr(model, 'd_model', 256),
            'nhead': getattr(model, 'nhead', 8),
            'num_encoder_layers': getattr(model, 'num_encoder_layers', 4),
            'dim_feedforward': getattr(model, 'dim_feedforward', 1024),
            'dropout': getattr(model, 'dropout', 0.1),
            'seq_length': getattr(model, 'seq_length', 30),
            'grid_size': getattr(model, 'grid_size', (50, 50)),
            'num_classes': 3 if getattr(model, 'task_type', 'classification') == 'classification' else None,
            'task_type': getattr(model, 'task_type', 'classification')
        }
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint guardado en {checkpoint_path}")

# Funciones de utilidad para evaluación
def predict_next_frame(model: SpatialTemporalTransformer, sequence: torch.Tensor,
                      device: str = 'cpu') -> torch.Tensor:
    """
    Predice el siguiente fotograma usando el modelo de regresión.

    Args:
        model: Modelo entrenado
        sequence: Secuencia de entrada [seq_length, height, width]
        device: Dispositivo

    Returns:
        Fotograma predicho [height, width]
    """
    model.eval()
    with torch.no_grad():
        # Agregar dimensión de batch
        x = sequence.unsqueeze(0).to(device)  # [1, seq_length, height, width]

        # Forward pass
        prediction = model(x)  # [1, height, width]

        return prediction.squeeze(0).cpu()

def classify_sequence_state(model: SpatialTemporalTransformer, sequence: torch.Tensor,
                           device: str = 'cpu') -> Tuple[int, torch.Tensor]:
    """
    Clasifica el estado de una secuencia.

    Args:
        model: Modelo entrenado
        sequence: Secuencia de entrada [seq_length, height, width]
        device: Dispositivo

    Returns:
        Tupla de (clase_predicha, probabilidades)
    """
    model.eval()
    with torch.no_grad():
        x = sequence.unsqueeze(0).to(device)
        logits = model(x)  # [1, num_classes]

        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        return predicted_class, probabilities.squeeze(0).cpu()

if __name__ == "__main__":
    # Ejemplo de uso
    print("Creando modelo de ejemplo...")

    # Modelo para regresión
    model_reg = create_model(task_type='regression', seq_length=30, grid_size=(50, 50))
    print(f"Modelo de regresión creado: {sum(p.numel() for p in model_reg.parameters())} parámetros")

    # Modelo para clasificación
    model_clf = create_model(task_type='classification', seq_length=30, grid_size=(50, 50), num_classes=3)
    print(f"Modelo de clasificación creado: {sum(p.numel() for p in model_clf.parameters())} parámetros")

    # Ejemplo de forward pass
    batch_size, seq_len, height, width = 2, 30, 50, 50
    x = torch.randn(batch_size, seq_len, height, width)

    with torch.no_grad():
        out_reg = model_reg(x)
        print(f"Salida regresión: {out_reg.shape}")  # [batch, height, width]

        out_clf = model_clf(x)
        print(f"Salida clasificación: {out_clf.shape}")  # [batch, num_classes]

    print("Arquitectura implementada correctamente.")