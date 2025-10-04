#!/usr/bin/env python3
"""
Entrenamiento del modelo Transformer Espacio-Temporal.
Implementa pipelines de entrenamiento para regresión y clasificación.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import h5py
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from model_architecture import create_model, save_model_checkpoint, load_model_checkpoint

class DeformationDataset(Dataset):
    """
    Dataset personalizado para datos de deformación con carga segmentada.
    """

    def __init__(self, h5_file: str, task_type: str = 'regression', transform=None, chunk_size: int = 1000):
        """
        Inicializa el dataset con carga segmentada.

        Args:
            h5_file: Ruta al archivo HDF5 con los datos
            task_type: Tipo de tarea ('regression' o 'classification')
            transform: Transformaciones opcionales
            chunk_size: Tamaño del chunk para carga segmentada
        """
        self.h5_file = h5_file
        self.task_type = task_type
        self.transform = transform
        self.chunk_size = chunk_size

        # Determinar qué dataset usar basado en el tipo de tarea
        self.seq_dataset = 'secuencias' if task_type == 'classification' else 'secuencias_entrada'
        self.label_dataset = 'etiquetas' if task_type == 'classification' else 'secuencias_objetivo'

        # Abrir archivo HDF5 y obtener metadatos (sin cargar datos)
        with h5py.File(h5_file, 'r') as f:
            grupo_principal = list(f.keys())[0]  # 'clasificacion' o 'regresion'
            grupo = f[grupo_principal]

            # Obtener forma de las secuencias sin cargarlas
            self.num_samples = grupo[self.seq_dataset].shape[0]
            self.seq_length = grupo[self.seq_dataset].shape[1]
            self.grid_size = (grupo[self.seq_dataset].shape[2], grupo[self.seq_dataset].shape[3])

            # Cargar etiquetas en memoria (son pequeñas)
            if task_type == 'classification':
                self.labels = grupo['etiquetas'][:]  # [num_samples]
                self.classes = json.loads(grupo.attrs['clases'])
                self.class_to_idx = json.loads(grupo.attrs['clase_a_indice'])
            elif task_type == 'regression':
                # Para regresión, las etiquetas son las secuencias_objetivo
                labels_raw = grupo['secuencias_objetivo'][:]  # [num_samples, 1, height, width]
                if len(labels_raw.shape) == 4:  # Si es secuencia completa, tomar el último fotograma
                    self.labels = labels_raw[:, -1]  # [num_samples, height, width]
                else:
                    self.labels = labels_raw  # [num_samples, height, width]

        # Convertir etiquetas a tensores de PyTorch
        if hasattr(self, 'labels') and self.labels is not None:
            self.labels = torch.from_numpy(self.labels).long() if task_type == 'classification' else torch.from_numpy(self.labels).float()

        # Cache para chunks cargados
        self.cache = {}
        self.cache_size = 5  # Mantener máximo 5 chunks en memoria

        print(f"Dataset inicializado con carga segmentada: {self.num_samples} muestras, {self.task_type}")
        print(f"Forma de secuencias: ({self.num_samples}, {self.seq_length}, {self.grid_size[0]}, {self.grid_size[1]})")

    def _load_sequence_chunk(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Carga un chunk de secuencias desde el archivo HDF5.
        """
        cache_key = f"seq_{start_idx}_{end_idx}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        with h5py.File(self.h5_file, 'r') as f:
            grupo_principal = list(f.keys())[0]
            grupo = f[grupo_principal]
            sequences_chunk = grupo[self.seq_dataset][start_idx:end_idx]  # [chunk_size, seq_length, height, width]

        # Convertir a tensor y cachear
        sequences_tensor = torch.from_numpy(sequences_chunk).float()

        # Gestionar cache (LRU simple)
        if len(self.cache) >= self.cache_size:
            # Remover el primer elemento (más antiguo)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = sequences_tensor
        return sequences_tensor

    def _load_label_chunk(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Carga un chunk de etiquetas para regresión (último fotograma).
        """
        cache_key = f"label_{start_idx}_{end_idx}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        with h5py.File(self.h5_file, 'r') as f:
            grupo_principal = list(f.keys())[0]
            grupo = f[grupo_principal]
            # Para regresión, usar el último fotograma como etiqueta
            labels_chunk = grupo[self.label_dataset][start_idx:end_idx]  # [chunk_size, height, width]

        labels_tensor = torch.from_numpy(labels_chunk).float()

        # Gestionar cache
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = labels_tensor
        return labels_tensor

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calcular qué chunk contiene este índice
        chunk_idx = idx // self.chunk_size
        start_idx = chunk_idx * self.chunk_size
        end_idx = min((chunk_idx + 1) * self.chunk_size, self.num_samples)
        local_idx = idx - start_idx

        # Cargar chunk de secuencias
        sequences_chunk = self._load_sequence_chunk(start_idx, end_idx)
        sequence = sequences_chunk[local_idx]  # [seq_length, height, width]

        # Obtener etiqueta
        if self.task_type == 'classification':
            label = self.labels[idx]
        else:  # regression
            if self.labels is not None:
                label = self.labels[idx]
            else:
                # Cargar chunk de etiquetas on-demand
                labels_chunk = self._load_label_chunk(start_idx, end_idx)
                label = labels_chunk[local_idx]

        if self.transform:
            sequence, label = self.transform(sequence, label)

        return sequence, label

def create_data_loaders(h5_file: str, task_type: str, batch_size: int = 8,
                       train_split: float = 0.8, num_workers: int = 4, chunk_size: int = 1000) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoaders para entrenamiento y validación con carga segmentada.

    Args:
        h5_file: Ruta al archivo HDF5
        task_type: Tipo de tarea
        batch_size: Tamaño del batch
        train_split: Proporción de datos para entrenamiento
        num_workers: Número de workers para DataLoader (0 recomendado para HDF5)
        chunk_size: Tamaño del chunk para carga segmentada

    Returns:
        Tupla de (train_loader, val_loader)
    """
    dataset = DeformationDataset(h5_file, task_type, chunk_size=chunk_size)

    # Dividir en train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                             generator=torch.Generator().manual_seed(42))

    # Crear DataLoaders optimizados
    prefetch = 2 if num_workers > 0 else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch)

    return train_loader, val_loader

class ModelTrainer:
    """
    Clase para entrenar modelos de predicción de deformación.
    """

    def __init__(self, model: nn.Module, device: str = 'auto', use_mixed_precision: bool = False, gradient_accumulation_steps: int = 1):
        """
        Inicializa el entrenador.

        Args:
            model: Modelo a entrenar
            device: Dispositivo ('auto', 'cpu', 'cuda')
            use_mixed_precision: Usar mixed precision training
            gradient_accumulation_steps: Número de pasos para acumular gradientes
        """
        self.model = model
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Configurar dispositivo
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Configurar AMP si está disponible y solicitado
        self.scaler = None
        if self.use_mixed_precision and self.device.type == 'cuda':
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
            except ImportError:
                print("Mixed precision no disponible, usando precisión completa")
                self.use_mixed_precision = False

        # Configurar loss y optimizer basado en el tipo de tarea
        if model.task_type == 'regression':
            self.criterion = nn.MSELoss()
        elif model.task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer con scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Historial de entrenamiento
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'learning_rates': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Entrena una época.

        Args:
            train_loader: DataLoader de entrenamiento

        Returns:
            Tupla de (loss_promedio, métrica_promedio)
        """
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc="Training")
        accumulation_step = 0
        
        for sequences, labels in progress_bar:
            sequences, labels = sequences.to(self.device), labels.to(self.device)

            # Forward pass con mixed precision si está disponible
            if self.use_mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps
            else:
                # Forward pass normal
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels) / self.gradient_accumulation_steps

            # Backward pass
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_step += 1
            
            # Actualizar pesos cada gradient_accumulation_steps
            if accumulation_step % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                accumulation_step = 0

            # Calcular métrica
            if self.model.task_type == 'regression':
                # MSE para regresión
                metric = loss.item()
            elif self.model.task_type == 'classification':
                # Accuracy para clasificación
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                metric = correct / labels.size(0)

            total_loss += loss.item()
            total_metric += metric

            # Actualizar progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'metric': f"{metric:.4f}"
            })

        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches

        return avg_loss, avg_metric

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Valida una época.

        Args:
            val_loader: DataLoader de validación

        Returns:
            Tupla de (loss_promedio, métrica_promedio)
        """
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                if self.model.task_type == 'regression':
                    metric = loss.item()
                elif self.model.task_type == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels).sum().item()
                    metric = correct / labels.size(0)

                total_loss += loss.item()
                total_metric += metric

        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches

        return avg_loss, avg_metric

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, patience: int = 10, save_dir: str = 'checkpoints') -> Dict:
        """
        Entrena el modelo completo.

        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            num_epochs: Número de épocas
            patience: Paciencia para early stopping
            save_dir: Directorio para guardar checkpoints

        Returns:
            Diccionario con historial de entrenamiento
        """
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        print(f"Iniciando entrenamiento en {self.device}")
        print(f"Tipo de tarea: {self.model.task_type}")
        print(f"Épocas: {num_epochs}, Paciencia: {patience}")

        for epoch in range(num_epochs):
            print(f"\nÉpoca {epoch + 1}/{num_epochs}")

            # Entrenamiento
            train_loss, train_metric = self.train_epoch(train_loader)

            # Validación
            val_loss, val_metric = self.validate_epoch(val_loader)

            # Actualizar scheduler
            self.scheduler.step()

            # Guardar en historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metric'].append(train_metric)
            self.history['val_metric'].append(val_metric)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Imprimir métricas
            print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                save_model_checkpoint(self.model, self.optimizer, epoch, val_loss, checkpoint_path)
                print(f"✓ Mejor modelo guardado (epoch {epoch + 1})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping en época {epoch + 1}")
                break

        # Cargar mejor modelo
        best_checkpoint = os.path.join(save_dir, 'best_model.pth')
        if os.path.exists(best_checkpoint):
            self.model = load_model_checkpoint(best_checkpoint, self.device)
            print(f"Mejor modelo cargado desde época {best_epoch + 1}")

        # Guardar historial final
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"Entrenamiento completado. Mejor época: {best_epoch + 1}")
        return self.history

    def plot_training_history(self, save_path: Optional[str] = None):
        """Grafica el historial de entrenamiento."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Metric
        metric_name = 'MSE' if self.model.task_type == 'regression' else 'Accuracy'
        ax2.plot(epochs, self.history['train_metric'], 'b-', label='Train')
        ax2.plot(epochs, self.history['val_metric'], 'r-', label='Validation')
        ax2.set_title(f'{metric_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.legend()
        ax2.grid(True)

        # Learning rate
        ax3.plot(epochs, self.history['learning_rates'], 'g-')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.set_yscale('log')
        ax3.grid(True)

        # Loss ratio (train/val)
        loss_ratios = [t/v if v > 0 else 0 for t, v in zip(self.history['train_loss'], self.history['val_loss'])]
        ax4.plot(epochs, loss_ratios, 'purple')
        ax4.set_title('Train/Val Loss Ratio')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Ratio')
        ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en {save_path}")
        else:
            plt.show()

def train_regression_model(h5_file: str, save_dir: str = 'checkpoints_regression',
                          num_epochs: int = 50, batch_size: int = 4, chunk_size: int = 1000) -> ModelTrainer:
    """
    Entrena un modelo para predicción de regresión.

    Args:
        h5_file: Archivo HDF5 con datos de regresión
        save_dir: Directorio para guardar checkpoints
        num_epochs: Número de épocas
        batch_size: Tamaño del batch
        chunk_size: Tamaño del chunk para carga segmentada

    Returns:
        Trainer con modelo entrenado
    """
    print("Entrenando modelo de regresión...")

    # Crear modelo
    model = create_model(task_type='regression', seq_length=29, grid_size=(50, 50))

    # Crear trainer
    trainer = ModelTrainer(model)

    # Crear data loaders
    train_loader, val_loader = create_data_loaders(h5_file, 'regression', batch_size=batch_size, chunk_size=chunk_size)

    # Entrenar
    trainer.train(train_loader, val_loader, num_epochs=num_epochs, save_dir=save_dir)

    return trainer

def train_classification_model(h5_file: str, save_dir: str = 'checkpoints_classification',
                              num_epochs: int = 50, batch_size: int = 8, chunk_size: int = 1000) -> ModelTrainer:
    """
    Entrena un modelo para clasificación.

    Args:
        h5_file: Archivo HDF5 con datos de clasificación
        save_dir: Directorio para guardar checkpoints
        num_epochs: Número de épocas
        batch_size: Tamaño del batch
        chunk_size: Tamaño del chunk para carga segmentada

    Returns:
        Trainer con modelo entrenado
    """
    print("Entrenando modelo de clasificación...")

    # Crear modelo simplificado para pruebas rápidas
    model = create_model(
        task_type='classification',
        seq_length=30,
        grid_size=(50, 50),
        num_classes=3,
        d_model=128,  # Reducido de 256
        nhead=4,      # Reducido de 8
        num_encoder_layers=3,  # Reducido de 6
        dim_feedforward=512   # Reducido de 1024
    )

    print(f"Modelo creado: {sum(p.numel() for p in model.parameters())} parámetros")

    # Crear trainer
    trainer = ModelTrainer(model)

    # Crear data loaders
    train_loader, val_loader = create_data_loaders(h5_file, 'classification', batch_size=batch_size, chunk_size=chunk_size)

    # Entrenar
    trainer.train(train_loader, val_loader, num_epochs=num_epochs, save_dir=save_dir)

    return trainer

def main():
    """Función principal de entrenamiento."""
    parser = argparse.ArgumentParser(description='Entrenar modelo de predicción de deformación')
    parser.add_argument('--task', type=str, choices=['regression', 'classification'],
                       default='classification', help='Tipo de tarea')
    parser.add_argument('--area', type=str, default='falla_anatolia',
                       help='Área de interés')
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=4, help='Tamaño del batch')
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Tamaño del chunk para carga segmentada')

    args = parser.parse_args()

    # Configurar archivo de datos
    if args.task == 'regression':
        h5_file = f"datasets/{args.area}_synthetic_regresion.h5"
        save_dir = f"checkpoints_{args.task}_{args.area}"
        trainer = train_regression_model(h5_file, save_dir, args.epochs, args.batch_size, args.chunk_size)
    elif args.task == 'classification':
        h5_file = f"datasets/{args.area}_synthetic_clasificacion.h5"
        save_dir = f"checkpoints_{args.task}_{args.area}"
        trainer = train_classification_model(h5_file, save_dir, args.epochs, args.batch_size, args.chunk_size)

    # Graficar historial
    plot_path = os.path.join(save_dir, 'training_history.png')
    trainer.plot_training_history(plot_path)

    print("Entrenamiento completado exitosamente!")

if __name__ == "__main__":
    # Para pruebas rápidas sin argumentos
    if len(os.sys.argv) == 1:
        print("Ejecutando entrenamiento de prueba...")

        # Verificar si existen archivos de datos
        test_files = [
            "datasets/falla_anatolia_clasificacion.h5",
            "datasets/falla_anatolia_regresion.h5"
        ]

        for h5_file in test_files:
            if os.path.exists(h5_file):
                print(f"Archivo encontrado: {h5_file}")
                try:
                    with h5py.File(h5_file, 'r') as f:
                        print(f"  Shape: {f['secuencias'].shape}")
                        print(f"  Tipo: {f.attrs.get('tipo_tarea', 'unknown')}")
                except Exception as e:
                    print(f"  Error al leer: {e}")
            else:
                print(f"Archivo no encontrado: {h5_file}")

        print("Para entrenar, ejecuta: python train_model.py --task classification --area falla_anatolia")
    else:
        main()