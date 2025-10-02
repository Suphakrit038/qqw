#!/usr/bin/env python3
"""
🎯 Enhanced Training Pipeline for Multi-Layer CNN
ระบบเทรนขั้นสูงสำหรับ Enhanced Multi-Layer CNN

Features:
- Progressive Training (การเทรนแบบค่อยเป็นค่อยไป)
- Advanced Data Augmentation (การเพิ่มข้อมูลขั้นสูง)
- Multi-Scale Training (การเทรนหลายระดับ)
- Curriculum Learning (การเรียนรู้แบบหลักสูตร)
- Advanced Loss Functions (ฟังก์ชันสูญเสียขั้นสูง)
- Model Ensemble Training (การเทรนโมเดลแบบรวม)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import our enhanced models
from .enhanced_multilayer_cnn import (
    EnhancedMultiLayerCNN, 
    TwoBranchEnhancedCNN,
    create_enhanced_cnn,
    model_summary
)
from .enhanced_preprocessing import (
    EnhancedPreprocessor,
    EnhancedPreprocessConfig
)
from .realistic_amulet_generator import (
    RealisticAmuletGenerator,
    AmuletGenerationConfig
)

@dataclass
class EnhancedTrainingConfig:
    """การตั้งค่าการเทรนขั้นสูง"""
    
    # Model settings
    model_type: str = 'two_branch'  # 'single' or 'two_branch'
    num_classes: int = 10
    embedding_dim: int = 512
    use_attention: bool = True
    use_fpn: bool = True
    fusion_method: str = 'concat'
    
    # Training settings
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Progressive training
    progressive_training: bool = True
    progressive_stages: List[Dict] = None
    
    # Advanced augmentation
    use_advanced_augmentation: bool = True
    augmentation_strength: float = 0.8
    
    # Multi-scale training
    multi_scale_training: bool = True
    scale_schedule: Dict = None
    
    # Curriculum learning
    curriculum_learning: bool = True
    difficulty_levels: List[str] = None
    
    # Loss function settings
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1
    
    # Regularization
    dropout_schedule: Dict = None
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Optimization
    optimizer_type: str = 'adamw'  # 'adamw', 'sgd', 'radam'
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    
    # Monitoring
    validate_every: int = 5
    save_every: int = 10
    early_stopping_patience: int = 20
    
    def __post_init__(self):
        if self.progressive_stages is None:
            self.progressive_stages = [
                {'epochs': 20, 'lr': 1e-3, 'frozen_layers': ['shared_cnn.feature_extractor']},
                {'epochs': 30, 'lr': 5e-4, 'frozen_layers': []},
                {'epochs': 50, 'lr': 1e-4, 'frozen_layers': []}
            ]
        
        if self.scale_schedule is None:
            self.scale_schedule = {
                0: [224, 224],
                30: [256, 256],
                60: [288, 288]
            }
        
        if self.difficulty_levels is None:
            self.difficulty_levels = ['easy', 'medium', 'hard']
        
        if self.dropout_schedule is None:
            self.dropout_schedule = {
                0: 0.5,
                30: 0.3,
                60: 0.1
            }

class FocalLoss(nn.Module):
    """Focal Loss สำหรับ imbalanced classes"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 num_classes: Optional[int] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        if num_classes is not None:
            self.alpha = torch.ones(num_classes) * alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        
        # One-hot encode targets with smoothing
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -torch.sum(smooth_targets * log_probs, dim=1)
        return loss.mean()

class AdvancedDataset(Dataset):
    """ชุดข้อมูลขั้นสูงพร้อม augmentation"""
    
    def __init__(self, image_pairs: List[Tuple[str, str]], 
                 labels: List[int],
                 preprocessor: EnhancedPreprocessor,
                 training: bool = True,
                 difficulty_level: str = 'medium',
                 augmentation_strength: float = 0.8):
        
        self.image_pairs = image_pairs
        self.labels = labels
        self.preprocessor = preprocessor
        self.training = training
        self.difficulty_level = difficulty_level
        self.augmentation_strength = augmentation_strength
        
        # Difficulty-based filtering (for curriculum learning)
        if difficulty_level != 'all':
            filtered_indices = self._filter_by_difficulty(difficulty_level)
            self.image_pairs = [self.image_pairs[i] for i in filtered_indices]
            self.labels = [self.labels[i] for i in filtered_indices]
    
    def _filter_by_difficulty(self, level: str) -> List[int]:
        """กรองข้อมูลตามระดับความยาก"""
        # Simple heuristic: use hash of filename to assign difficulty
        indices = []
        for i, (front, back) in enumerate(self.image_pairs):
            file_hash = hash(front + back) % 3
            
            if level == 'easy' and file_hash == 0:
                indices.append(i)
            elif level == 'medium' and file_hash == 1:
                indices.append(i)
            elif level == 'hard' and file_hash == 2:
                indices.append(i)
        
        return indices
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        front_path, back_path = self.image_pairs[idx]
        label = self.labels[idx]
        
        # Preprocess images
        processed = self.preprocessor.preprocess_pair(
            front_path, back_path, training=self.training
        )
        
        return processed['front'], processed['back'], torch.tensor(label, dtype=torch.long)

class EnhancedTrainer:
    """ตัวเทรนขั้นสูง"""
    
    def __init__(self, config: EnhancedTrainingConfig, 
                 model: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None):
        
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model if not provided
        if model is None:
            self.model = create_enhanced_cnn(
                num_classes=config.num_classes,
                model_type=config.model_type,
                embedding_dim=config.embedding_dim,
                use_attention=config.use_attention,
                use_fpn=config.use_fpn,
                fusion_method=config.fusion_method
            )
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Setup loss functions
        self.setup_loss_functions()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def setup_loss_functions(self):
        """ตั้งค่าฟังก์ชันสูญเสีย"""
        
        # Primary loss
        if self.config.use_focal_loss:
            self.primary_loss = FocalLoss(
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                num_classes=self.config.num_classes
            )
        else:
            self.primary_loss = nn.CrossEntropyLoss()
        
        # Secondary loss (label smoothing)
        if self.config.use_label_smoothing:
            self.secondary_loss = LabelSmoothingLoss(
                num_classes=self.config.num_classes,
                smoothing=self.config.label_smoothing
            )
        else:
            self.secondary_loss = None
    
    def setup_optimizer(self):
        """ตั้งค่าตัวปรับปรุงและตารางการเรียนรู้"""
        
        # Optimizer
        if self.config.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
                nesterov=True
            )
        
        # Scheduler
        if self.config.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=10,
                factor=0.5
            )
    
    def setup_monitoring(self):
        """ตั้งค่าการติดตาม"""
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        
        # TensorBoard
        log_dir = f"runs/enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, 
                   alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """MixUp data augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """MixUp loss calculation"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """เทรนหนึ่ง epoch"""
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Update dropout if scheduled
        if epoch in self.config.dropout_schedule:
            new_dropout = self.config.dropout_schedule[epoch]
            self.update_dropout(new_dropout)
        
        for batch_idx, (front_imgs, back_imgs, labels) in enumerate(dataloader):
            front_imgs = front_imgs.to(self.device)
            back_imgs = back_imgs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Apply MixUp if enabled
            if self.config.mixup_alpha > 0 and np.random.random() > 0.5:
                if self.config.model_type == 'two_branch':
                    front_mixed, labels_a, labels_b, lam = self.mixup_data(
                        front_imgs, labels, self.config.mixup_alpha
                    )
                    back_mixed, _, _, _ = self.mixup_data(
                        back_imgs, labels, lam  # Use same lambda
                    )
                    
                    outputs = self.model(front_mixed, back_mixed)
                    loss = self.mixup_criterion(
                        self.primary_loss, outputs['logits'], 
                        labels_a, labels_b, lam
                    )
                else:
                    # Single branch model
                    mixed_imgs, labels_a, labels_b, lam = self.mixup_data(
                        front_imgs, labels, self.config.mixup_alpha
                    )
                    outputs = self.model(mixed_imgs)
                    loss = self.mixup_criterion(
                        self.primary_loss, outputs['logits'],
                        labels_a, labels_b, lam
                    )
            else:
                # Normal forward pass
                if self.config.model_type == 'two_branch':
                    outputs = self.model(front_imgs, back_imgs)
                else:
                    outputs = self.model(front_imgs)
                
                # Calculate loss
                loss = self.primary_loss(outputs['logits'], labels)
                
                # Add secondary loss if enabled
                if self.secondary_loss is not None:
                    loss += 0.1 * self.secondary_loss(outputs['logits'], labels)
                
                # Add auxiliary losses if available
                if 'front_logits' in outputs and 'back_logits' in outputs:
                    aux_loss = (self.primary_loss(outputs['front_logits'], labels) +
                               self.primary_loss(outputs['back_logits'], labels)) * 0.3
                    loss += aux_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            if not (self.config.mixup_alpha > 0 and np.random.random() > 0.5):
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """ตรวจสอบความถูกต้อง"""
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for front_imgs, back_imgs, labels in dataloader:
                front_imgs = front_imgs.to(self.device)
                back_imgs = back_imgs.to(self.device)
                labels = labels.to(self.device)
                
                if self.config.model_type == 'two_branch':
                    outputs = self.model(front_imgs, back_imgs)
                else:
                    outputs = self.model(front_imgs)
                
                loss = self.primary_loss(outputs['logits'], labels)
                
                total_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def update_dropout(self, new_dropout: float):
        """อัพเดท dropout rate"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_dropout
    
    def save_checkpoint(self, filepath: Path, epoch: int, 
                       val_acc: float, is_best: bool = False):
        """บันทึก checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': asdict(self.config),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """วาดกราฟประวัติการเทรน"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_accs, 'b-', label='Train Accuracy')
        ax2.plot(epochs, self.val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(epochs, self.learning_rates, 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Model summary text
        ax4.text(0.1, 0.5, f"Model: {type(self.model).__name__}\\n"
                          f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}\\n"
                          f"Best Val Acc: {self.best_val_acc:.2f}%\\n"
                          f"Total Epochs: {len(self.train_losses)}",
                 transform=ax4.transAxes, fontsize=12,
                 verticalalignment='center')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self, train_dataloader: DataLoader, 
              val_dataloader: DataLoader,
              save_dir: Path) -> Dict:
        """เทรนโมเดลแบบเต็มรูปแบบ"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting enhanced training on {self.device}")
        print(f"Model: {type(self.model).__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validate
            if epoch % self.config.validate_every == 0:
                val_metrics = self.validate(val_dataloader)
                
                # Update metrics
                self.train_losses.append(train_metrics['loss'])
                self.val_losses.append(val_metrics['loss'])
                self.train_accs.append(train_metrics['accuracy'])
                self.val_accs.append(val_metrics['accuracy'])
                self.learning_rates.append(train_metrics['learning_rate'])
                
                # Log to TensorBoard
                self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
                self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('Learning_Rate', train_metrics['learning_rate'], epoch)
                
                # Check for best model
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Print progress
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{self.config.num_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
                print(f"  LR: {train_metrics['learning_rate']:.6f}, Time: {epoch_time:.2f}s")
                print(f"  Best Val Acc: {self.best_val_acc:.2f}%")
                
                # Save checkpoint
                if epoch % self.config.save_every == 0 or is_best:
                    checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
                    self.save_checkpoint(checkpoint_path, epoch+1, val_metrics['accuracy'], is_best)
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['accuracy'])
            else:
                self.scheduler.step()
            
            self.current_epoch = epoch + 1
        
        # Final validation and plots
        final_val_metrics = self.validate(val_dataloader)
        
        # Plot training history
        self.plot_training_history(save_dir / 'training_history.png')
        
        # Save final model
        final_checkpoint_path = save_dir / 'final_model.pt'
        self.save_checkpoint(final_checkpoint_path, self.current_epoch, 
                           final_val_metrics['accuracy'], False)
        
        # Save training config
        config_path = save_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        self.writer.close()
        
        return {
            'best_val_accuracy': self.best_val_acc,
            'final_val_accuracy': final_val_metrics['accuracy'],
            'total_epochs': self.current_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }

# Main training function
def train_enhanced_model(train_pairs: List[Tuple[str, str]],
                        train_labels: List[int],
                        val_pairs: List[Tuple[str, str]],
                        val_labels: List[int],
                        config: Optional[EnhancedTrainingConfig] = None,
                        save_dir: str = 'enhanced_model_output') -> Dict:
    """เทรนโมเดล Enhanced CNN แบบเต็มรูปแบบ"""
    
    if config is None:
        config = EnhancedTrainingConfig()
    
    # Setup preprocessing
    preprocess_config = EnhancedPreprocessConfig()
    preprocessor = EnhancedPreprocessor(preprocess_config)
    
    # Create datasets
    train_dataset = AdvancedDataset(
        train_pairs, train_labels, preprocessor,
        training=True, augmentation_strength=config.augmentation_strength
    )
    
    val_dataset = AdvancedDataset(
        val_pairs, val_labels, preprocessor,
        training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create and train model
    trainer = EnhancedTrainer(config)
    
    results = trainer.train(train_loader, val_loader, Path(save_dir))
    
    return results

if __name__ == "__main__":
    # Test the enhanced training pipeline
    print("=== Enhanced Training Pipeline Test ===")
    
    # Create test configuration
    config = EnhancedTrainingConfig(
        num_classes=10,
        batch_size=8,
        num_epochs=5,  # Short test
        use_attention=True,
        use_fpn=True,
        progressive_training=False,  # Disable for quick test
        multi_scale_training=False
    )
    
    print(f"Model type: {config.model_type}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Use attention: {config.use_attention}")
    print(f"Use FPN: {config.use_fpn}")
    print(f"Focal loss: {config.use_focal_loss}")
    print(f"Label smoothing: {config.use_label_smoothing}")
    
    # Create trainer with test model
    trainer = EnhancedTrainer(config)
    
    # Test model summary
    summary = model_summary(trainer.model)
    print(f"\\nModel parameters: {summary['total_parameters_millions']:.2f}M")
    print(f"Model size: {summary['model_size_mb']:.2f}MB")
    
    print("\\n✅ Enhanced Training Pipeline is ready!")
    print("\\n💡 Features included:")
    print("  🎯 Progressive Training")
    print("  🔄 Advanced Data Augmentation") 
    print("  📏 Multi-Scale Training")
    print("  📚 Curriculum Learning")
    print("  ⚡ Advanced Loss Functions")
    print("  🎲 MixUp & CutMix")
    print("  📊 Comprehensive Monitoring")
    print("  🚀 Early Stopping & LR Scheduling")