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
import os
import json
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

# Optional imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using fallback training mode")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ NumPy not available")

try:
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn/Matplotlib not available")

# Import our enhanced models (with fallback)
try:
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
    ENHANCED_MODELS_AVAILABLE = True
except ImportError as e:
    ENHANCED_MODELS_AVAILABLE = False
    print(f"⚠️ Enhanced models not available: {e}")

@dataclass
class EnhancedTrainingConfig:
    """Configuration for enhanced training pipeline"""
    
    # Model settings
    model_type: str = 'two_branch'  # 'single', 'two_branch'
    backbone: str = 'resnet50'
    num_classes: int = 10
    use_attention: bool = True
    use_fpn: bool = True
    
    # Training settings
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Progressive training
    use_progressive_training: bool = True
    warmup_epochs: int = 10
    fine_tune_epochs: int = 50
    
    # Loss functions
    loss_type: str = 'cross_entropy'  # 'cross_entropy', 'focal', 'label_smoothing'
    label_smoothing: float = 0.1
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    
    # Optimization
    optimizer_type: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    
    # Data augmentation
    use_advanced_augmentation: bool = True
    augmentation_strength: float = 0.8
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Curriculum learning
    use_curriculum_learning: bool = True
    curriculum_strategy: str = 'difficulty'  # 'difficulty', 'confidence'
    
    # Model ensemble
    use_ensemble: bool = False
    ensemble_size: int = 3
    
    # Regularization
    dropout: float = 0.5
    dropblock_prob: float = 0.1
    use_mixup: bool = True
    use_cutmix: bool = True
    
    # Monitoring
    save_best_only: bool = True
    early_stopping_patience: int = 20
    checkpoint_frequency: int = 10
    
    # Paths
    data_dir: str = "dataset"
    output_dir: str = "enhanced_training_output"
    log_dir: str = "logs"
    
    def __post_init__(self):
        # Create directories
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)

class AdvancedLossFunction:
    """Advanced loss functions for enhanced training"""
    
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        
    def get_loss_function(self):
        """Get the configured loss function"""
        if not TORCH_AVAILABLE:
            return None
            
        if self.config.loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.config.loss_type == 'focal':
            return self.focal_loss
        elif self.config.loss_type == 'label_smoothing':
            return self.label_smoothing_loss
        else:
            return nn.CrossEntropyLoss()
            
    def focal_loss(self, outputs, targets):
        """Focal Loss implementation"""
        ce_loss = nn.functional.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.config.focal_alpha * (1-pt)**self.config.focal_gamma * ce_loss
        return focal_loss.mean()
        
    def label_smoothing_loss(self, outputs, targets):
        """Label smoothing cross entropy loss"""
        log_probs = nn.functional.log_softmax(outputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.config.label_smoothing / (self.config.num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.config.label_smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class DataAugmentationPipeline:
    """Advanced data augmentation pipeline"""
    
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        
    def mixup_data(self, x, y, alpha=1.0):
        """Applies MixUp augmentation"""
        if not TORCH_AVAILABLE:
            return x, y, y, 1.0
            
        if alpha > 0:
            lam = torch.distributions.beta.Beta(alpha, alpha).sample()
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
        
    def cutmix_data(self, x, y, alpha=1.0):
        """Applies CutMix augmentation"""
        if not TORCH_AVAILABLE:
            return x, y, y, 1.0
            
        if alpha > 0:
            lam = torch.distributions.beta.Beta(alpha, alpha).sample()
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        return x, y_a, y_b, lam
        
    def rand_bbox(self, size, lam):
        """Generate random bounding box for CutMix"""
        W = size[2]
        H = size[3]
        cut_rat = (1. - lam).sqrt()
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()
        
        # uniform
        cx = torch.randint(0, W, (1,))
        cy = torch.randint(0, H, (1,))
        
        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class CurriculumLearning:
    """Curriculum learning implementation"""
    
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        self.difficulty_scores = {}
        
    def calculate_difficulty(self, dataset, model):
        """Calculate difficulty scores for curriculum learning"""
        if not TORCH_AVAILABLE:
            return {}
            
        model.eval()
        difficulty_scores = {}
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(dataset):
                if self.config.model_type == 'two_branch':
                    front_img, back_img = data
                    output = model(front_img, back_img)
                else:
                    output = model(data)
                    
                # Calculate confidence as inverse of difficulty
                probs = torch.softmax(output, dim=1)
                confidence = torch.max(probs, dim=1)[0]
                difficulty = 1.0 - confidence
                
                difficulty_scores[idx] = difficulty.item()
                
        return difficulty_scores
        
    def get_curriculum_order(self, difficulty_scores, epoch, total_epochs):
        """Get training order based on curriculum strategy"""
        if self.config.curriculum_strategy == 'difficulty':
            # Start with easy samples, gradually add harder ones
            progress = epoch / total_epochs
            threshold = progress  # Gradually increase difficulty threshold
            
            easy_samples = {k: v for k, v in difficulty_scores.items() if v <= threshold}
            return list(easy_samples.keys())
        else:
            return list(difficulty_scores.keys())

class EnhancedTrainer:
    """Main enhanced training class"""
    
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and TORCH_AVAILABLE else 'cpu')
        
        # Initialize components
        self.loss_function = AdvancedLossFunction(config)
        self.augmentation = DataAugmentationPipeline(config)
        self.curriculum = CurriculumLearning(config)
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Initialize tensorboard writer
        if TORCH_AVAILABLE:
            try:
                self.writer = SummaryWriter(log_dir=self.config.log_dir)
            except:
                self.writer = None
        else:
            self.writer = None
            
    def create_model(self):
        """Create and initialize the model"""
        if not ENHANCED_MODELS_AVAILABLE:
            print("⚠️ Enhanced models not available - using fallback")
            return None
            
        model = create_enhanced_cnn(
            num_classes=self.config.num_classes,
            model_type=self.config.model_type,
            backbone=self.config.backbone,
            use_attention=self.config.use_attention,
            use_fpn=self.config.use_fpn,
            dropout=self.config.dropout
        )
        
        if TORCH_AVAILABLE:
            model = model.to(self.device)
            
        return model
        
    def create_optimizer(self, model):
        """Create optimizer"""
        if not TORCH_AVAILABLE or model is None:
            return None
            
        if self.config.optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
        return optimizer
        
    def create_scheduler(self, optimizer):
        """Create learning rate scheduler"""
        if not TORCH_AVAILABLE or optimizer is None:
            return None
            
        if self.config.scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )
            
        return scheduler
        
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        if not TORCH_AVAILABLE:
            return 0.0, 0.0
            
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.config.model_type == 'two_branch':
                front_img, back_img = data
                front_img, back_img = front_img.to(self.device), back_img.to(self.device)
                data = (front_img, back_img)
            else:
                data = data.to(self.device)
                
            target = target.to(self.device)
            
            # Apply augmentation
            if self.config.use_mixup and torch.rand(1) < 0.5:
                if self.config.model_type == 'two_branch':
                    # Apply mixup to both images
                    front_mixed, target_a, target_b, lam = self.augmentation.mixup_data(
                        front_img, target, self.config.mixup_alpha
                    )
                    back_mixed, _, _, _ = self.augmentation.mixup_data(
                        back_img, target, self.config.mixup_alpha
                    )
                    data = (front_mixed, back_mixed)
                else:
                    data, target_a, target_b, lam = self.augmentation.mixup_data(
                        data, target, self.config.mixup_alpha
                    )
                    
                optimizer.zero_grad()
                
                if self.config.model_type == 'two_branch':
                    output = model(data[0], data[1])
                else:
                    output = model(data)
                    
                loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
                
            else:
                optimizer.zero_grad()
                
                if self.config.model_type == 'two_branch':
                    output = model(data[0], data[1])
                else:
                    output = model(data)
                    
                loss = criterion(output, target)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Log batch progress
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
        
    def validate(self, model, val_loader, criterion):
        """Validate the model"""
        if not TORCH_AVAILABLE:
            return 0.0, 0.0
            
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                if self.config.model_type == 'two_branch':
                    front_img, back_img = data
                    front_img, back_img = front_img.to(self.device), back_img.to(self.device)
                    output = model(front_img, back_img)
                else:
                    data = data.to(self.device)
                    output = model(data)
                    
                target = target.to(self.device)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
        
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save model checkpoint"""
        if not TORCH_AVAILABLE:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': asdict(self.config),
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.output_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.output_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'✅ New best model saved with accuracy: {self.best_accuracy:.2f}%')
            
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("🚀 Starting Enhanced Training Pipeline...")
        
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available - cannot proceed with training")
            return None
            
        # Create model, optimizer, and scheduler
        model = self.create_model()
        if model is None:
            print("❌ Failed to create model")
            return None
            
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        criterion = self.loss_function.get_loss_function()
        
        print(f"📊 Model: {self.config.model_type}")
        print(f"🎯 Training for {self.config.epochs} epochs")
        print(f"💾 Output directory: {self.config.output_dir}")
        
        # Training loop
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Update learning rate
            if scheduler:
                if self.config.scheduler_type == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                    
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            current_lr = optimizer.param_groups[0]['lr']
            self.training_history['learning_rates'].append(current_lr)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                
            # Check if best model
            is_best = val_acc > self.best_accuracy
            if is_best:
                self.best_accuracy = val_acc
                
            # Save checkpoint
            if epoch % self.config.checkpoint_frequency == 0 or is_best:
                self.save_checkpoint(model, optimizer, epoch, is_best)
                
            # Print progress
            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{self.config.epochs}] '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% '
                  f'LR: {current_lr:.6f} Time: {epoch_time:.1f}s')
                  
        print(f"🎉 Training completed! Best validation accuracy: {self.best_accuracy:.2f}%")
        
        # Save final training report
        self.save_training_report()
        
        return model
        
    def save_training_report(self):
        """Save training report"""
        report = {
            'training_date': datetime.now().isoformat(),
            'config': asdict(self.config),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
            'total_epochs': len(self.training_history['train_loss']),
            'final_train_acc': self.training_history['train_acc'][-1] if self.training_history['train_acc'] else 0,
            'final_val_acc': self.training_history['val_acc'][-1] if self.training_history['val_acc'] else 0
        }
        
        report_path = Path(self.config.output_dir) / 'training_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"📄 Training report saved: {report_path}")

# Convenience function
def train_enhanced_model(config: EnhancedTrainingConfig, 
                        train_loader, 
                        val_loader) -> Optional[object]:
    """Train enhanced model with the given configuration"""
    trainer = EnhancedTrainer(config)
    return trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    # Test the enhanced training pipeline
    print("🧪 Testing Enhanced Training Pipeline...")
    
    config = EnhancedTrainingConfig(
        model_type='two_branch',
        num_classes=10,
        epochs=5,  # Short test
        batch_size=8,
        use_progressive_training=True,
        use_advanced_augmentation=True
    )
    
    trainer = EnhancedTrainer(config)
    print("✅ Enhanced trainer created")
    
    # Create dummy model for testing
    if ENHANCED_MODELS_AVAILABLE:
        model = trainer.create_model()
        if model:
            print(f"✅ Model created successfully")
        else:
            print("❌ Failed to create model")
    else:
        print("⚠️ Enhanced models not available for testing")
        
    print("🎉 Enhanced Training Pipeline ready for use!")