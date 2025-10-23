import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EnhancedAmazonReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'NEG': 0, 'NEU': 1, 'POS': 2}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 更智能的编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False  # 对于分类任务通常不需要
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_map[label], dtype=torch.long)
        }

class EnhancedBERTTrainer:
    def __init__(self, model_name='bert-base-chinese', num_classes=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 增强的模型配置
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes,
            hidden_dropout_prob=0.2,    # 调整dropout
            attention_probs_dropout_prob=0.1,
            classifier_dropout=0.3,
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.to(self.device)
        
        # 标签映射
        self.label_map = {'NEG': 0, 'NEU': 1, 'POS': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        print("增强BERT模型初始化完成")
        
    def calculate_optimal_max_length(self, train_df, text_column='cleaned_text', max_cap=256):
        """更智能的max_length计算"""
        text_lengths = train_df[text_column].astype(str).str.len()
        quantile_90 = int(text_lengths.quantile(0.90))  # 使用90%分位数
        optimal_length = min(max(quantile_90, 64), max_cap)  # 确保最小长度
        
        print(f"文本长度统计:")
        print(f"  最小长度: {text_lengths.min()}")
        print(f"  平均长度: {text_lengths.mean():.1f}")
        print(f"  最大长度: {text_lengths.max()}")
        print(f"  90%分位数: {quantile_90}")
        print(f"  最终max_length: {optimal_length}")
        
        return optimal_length
    
    def compute_class_weights(self, labels):
        """计算类别权重处理不平衡数据"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array([0, 1, 2]),
            y=labels
        )
        print(f"类别权重: {class_weights}")
        return torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
    def prepare_data(self, train_df, val_df, test_df, batch_size=16):
        """增强的数据准备"""
        print("准备数据加载器...")
        
        # 数据质量检查
        self._check_data_quality(train_df, '训练集')
        self._check_data_quality(val_df, '验证集')
        self._check_data_quality(test_df, '测试集')
        
        # 智能计算max_length
        max_length = self.calculate_optimal_max_length(train_df)
        
        # 计算类别权重
        train_labels = [self.label_map[label] for label in train_df['label'].values]
        self.class_weights = self.compute_class_weights(train_labels)
        
        # 数据集
        train_dataset = EnhancedAmazonReviewDataset(
            texts=train_df['cleaned_text'].values,
            labels=train_df['label'].values,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        val_dataset = EnhancedAmazonReviewDataset(
            texts=val_df['cleaned_text'].values,
            labels=val_df['label'].values,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        test_dataset = EnhancedAmazonReviewDataset(
            texts=test_df['cleaned_text'].values,
            labels=test_df['label'].values,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        # 数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"训练集: {len(train_dataset)} 条")
        print(f"验证集: {len(val_dataset)} 条")
        print(f"测试集: {len(test_dataset)} 条")
        
        print(f"\n训练集标签分布:")
        label_counts = train_df['label'].value_counts()
        print(label_counts)
        
        return max_length
    
    def _check_data_quality(self, df, dataset_name):
        """检查数据质量"""
        print(f"\n{dataset_name}数据质量检查:")
        print(f"  总样本数: {len(df)}")
        print(f"  空值数量: {df['cleaned_text'].isnull().sum()}")
        print(f"  文本长度统计:")
        text_lengths = df['cleaned_text'].astype(str).str.len()
        print(f"    最短: {text_lengths.min()}, 最长: {text_lengths.max()}, 平均: {text_lengths.mean():.1f}")
        
        # 检查标签分布
        label_dist = df['label'].value_counts()
        print(f"  标签分布: {dict(label_dist)}")
        
        # 检查重复样本
        duplicates = df.duplicated(subset=['cleaned_text']).sum()
        print(f"  重复样本: {duplicates}")
    
    def train(self, epochs=8, learning_rate=2e-5):
        """增强的训练过程"""
        print("开始增强训练...")
        
        # 优化器配置
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        
        # 学习率调度器
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10%的warmup
            num_training_steps=total_steps
        )
        
        # 训练记录
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_accuracy = 0
        patience = 3
        patience_counter = 0
        
        print(f"训练参数:")
        print(f"  最大epoch数: {epochs}")
        print(f"  学习率: {learning_rate}")
        print(f"  总步数: {total_steps}")
        print(f"  Warmup步数: {int(0.1 * total_steps)}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(self.train_loader, desc="训练")
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # 应用类别权重
                logits = outputs.logits
                if self.class_weights is not None:
                    loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                    loss = loss_fct(logits.view(-1, 3), labels.view(-1))
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 统计
                total_train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                current_acc = train_correct / train_total
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'LR': f'{current_lr:.2e}'
                })
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            train_accuracy = train_correct / train_total
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            # 验证阶段
            val_accuracy, val_report, val_cm = self.evaluate(self.val_loader)
            val_accuracies.append(val_accuracy)
            
            print(f"\n训练统计:")
            print(f"  训练损失: {avg_train_loss:.4f}")
            print(f"  训练准确率: {train_accuracy:.4f}")
            print(f"  验证准确率: {val_accuracy:.4f}")
            print(f"  过拟合差距: {train_accuracy - val_accuracy:.4f}")
            
            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                self.save_model('best_enhanced_model')
                print(f"  ✅ 新的最佳模型保存，准确率: {val_accuracy:.4f}")
                
                # 保存每个类别的最佳准确率
                val_details = self.get_detailed_metrics(self.val_loader)
                print(f"  各类别准确率 - 差评: {val_details['NEG']:.4f}, 中评: {val_details['NEU']:.4f}, 好评: {val_details['POS']:.4f}")
            else:
                patience_counter += 1
                print(f"  ⏳ 准确率未提升，耐心计数: {patience_counter}/{patience}")
            
            # 早停
            if patience_counter >= patience:
                print("  🛑 早停触发")
                break
        
        print(f"\n训练完成，最佳验证准确率: {best_accuracy:.4f}")
        return train_losses, train_accuracies, val_accuracies
    
    def get_detailed_metrics(self, data_loader):
        """获取详细的各类别准确率"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for class_id, class_name in self.reverse_label_map.items():
            mask = true_labels == class_id
            if mask.sum() > 0:
                class_acc = (predictions[mask] == class_id).mean()
                class_accuracies[class_name] = class_acc
            else:
                class_accuracies[class_name] = 0.0
                
        return class_accuracies
    
    def evaluate(self, data_loader, dataset_name=""):
        """评估模型"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"评估{dataset_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        
        pred_labels = [self.reverse_label_map[p] for p in predictions]
        true_labels_str = [self.reverse_label_map[t] for t in true_labels]
        
        report = classification_report(true_labels_str, pred_labels, 
                                     target_names=['差评', '中评', '好评'])
        
        cm = confusion_matrix(true_labels_str, pred_labels, labels=['NEG', 'NEU', 'POS'])
        
        return accuracy, report, cm
    
    def save_model(self, model_dir):
        """保存模型"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'label_map': self.label_map,
                'reverse_label_map': self.reverse_label_map
            }
        }, f'{model_dir}/model_weights.pth')
        
        self.tokenizer.save_pretrained(model_dir)
        print(f"模型保存到: {model_dir}")
    
    def load_model(self, model_dir):
        """加载模型"""
        checkpoint = torch.load(f'{model_dir}/model_weights.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        if 'model_config' in checkpoint:
            self.label_map = checkpoint['model_config']['label_map']
            self.reverse_label_map = checkpoint['model_config']['reverse_label_map']
        
        print(f"模型从 {model_dir} 加载")

def data_preprocessing_analysis(train_df, val_df, test_df):
    """数据预处理分析和建议"""
    print("=== 数据预处理分析 ===")
    
    # 检查标签分布
    print("\n1. 标签分布分析:")
    for name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
        print(f"{name}: {df['label'].value_counts().to_dict()}")
    
    # 检查文本质量
    print("\n2. 文本质量分析:")
    for name, df in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
        text_lengths = df['cleaned_text'].astype(str).str.len()
        print(f"{name} - 平均长度: {text_lengths.mean():.1f}, 空文本: {df['cleaned_text'].isnull().sum()}")
    
    # 建议
    print("\n3. 改进建议:")
    print("   - 确保训练集足够大(建议10,000+样本)")
    print("   - 检查标签标注质量")
    print("   - 处理类别不平衡")
    print("   - 确保文本清洗质量")

def main():
    # 优化参数设置
    BATCH_SIZE = 16
    EPOCHS = 8
    LEARNING_RATE = 2e-5
    DATA_DIR = 'data_emoji'
    
    print("开始增强训练商品评价分类模型...")
    print("=" * 60)
    print("核心优化策略:")
    print("  ✅ 数据质量检查和清洗")
    print("  ✅ 类别权重处理不平衡")
    print("  ✅ 学习率调度器")
    print("  ✅ 更合适的max_length计算")
    print("  ✅ 详细的各类别准确率监控")
    print("=" * 60)
    
    # 加载预处理数据
    print("加载预处理数据...")
    try:
        train_df = pd.read_csv(f'{DATA_DIR}/train_data.csv')
        val_df = pd.read_csv(f'{DATA_DIR}/val_data.csv')
        test_df = pd.read_csv(f'{DATA_DIR}/test_data.csv')
        
        print(f"训练集: {len(train_df)} 条")
        print(f"验证集: {len(val_df)} 条")
        print(f"测试集: {len(test_df)} 条")
        
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到 - {e}")
        return
    
    # 数据预处理分析
    data_preprocessing_analysis(train_df, val_df, test_df)
    
    # 初始化训练器
    trainer = EnhancedBERTTrainer()
    
    # 准备数据
    max_length = trainer.prepare_data(train_df, val_df, test_df, batch_size=BATCH_SIZE)
    
    # 开始训练
    print(f"\n开始训练...")
    print(f"训练参数: batch_size={BATCH_SIZE}, max_length={max_length}")
    print(f"epochs={EPOCHS}, lr={LEARNING_RATE}")
    print("=" * 60)
    
    start_time = time.time()
    train_losses, train_accuracies, val_accuracies = trainer.train(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    end_time = time.time()
    
    training_time = (end_time - start_time) / 60
    print(f"\n训练完成，总耗时: {training_time:.2f} 分钟")
    
    # 加载最佳模型进行最终测试
    print(f"\n加载最佳模型进行最终测试...")
    trainer.load_model('best_enhanced_model')
    
    # 测试集评估
    print(f"\n=== 最终测试集评估 ===")
    test_accuracy, test_report, test_cm = trainer.evaluate(trainer.test_loader, "测试集")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 详细各类别准确率
    test_details = trainer.get_detailed_metrics(trainer.test_loader)
    print(f"\n各类别测试准确率:")
    for class_name, acc in test_details.items():
        print(f"  {class_name}: {acc:.4f}")
    
    print(f"\n测试集分类报告:")
    print(test_report)
    
    # 显示混淆矩阵
    print(f"\n测试集混淆矩阵:")
    print("       预测: 差评  中评  好评")
    print("真实:")
    labels = ['差评', '中评', '好评']
    for i, true_label in enumerate(labels):
        print(f"{true_label}:   {test_cm[i][0]:5d} {test_cm[i][1]:5d} {test_cm[i][2]:5d}")
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    })
    history_df.to_csv('enhanced_training_history.csv', index=False)
    
    print(f"\n训练完成!")
    print(f"测试准确率: {test_accuracy:.4f}")
    print(f"训练耗时: {training_time:.2f} 分钟")

if __name__ == "__main__":
    main()