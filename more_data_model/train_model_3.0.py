import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import time
from tqdm import tqdm
import warnings
import math
warnings.filterwarnings('ignore')

# 混合精度训练
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("混合精度训练不可用")

class AmazonReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 标签映射
        self.label_map = {'NEG': 0, 'NEU': 1, 'POS': 2}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_map[label], dtype=torch.long)
        }

class OptimizedBERTTrainer:
    def __init__(self, model_name='bert-base-chinese', num_classes=3, dropout_rate=0.4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 混合精度训练
        self.scaler = GradScaler() if (self.device.type == 'cuda' and AMP_AVAILABLE) else None
        
        # 加载tokenizer和模型 - 直接使用原始BERT模型，通过参数设置dropout
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes,
            hidden_dropout_prob=dropout_rate,  # BERT内部的dropout
            attention_probs_dropout_prob=dropout_rate  # 注意力机制的dropout
        )
        self.model.to(self.device)
        
        # 标签映射
        self.label_map = {'NEG': 0, 'NEU': 1, 'POS': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        print("优化模型初始化完成")
        print(f"混合精度训练: {'启用' if self.scaler else '禁用'}")
        print(f"Dropout率: {dropout_rate}")
    
    def calculate_optimal_max_length(self, train_df, text_column='cleaned_text', max_cap=256):
        """智能计算最优max_length - 基于数据分布"""
        text_lengths = train_df[text_column].str.len()
        
        # 计算统计量
        quantile_95 = int(text_lengths.quantile(0.95))
        quantile_90 = int(text_lengths.quantile(0.90))
        
        # 动态选择max_length：平衡效率和覆盖度
        if quantile_95 <= 128:
            optimal_length = 128
        elif quantile_95 <= 192:
            optimal_length = 192
        else:
            optimal_length = min(quantile_90, max_cap)
        
        print(f"文本长度统计:")
        print(f"  平均长度: {text_lengths.mean():.1f}")
        print(f"  95%分位数: {quantile_95}")
        print(f"  最终max_length: {optimal_length}")
        
        return optimal_length
    
    def prepare_data(self, train_df, val_df, test_df, batch_size=16):
        """准备数据加载器"""
        print("准备数据加载器...")
        
        # 智能计算max_length
        max_length = self.calculate_optimal_max_length(train_df)
        
        # 训练集
        train_dataset = AmazonReviewDataset(
            texts=train_df['cleaned_text'].values,
            labels=train_df['label'].values,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        # 验证集
        val_dataset = AmazonReviewDataset(
            texts=val_df['cleaned_text'].values,
            labels=val_df['label'].values,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        # 测试集抽样 - 减少测试集大小
        if len(test_df) > 2000:
            test_df_sampled = test_df.sample(2000, random_state=42)
            print(f"测试集抽样: {len(test_df_sampled)} 条")
        else:
            test_df_sampled = test_df
        
        test_dataset = AmazonReviewDataset(
            texts=test_df_sampled['cleaned_text'].values,
            labels=test_df_sampled['label'].values,
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
        
        # 显示标签分布
        print(f"\n训练集标签分布:")
        print(train_df['label'].value_counts())
        
        return max_length
    
    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        """改进的余弦退火调度器"""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    def train(self, epochs=20, learning_rate=2e-5, warmup_ratio=0.1):  # 修改：epochs从5改为20
        """优化训练模型"""
        print("开始优化训练...")
        
        # 优化器 - 使用权重衰减正则化
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
        
        # 学习率调度 - 余弦退火 + warmup
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = self.get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练记录
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # 早停策略 - 修改耐心值
        best_accuracy = 0
        patience = 5  # 修改：从3改为5
        patience_counter = 0
        min_delta = 0.002  # 增加最小改进阈值
        
        print(f"训练参数:")
        print(f"  最大epoch数: {epochs}")
        print(f"  早停耐心: {patience}")
        print(f"  最小改进阈值: {min_delta}")
        
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
                # 移动到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 混合精度训练
                if self.scaler:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                        logits = outputs.logits
                    
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                
                # 统计
                total_train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                # 更新进度条
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
            val_accuracy, val_report = self.evaluate(self.val_loader)
            val_accuracies.append(val_accuracy)
            
            print(f"\n训练统计:")
            print(f"  训练损失: {avg_train_loss:.4f}")
            print(f"  训练准确率: {train_accuracy:.4f}")
            print(f"  验证准确率: {val_accuracy:.4f}")
            
            # 改进的早停策略
            if val_accuracy > best_accuracy + min_delta:
                best_accuracy = val_accuracy
                patience_counter = 0
                self.save_model('best_optimized_model')
                print(f"  ✅ 新的最佳模型保存，准确率: {val_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"  ⏳ 准确率未提升，耐心计数: {patience_counter}/{patience}")
            
            # 早停触发
            if patience_counter >= patience:
                print("  🛑 早停触发")
                break
        
        print(f"\n训练完成，最佳验证准确率: {best_accuracy:.4f}")
        print(f"实际训练轮数: {len(train_losses)}")
        return train_losses, train_accuracies, val_accuracies
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        
        # 转换为原始标签
        pred_labels = [self.reverse_label_map[p] for p in predictions]
        true_labels_str = [self.reverse_label_map[t] for t in true_labels]
        
        report = classification_report(true_labels_str, pred_labels, 
                                     target_names=['差评', '中评', '好评'])
        
        return accuracy, report
    
    def predict(self, texts):
        """预测新文本"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
                pred = torch.argmax(logits, dim=1).cpu().item()
                prob = probs.cpu().numpy()[0]
                
                predictions.append(self.reverse_label_map[pred])
                probabilities.append(prob)
        
        return predictions, probabilities
    
    def save_model(self, model_dir):
        """保存模型"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 直接使用transformers的保存方法
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        print(f"模型保存到: {model_dir}")
    
    def load_model(self, model_dir):
        """加载模型"""
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        print(f"模型从 {model_dir} 加载")

def main():
    # 优化参数设置 - 增加训练轮次
    BATCH_SIZE = 16
    EPOCHS = 20  # 修改：从5改为20
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    DROPOUT_RATE = 0.4  # Dropout提高到40%
    
    print("开始扩展训练商品评价分类模型...")
    print("=" * 60)
    print("使用的优化技术:")
    print("  ✅ 混合精度训练")
    print("  ✅ 智能max_length计算")
    print("  ✅ 余弦退火学习率调度 + Warmup")
    print("  ✅ 扩展早停策略(耐心=5)")  # 修改描述
    print(f"  ✅ 高Dropout正则化({DROPOUT_RATE*100}%)")
    print("  ✅ 测试集抽样(最多2000条)")
    print("  ✅ 权重衰减 + 梯度裁剪")
    print(f"  ✅ 扩展训练轮次({EPOCHS}轮)")  # 新增描述
    print("=" * 60)
    
    # 加载数据
    print("加载预处理数据...")
    try:
        train_df = pd.read_csv('data/train_data.csv')
        val_df = pd.read_csv('data/val_data.csv')
        test_df = pd.read_csv('data/test_data.csv')
        
        print(f"训练集: {len(train_df)} 条")
        print(f"验证集: {len(val_df)} 条")
        print(f"测试集: {len(test_df)} 条")
        
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到 - {e}")
        return
    
    # 抽样训练（避免数据量太大）
    if len(train_df) > 30000:  # 进一步减少训练数据量
        train_df = train_df.sample(30000, random_state=42)
        print(f"训练集抽样: {len(train_df)} 条")
    
    if len(val_df) > 5000:  # 减少验证集大小
        val_df = val_df.sample(5000, random_state=42)
        print(f"验证集抽样: {len(val_df)} 条")
    
    # 初始化优化训练器
    trainer = OptimizedBERTTrainer(dropout_rate=DROPOUT_RATE)
    
    # 准备数据（会自动计算最优max_length）
    max_length = trainer.prepare_data(train_df, val_df, test_df, batch_size=BATCH_SIZE)
    
    # 开始训练
    print(f"\n开始扩展训练...")
    print(f"训练参数: batch_size={BATCH_SIZE}, max_length={max_length}")
    print(f"epochs={EPOCHS}, dropout={DROPOUT_RATE}")
    print("=" * 60)
    
    start_time = time.time()
    train_losses, train_accuracies, val_accuracies = trainer.train(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO
    )
    end_time = time.time()
    
    training_time = (end_time - start_time) / 60
    print(f"\n训练完成，总耗时: {training_time:.2f} 分钟")
    print(f"实际训练轮数: {len(train_losses)}")
    
    # 加载最佳模型进行最终测试
    print(f"\n加载最佳模型进行最终测试...")
    trainer.load_model('best_optimized_model')
    
    # 测试集评估
    print(f"\n=== 最终测试集评估 ===")
    test_accuracy, test_report = trainer.evaluate(trainer.test_loader)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    if test_accuracy >= 0.90:
        print("✅ 达到优秀准确率90%以上!")
    elif test_accuracy >= 0.85:
        print("✅ 达到良好准确率85%以上!")
    elif test_accuracy >= 0.80:
        print("⚠️  准确率尚可，可以接受")
    else:
        print("❌ 准确率较低，需要调整")
    
    print(f"\n测试集分类报告:")
    print(test_report)
    
    # 保存最终模型
    print(f"\n保存最终模型...")
    trainer.save_model('final_optimized_model')
    
    # 示例预测
    print(f"\n=== 示例预测 ===")
    sample_texts = [
        "这个商品质量很好，非常满意！物流也很快！",
        "一般般吧，没什么特别的感觉，价格还算合理",
        "质量太差了，根本不能用，浪费钱"
    ]
    
    predictions, probabilities = trainer.predict(sample_texts)
    for i, text in enumerate(sample_texts):
        pred_label = predictions[i]
        prob = probabilities[i]
        prob_neg, prob_neu, prob_pos = prob[0], prob[1], prob[2]
        
        print(f"文本: {text}")
        print(f"预测: {pred_label}")
        print(f"概率: 差评({prob_neg:.3f}), 中评({prob_neu:.3f}), 好评({prob_pos:.3f})")
        print()
    
    print(f"\n=== 扩展训练结果总结 ===")
    print(f"最佳验证准确率: {max(val_accuracies):.4f}")
    print(f"最终测试准确率: {test_accuracy:.4f}")
    print(f"训练耗时: {training_time:.2f} 分钟")
    print(f"实际训练轮数: {len(train_losses)}")
    print(f"Dropout率: {DROPOUT_RATE}")
    print(f"最大训练轮次: {EPOCHS}")
    print(f"早停耐心值: 5")
    print(f"模型已保存")

if __name__ == "__main__":
    main()