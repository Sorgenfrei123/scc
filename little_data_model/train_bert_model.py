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
warnings.filterwarnings('ignore')

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

class BERTTrainer:
    def __init__(self, model_name='bert-base-chinese', num_classes=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes
        )
        self.model.to(self.device)
        
        # 标签映射
        self.label_map = {'NEG': 0, 'NEU': 1, 'POS': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        print("模型初始化完成")
        
    def prepare_data(self, train_df, val_df, test_df, batch_size=16, max_length=128):
        """准备数据加载器"""
        print("准备数据加载器...")
        
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
        
        # 测试集
        test_dataset = AmazonReviewDataset(
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
        
    def train(self, train_loader, val_loader, epochs=3, learning_rate=2e-5):
        """训练模型"""
        print("开始训练模型...")
        
        # 优化器和调度器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 训练记录
        train_losses = []
        val_accuracies = []
        
        best_accuracy = 0
        patience = 2
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc="训练")
            for batch in progress_bar:
                # 移动到设备
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
                logits = outputs.logits
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # 统计
                total_train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{train_correct/train_total:.4f}'
                })
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            val_accuracy, val_report = self.evaluate(val_loader)
            val_accuracies.append(val_accuracy)
            
            print(f"\n训练损失: {avg_train_loss:.4f}")
            print(f"训练准确率: {train_accuracy:.4f}")
            print(f"验证准确率: {val_accuracy:.4f}")
            
            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                self.save_model('best_model')
                print(f"新的最佳模型保存，准确率: {val_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"准确率未提升，耐心计数: {patience_counter}/{patience}")
            
            # 早停
            if patience_counter >= patience:
                print("早停触发")
                break
        
        return train_losses, val_accuracies
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        predictions = []
        true_labels = []
        total_loss = 0
        
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
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
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
    # 参数设置
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    print("开始训练商品评价分类模型...")
    
    # 加载数据
    print("加载预处理数据...")
    train_df = pd.read_csv('data/train_data.csv')
    val_df = pd.read_csv('data/val_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")
    
    # 抽样训练（避免数据量太大）
    if len(train_df) > 50000:
        train_df = train_df.sample(50000, random_state=42)
        print(f"训练集抽样: {len(train_df)} 条")
    
    if len(val_df) > 10000:
        val_df = val_df.sample(10000, random_state=42)
        print(f"验证集抽样: {len(val_df)} 条")
    
    # 初始化训练器
    trainer = BERTTrainer()
    
    # 准备数据
    trainer.prepare_data(train_df, val_df, test_df, 
                        batch_size=BATCH_SIZE, 
                        max_length=MAX_LENGTH)
    
    # 训练模型
    start_time = time.time()
    train_losses, val_accuracies = trainer.train(
        trainer.train_loader, 
        trainer.val_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    end_time = time.time()
    
    print(f"\n训练完成，总耗时: {(end_time - start_time)/60:.2f} 分钟")
    
    # 测试集评估
    print("\n=== 测试集评估 ===")
    test_accuracy, test_report = trainer.evaluate(trainer.test_loader)
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"\n测试集分类报告:")
    print(test_report)
    
    # 保存最终模型
    trainer.save_model('trained_amazon_model')
    
    # 示例预测
    print("\n=== 示例预测 ===")
    sample_texts = [
        "这个商品质量很好，非常满意！",
        "一般般，没什么特别的感觉",
        "质量太差了，根本不能用"
    ]
    
    predictions, probabilities = trainer.predict(sample_texts)
    for i, text in enumerate(sample_texts):
        print(f"文本: {text}")
        print(f"预测: {predictions[i]}, 概率: {probabilities[i]}")
        print()

if __name__ == "__main__":
    main()