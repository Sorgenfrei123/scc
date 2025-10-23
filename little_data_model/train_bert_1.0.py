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
        
        # 显示标签分布
        print(f"\n训练集标签分布:")
        print(train_df['label'].value_counts())
        print(f"\n验证集标签分布:")
        print(val_df['label'].value_counts())
        print(f"\n测试集标签分布:")
        print(test_df['label'].value_counts())
        
    def train(self, epochs=3, learning_rate=2e-5, warmup_ratio=0.1):
        """训练模型"""
        print("开始训练模型...")
        
        # 优化器和调度器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练记录
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_accuracy = 0
        patience = 3
        patience_counter = 0
        
        print(f"总训练步数: {total_steps}")
        print(f"预热步数: {warmup_steps}")
        
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
                
                # 更新进度条
                current_acc = train_correct / train_total
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}'
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
            
            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                self.save_model('best_amazon_model')
                print(f"  ✅ 新的最佳模型保存，准确率: {val_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"  ⏳ 准确率未提升，耐心计数: {patience_counter}/{patience}")
            
            # 早停
            if patience_counter >= patience:
                print("  🛑 早停触发")
                break
        
        print(f"\n训练完成，最佳验证准确率: {best_accuracy:.4f}")
        return train_losses, train_accuracies, val_accuracies
    
    def evaluate(self, data_loader, dataset_name=""):
        """评估模型"""
        self.model.eval()
        predictions = []
        true_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"评估{dataset_name}"):
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
        
        cm = confusion_matrix(true_labels_str, pred_labels, labels=['NEG', 'NEU', 'POS'])
        
        return accuracy, report, cm
    
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
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    DATA_DIR = 'data_emoji'  # 预处理数据目录
    
    print("开始训练商品评价分类模型...")
    print("=" * 50)
    
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
        print("请先运行数据预处理脚本生成预处理数据")
        return
    
    # 检查数据
    print(f"\n数据检查:")
    print(f"训练集列名: {train_df.columns.tolist()}")
    print(f"是否有cleaned_text列: {'cleaned_text' in train_df.columns}")
    print(f"是否有label列: {'label' in train_df.columns}")
    
    # 显示前几条数据
    print(f"\n训练集前3条数据:")
    for i in range(min(3, len(train_df))):
        print(f"  {i+1}. 文本: {train_df.iloc[i]['cleaned_text'][:60]}...")
        print(f"     标签: {train_df.iloc[i]['label']}")
    
    # 初始化训练器
    trainer = BERTTrainer()
    
    # 准备数据
    trainer.prepare_data(train_df, val_df, test_df, 
                        batch_size=BATCH_SIZE, 
                        max_length=MAX_LENGTH)
    
    # 训练模型
    print(f"\n开始训练...")
    print(f"参数: batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}, epochs={EPOCHS}, lr={LEARNING_RATE}")
    print("=" * 50)
    
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
    trainer.load_model('best_amazon_model')
    
    # 测试集评估
    print(f"\n=== 最终测试集评估 ===")
    test_accuracy, test_report, test_cm = trainer.evaluate(trainer.test_loader, "测试集")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"\n测试集分类报告:")
    print(test_report)
    
    # 显示混淆矩阵
    print(f"\n测试集混淆矩阵:")
    print("       预测: 差评  中评  好评")
    print("真实:")
    labels = ['差评', '中评', '好评']
    for i, true_label in enumerate(labels):
        print(f"{true_label}:   {test_cm[i][0]:5d} {test_cm[i][1]:5d} {test_cm[i][2]:5d}")
    
    # 保存最终模型
    print(f"\n保存最终模型...")
    trainer.save_model('final_amazon_model')
    
    # 示例预测
    print(f"\n=== 示例预测 ===")
    sample_texts = [
        "这个商品质量很好，非常满意！物流也很快！",
        "一般般吧，没什么特别的感觉，价格还算合理",
        "质量太差了，根本不能用，浪费钱",
        "东西还不错，就是包装有点简陋 😊",
        "超级差劲！客服态度也不好 👎"
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
    
    # 保存训练结果
    print(f"\n=== 训练结果总结 ===")
    print(f"最佳验证准确率: {max(val_accuracies):.4f}")
    print(f"最终测试准确率: {test_accuracy:.4f}")
    print(f"训练耗时: {training_time:.2f} 分钟")
    print(f"模型已保存到: best_amazon_model/ 和 final_amazon_model/")
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    })
    history_df.to_csv('training_history.csv', index=False)
    print(f"训练历史已保存到: training_history.csv")

if __name__ == "__main__":
    main()