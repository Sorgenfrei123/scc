import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import time
import warnings
warnings.filterwarnings('ignore')

class BaselineEvaluator:
    def __init__(self, model_name='bert-base-chinese'):
        print("正在加载BERT-base-chinese初始模型...")
        
        # 使用transformers的pipeline进行情感分析
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        print(f"模型加载完成，使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # 标签映射
        self.label_map = {'NEG': '差评', 'NEU': '中评', 'POS': '好评'}

    def sentiment_score_to_label(self, score):
        """将0-1的情感分数转换为三分类标签"""
        if score < 0.33:
            return 'NEG'  # 差评
        elif score < 0.67:
            return 'NEU'  # 中评
        else:
            return 'POS'  # 好评

    def predict_sentiment(self, text):
        """预测单个文本的情感"""
        try:
            # 限制文本长度，避免BERT最大长度限制
            if len(text) > 500:
                text = text[:500]
                
            result = self.sentiment_pipeline(text)
            
            # 获取情感分数
            # transformers的情感分析pipeline返回LABEL_0(负面)或LABEL_1(正面)
            if result[0]['label'] == 'LABEL_1':  # 正面
                score = result[0]['score']
            else:  # 负面
                score = 1 - result[0]['score']
            
            predicted_label = self.sentiment_score_to_label(score)
            return predicted_label, score
            
        except Exception as e:
            print(f"预测失败: {text[:50]}... 错误: {e}")
            return 'NEU', 0.5  # 默认返回中评

    def batch_predict(self, texts, batch_size=16):
        """批量预测"""
        print("正在进行批量预测...")
        predictions = []
        scores = []
        
        total_samples = len(texts)
        
        for i in range(0, total_samples, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = []
            batch_scores = []
            
            for text in batch_texts:
                pred_label, score = self.predict_sentiment(text)
                batch_predictions.append(pred_label)
                batch_scores.append(score)
            
            predictions.extend(batch_predictions)
            scores.extend(batch_scores)
            
            if (i // batch_size) % 10 == 0:  # 每10个batch打印进度
                progress = min(i + batch_size, total_samples)
                print(f"进度: {progress}/{total_samples} ({progress/total_samples*100:.1f}%)")
        
        return predictions, scores

    def evaluate_dataset(self, df, dataset_name="验证集"):
        """评估数据集"""
        print(f"\n=== 正在评估 {dataset_name} ===")
        print(f"数据量: {len(df)}")
        print(f"标签分布:")
        print(df['label'].value_counts())
        
        # 获取真实标签和文本
        true_labels = df['label'].tolist()
        texts = df['cleaned_text'].tolist()
        
        # 批量预测
        start_time = time.time()
        predicted_labels, predicted_scores = self.batch_predict(texts)
        end_time = time.time()
        
        print(f"预测完成，耗时: {end_time - start_time:.2f}秒")
        
        # 计算准确率
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"\n{dataset_name}准确率: {accuracy:.4f}")
        
        # 分类报告
        print(f"\n{dataset_name}分类报告:")
        print(classification_report(true_labels, predicted_labels, 
                                  target_names=['差评', '中评', '好评']))
        
        # 混淆矩阵（文本形式）
        cm = confusion_matrix(true_labels, predicted_labels, labels=['NEG', 'NEU', 'POS'])
        print(f"\n{dataset_name}混淆矩阵:")
        print("       预测: 差评  中评  好评")
        print("真实:")
        labels = ['差评', '中评', '好评']
        for i, true_label in enumerate(labels):
            print(f"{true_label}:   {cm[i][0]:5d} {cm[i][1]:5d} {cm[i][2]:5d}")
        
        # 计算每个类别的准确率
        print(f"\n{dataset_name}各类别准确率:")
        for label in ['NEG', 'NEU', 'POS']:
            mask = np.array(true_labels) == label
            if mask.sum() > 0:
                label_accuracy = accuracy_score(
                    np.array(true_labels)[mask], 
                    np.array(predicted_labels)[mask]
                )
                print(f"  {self.label_map[label]}: {label_accuracy:.4f} ({mask.sum()}条)")
        
        # 保存预测结果
        results_df = df.copy()
        results_df['predicted_label'] = predicted_labels
        results_df['predicted_score'] = predicted_scores
        results_df['is_correct'] = results_df['label'] == results_df['predicted_label']
        
        return accuracy, results_df

    def analyze_errors(self, results_df, dataset_name, top_n=10):
        """分析预测错误的样本"""
        print(f"\n=== {dataset_name}错误分析 ===")
        
        # 找出预测错误的样本
        errors = results_df[results_df['label'] != results_df['predicted_label']]
        print(f"错误样本数量: {len(errors)}")
        print(f"总体准确率: {1 - len(errors)/len(results_df):.4f}")
        
        if len(errors) > 0:
            # 按错误类型分组
            error_types = errors.groupby(['label', 'predicted_label']).size().reset_index(name='count')
            print("\n错误类型分布:")
            for _, row in error_types.iterrows():
                true_label = self.label_map.get(row['label'], row['label'])
                pred_label = self.label_map.get(row['predicted_label'], row['predicted_label'])
                print(f"  {true_label} -> {pred_label}: {row['count']}个")
            
            # 显示一些典型错误样本
            print(f"\n前{top_n}个错误样本:")
            for i, (idx, row) in enumerate(errors.head(top_n).iterrows()):
                true_label = self.label_map.get(row['label'], row['label'])
                pred_label = self.label_map.get(row['predicted_label'], row['predicted_label'])
                print(f"\n样本 {i+1}:")
                print(f"  文本: {row['cleaned_text'][:100]}...")
                print(f"  真实: {true_label}, 预测: {pred_label}, 分数: {row['predicted_score']:.3f}")
                print(f"  是否正确: {'否'}")
        
        return errors

    def run_evaluation(self, data_dir='data_emoji'):
        """运行完整的评估流程"""
        print("开始评估BERT-base-chinese初始模型在预处理数据集上的表现...")
        print("=" * 60)
        
        # 创建结果目录
        os.makedirs('baseline_results', exist_ok=True)
        
        # 加载预处理后的数据
        try:
            print("\n正在加载预处理数据...")
            val_df = pd.read_csv(f'{data_dir}/val_data.csv')
            test_df = pd.read_csv(f'{data_dir}/test_data.csv')
            
            print(f"验证集: {len(val_df)} 条")
            print(f"测试集: {len(test_df)} 条")
            
            # 显示数据基本信息
            print(f"\n验证集标签分布:")
            print(val_df['label'].value_counts())
            print(f"\n测试集标签分布:")
            print(test_df['label'].value_counts())
            
        except FileNotFoundError as e:
            print(f"数据文件未找到: {e}")
            print("请先运行数据预处理脚本生成预处理数据")
            return None, None
        
        # 评估验证集
        val_accuracy, val_results = self.evaluate_dataset(val_df, "验证集")
        val_errors = self.analyze_errors(val_results, "验证集")
        
        # 评估测试集
        test_accuracy, test_results = self.evaluate_dataset(test_df, "测试集")
        test_errors = self.analyze_errors(test_results, "测试集")
        
        # 保存详细结果
        val_results.to_csv('baseline_results/validation_predictions.csv', index=False, encoding='utf-8')
        test_results.to_csv('baseline_results/test_predictions.csv', index=False, encoding='utf-8')
        val_errors.to_csv('baseline_results/validation_errors.csv', index=False, encoding='utf-8')
        test_errors.to_csv('baseline_results/test_errors.csv', index=False, encoding='utf-8')
        
        # 总结报告
        print("\n" + "=" * 60)
        print("BERT-base-chinese初始模型评估总结报告")
        print("=" * 60)
        print(f"验证集准确率: {val_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"平均准确率: {(val_accuracy + test_accuracy) / 2:.4f}")
        
        # 详细统计
        print(f"\n详细统计:")
        print(f"验证集总样本: {len(val_df)}")
        print(f"验证集正确预测: {len(val_df) - len(val_errors)}")
        print(f"验证集错误预测: {len(val_errors)}")
        print(f"测试集总样本: {len(test_df)}")
        print(f"测试集正确预测: {len(test_df) - len(test_errors)}")
        print(f"测试集错误预测: {len(test_errors)}")
        
        # 保存总结报告
        with open('baseline_results/summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("BERT-base-chinese初始模型评估总结报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"验证集准确率: {val_accuracy:.4f}\n")
            f.write(f"测试集准确率: {test_accuracy:.4f}\n")
            f.write(f"平均准确率: {(val_accuracy + test_accuracy) / 2:.4f}\n\n")
            f.write(f"验证集总样本: {len(val_df)}\n")
            f.write(f"验证集正确预测: {len(val_df) - len(val_errors)}\n")
            f.write(f"验证集错误预测: {len(val_errors)}\n")
            f.write(f"测试集总样本: {len(test_df)}\n")
            f.write(f"测试集正确预测: {len(test_df) - len(test_errors)}\n")
            f.write(f"测试集错误预测: {len(test_errors)}\n")
        
        print(f"\n详细结果已保存到 baseline_results/ 目录")
        
        return val_accuracy, test_accuracy

# 运行评估
if __name__ == "__main__":
    # 初始化评估器
    evaluator = BaselineEvaluator()
    
    # 运行完整评估
    print("开始评估BERT-base-chinese初始模型...")
    print("注意：这是未经微调的通用模型在商品评价数据上的表现")
    print("这将作为后续微调模型的baseline参考")
    print("=" * 60)
    
    val_acc, test_acc = evaluator.run_evaluation(data_dir='data_emoji')
    
    if val_acc is not None and test_acc is not None:
        print("\n" + "=" * 60)
        print("评估完成！")
        print(f"最终结果:")
        print(f"验证集准确率: {val_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        print(f"平均准确率: {(val_acc + test_acc) / 2:.4f}")
        print("\n这个准确率将作为后续微调模型的baseline参考")
    else:
        print("\n评估失败，请检查数据文件是否存在")