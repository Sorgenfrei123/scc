import pandas as pd
import numpy as np
import re
import jieba
from collections import Counter
import random
import os
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        # 中文停用词列表
        self.stopwords = self.load_stopwords()
        
    def load_stopwords(self):
        """加载停用词表"""
        stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'
        ])
        punctuation = '！？。，；：""''（）《》【】~@#￥%……&*（）——+={}|[]\\:;"<>?,./·`'
        stopwords.update(set(punctuation))
        return stopwords
    
    def load_data(self, file_path):
        """加载数据"""
        print("正在加载数据...")
        print(f"尝试读取文件: {file_path}")
        print(f"文件是否存在: {os.path.exists(file_path)}")
        
        df = pd.read_csv(file_path)
        print(f"原始数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        return df
    
    def basic_clean(self, df, text_column='comment'):
        """基础数据清洗"""
        print("\n=== 开始基础数据清洗 ===")
        
        # 1. 处理缺失值
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        print(f"删除缺失值: {initial_count} -> {len(df)}")
        
        # 2. 去除重复数据
        initial_count = len(df)
        df = df.drop_duplicates(subset=[text_column])
        print(f"删除重复数据: {initial_count} -> {len(df)}")
        
        # 3. 过滤过短文本
        initial_count = len(df)
        df = df[df[text_column].str.len() >= 5]
        print(f"过滤过短文本: {initial_count} -> {len(df)}")
        
        return df
    
    def clean_text(self, text):
        """文本清洗"""
        if not isinstance(text, str):
            return ""
        
        # 1. 去除URL
        text = re.sub(r'https?://\S+', '', text)
        
        # 2. 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 3. 处理长数字序列
        text = re.sub(r'\b\d{10,}\b', '', text)
        
        # 4. 标准化标点
        text = re.sub(r'[~!@#$%^&*()_+\-=，。？；：""''【】{}|、]', ' ', text)
        
        # 5. 去除多余空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_labels(self, df, rating_column='rating'):
        """处理标签 - 将评分转换为三分类"""
        print("\n=== 处理标签 ===")
        
        if rating_column in df.columns:
            def rating_to_label(rating):
                try:
                    rating = float(rating)
                    if rating <= 2:
                        return 'NEG'  # 差评
                    elif rating == 3:
                        return 'NEU'  # 中评
                    else:
                        return 'POS'  # 好评
                except:
                    return 'NEU'  # 默认中评
            
            df['label'] = df[rating_column].apply(rating_to_label)
            print("标签分布:")
            print(df['label'].value_counts())
        else:
            print(f"警告: 未找到 {rating_column} 列")
            print("数据前3行:")
            print(df.head(3))
        
        return df

    def random_deletion(self, text, p=0.1):
        """随机删除数据增强"""
        words = jieba.lcut(text)
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def random_swap(self, text, n=1):
        """随机交换数据增强"""
        words = jieba.lcut(text)
        if len(words) <= 1:
            return text
        
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def random_insertion(self, text, n=1):
        """随机插入数据增强"""
        words = jieba.lcut(text)
        if len(words) <= 1:
            return text
        
        new_words = words.copy()
        for _ in range(n):
            candidate_words = [word for word in words if word not in self.stopwords]
            if candidate_words:
                random_word = random.choice(candidate_words)
                random_pos = random.randint(0, len(new_words))
                new_words.insert(random_pos, random_word)
        
        return ' '.join(new_words)
    
    def augment_data(self, df, text_column='comment', label_column='label', augment_per_class=1000):
        """数据增强"""
        print("\n=== 开始数据增强 ===")
        
        augmented_data = []
        label_counts = df[label_column].value_counts()
        
        for label in label_counts.index:
            label_data = df[df[label_column] == label]
            current_count = len(label_data)
            needed = max(0, augment_per_class - current_count)
            
            print(f"标签 {label}: 现有 {current_count} 条，需要增强 {needed} 条")
            
            if needed > 0:
                augmented_count = 0
                while augmented_count < needed and len(label_data) > 0:
                    sample = label_data.sample(1).iloc[0]
                    original_text = sample[text_column]
                    
                    aug_method = random.choice(['deletion', 'swap', 'insertion'])
                    
                    try:
                        if aug_method == 'deletion' and len(original_text) > 15:
                            augmented_text = self.random_deletion(original_text)
                        elif aug_method == 'swap' and len(original_text) > 10:
                            augmented_text = self.random_swap(original_text)
                        elif aug_method == 'insertion' and len(original_text) > 10:
                            augmented_text = self.random_insertion(original_text)
                        else:
                            augmented_text = original_text
                        
                        if augmented_text != original_text:
                            new_sample = sample.copy()
                            new_sample[text_column] = augmented_text
                            augmented_data.append(new_sample)
                            augmented_count += 1
                            
                    except Exception as e:
                        continue
        
        if augmented_data:
            augmented_df = pd.DataFrame(augmented_data)
            df = pd.concat([df, augmented_df], ignore_index=True)
            print(f"数据增强完成，新增 {len(augmented_data)} 条数据")
            print(f"增强后数据形状: {df.shape}")
        
        return df
    
    def split_dataset(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """划分训练集、验证集和测试集"""
        print("\n=== 开始数据集划分 ===")
        
        # 首先划分训练+验证集和测试集
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['label'] if 'label' in df.columns else None
        )
        
        # 然后从训练+验证集中划分训练集和验证集
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_size/(1-test_size),  # 调整比例
            random_state=random_state, 
            stratify=train_val_df['label'] if 'label' in train_val_df.columns else None
        )
        
        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")
        print(f"测试集大小: {len(test_df)}")
        
        # 检查各数据集的标签分布
        if 'label' in df.columns:
            print("\n训练集标签分布:")
            print(train_df['label'].value_counts())
            print("\n验证集标签分布:")
            print(val_df['label'].value_counts())
            print("\n测试集标签分布:")
            print(test_df['label'].value_counts())
        
        return train_df, val_df, test_df
    
    def preprocess_pipeline(self, file_path, text_column='comment', rating_column='rating', 
                          augment=False, test_size=0.2, val_size=0.1, output_dir='../data'):
        """完整的数据预处理流程"""
        
        # 1. 加载数据
        df = self.load_data(file_path)
        
        # 2. 基础清洗
        df = self.basic_clean(df, text_column)
        
        # 3. 文本清洗
        print("\n=== 文本清洗 ===")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # 4. 处理标签
        df = self.process_labels(df, rating_column)
        
        # 5. 数据增强（可选）
        if augment and 'label' in df.columns:
            df = self.augment_data(df, 'cleaned_text', 'label')
        
        # 6. 划分数据集
        train_df, val_df, test_df = self.split_dataset(df, test_size, val_size)
        
        # 7. 保存处理后的数据
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, 'train_data.csv')
        val_path = os.path.join(output_dir, 'val_data.csv')
        test_path = os.path.join(output_dir, 'test_data.csv')
        all_path = os.path.join(output_dir, 'all_processed_data.csv')
        
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        df.to_csv(all_path, index=False, encoding='utf-8')
        
        print(f"\n处理后的数据已保存到:")
        print(f"  训练集: {train_path}")
        print(f"  验证集: {val_path}")
        print(f"  测试集: {test_path}")
        print(f"  完整数据: {all_path}")
        
        # 8. 最终统计
        print("\n=== 预处理完成 ===")
        print(f"最终数据形状:")
        print(f"  训练集: {train_df.shape}")
        print(f"  验证集: {val_df.shape}")
        print(f"  测试集: {test_df.shape}")
        
        if 'label' in df.columns:
            print("\n各数据集标签分布:")
            print("训练集:")
            print(train_df['label'].value_counts())
            print("\n验证集:")
            print(val_df['label'].value_counts())
            print("\n测试集:")
            print(test_df['label'].value_counts())
        
        # 显示一些样例
        print("\n清洗前后样例 (训练集):")
        for i in range(min(2, len(train_df))):
            print(f"样本 {i+1}:")
            print(f"  原始: {train_df.iloc[i][text_column][:50]}...")
            print(f"  清洗后: {train_df.iloc[i]['cleaned_text'][:50]}...")
            if 'label' in train_df.columns:
                print(f"  标签: {train_df.iloc[i]['label']}")
            print()
        
        return train_df, val_df, test_df

# 使用示例
if __name__ == "__main__":
    # 初始化预处理器
    preprocessor = DataPreprocessor()
    
    # 运行预处理流程
    train_data, val_data, test_data = preprocessor.preprocess_pipeline(
        file_path='yf_amazon/ratings.csv',
        text_column='comment',      # 评论文本列
        rating_column='rating',     # 评分列
        augment=True,               # 是否进行数据增强
        test_size=0.2,              # 测试集比例 20%
        val_size=0.1,               # 验证集比例 10%
        output_dir='data'           # 输出目录
    )