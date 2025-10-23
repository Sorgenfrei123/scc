# data_processing.py
import pandas as pd
import numpy as np
import re
import jieba
from collections import Counter
import random
import os
from sklearn.utils import resample
import emoji

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
        """文本清洗 - 保留emoji表情"""
        if not isinstance(text, str):
            return ""
        
        # 1. 提取emoji表情
        emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
        emoji_text = ' '.join(emoji_list) if emoji_list else ''
        
        # 2. 去除URL
        text = re.sub(r'https?://\S+', '', text)
        
        # 3. 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 4. 处理长数字序列
        text = re.sub(r'\b\d{10,}\b', '', text)
        
        # 5. 标准化标点但保留emoji
        text = re.sub(r'[~!@#$%^&*()_+\-=，。？；：""''【】{}|、]', ' ', text)
        
        # 6. 去除多余空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 7. 如果有emoji，在文本末尾添加emoji描述
        if emoji_text:
            text = f"{text} [EMOJI {emoji_text}]"
        
        return text
    
    def analyze_emoji_usage(self, df, text_column='comment'):
        """分析emoji使用情况"""
        print("\n=== Emoji使用分析 ===")
        
        emoji_counter = Counter()
        total_emoji = 0
        texts_with_emoji = 0
        
        for text in df[text_column]:
            if isinstance(text, str):
                emojis_in_text = [c for c in text if c in emoji.EMOJI_DATA]
                if emojis_in_text:
                    texts_with_emoji += 1
                    total_emoji += len(emojis_in_text)
                    emoji_counter.update(emojis_in_text)
        
        print(f"包含emoji的文本数量: {texts_with_emoji}/{len(df)} ({texts_with_emoji/len(df)*100:.2f}%)")
        print(f"总emoji数量: {total_emoji}")
        print("最常见的emoji:")
        for emoji_char, count in emoji_counter.most_common(10):
            print(f"  {emoji_char}: {count}次")
        
        return emoji_counter

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
            print("原始标签分布:")
            print(df['label'].value_counts())
        else:
            print(f"警告: 未找到 {rating_column} 列")
            print("数据前3行:")
            print(df.head(3))
        
        return df

    def balanced_sampling(self, df, target_sizes={'train': 12000, 'val': 3000, 'test': 3000}):
        """平衡抽样 - 确保每个类别在训练/验证/测试集中数量均衡"""
        print("\n=== 开始平衡抽样 ===")
        print(f"目标数据量 - 训练集: {target_sizes['train']}, 验证集: {target_sizes['val']}, 测试集: {target_sizes['test']}")
        
        # 按标签分组
        neg_df = df[df['label'] == 'NEG']
        neu_df = df[df['label'] == 'NEU']
        pos_df = df[df['label'] == 'POS']
        
        print(f"原始分布 - 差评: {len(neg_df)}, 中评: {len(neu_df)}, 好评: {len(pos_df)}")
        
        # 计算每个数据集每个类别的目标样本数
        train_per_class = target_sizes['train'] // 3
        val_per_class = target_sizes['val'] // 3
        test_per_class = target_sizes['test'] // 3
        
        print(f"每个类别目标样本数 - 训练: {train_per_class}, 验证: {val_per_class}, 测试: {test_per_class}")
        
        # 为每个类别抽样训练、验证、测试集
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for label_df, label_name in [(neg_df, '差评'), (neu_df, '中评'), (pos_df, '好评')]:
            if len(label_df) < (train_per_class + val_per_class + test_per_class):
                print(f"警告: {label_name} 数据不足，使用全部可用数据")
                available_samples = len(label_df)
                # 按比例分配
                train_samples = min(train_per_class, available_samples * train_per_class // (train_per_class + val_per_class + test_per_class))
                val_samples = min(val_per_class, available_samples * val_per_class // (train_per_class + val_per_class + test_per_class))
                test_samples = available_samples - train_samples - val_samples
            else:
                train_samples = train_per_class
                val_samples = val_per_class
                test_samples = test_per_class
            
            # 随机抽样
            label_sampled = label_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            train_label = label_sampled.iloc[:train_samples]
            val_label = label_sampled.iloc[train_samples:train_samples + val_samples]
            test_label = label_sampled.iloc[train_samples + val_samples:train_samples + val_samples + test_samples]
            
            train_dfs.append(train_label)
            val_dfs.append(val_label)
            test_dfs.append(test_label)
            
            print(f"{label_name}: 训练{len(train_label)}条, 验证{len(val_label)}条, 测试{len(test_label)}条")
        
        # 合并所有数据
        train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=42)
        val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=42)
        test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=42)
        
        print(f"\n最终抽样结果:")
        print(f"训练集: {len(train_df)} 条")
        print(f"验证集: {len(val_df)} 条")
        print(f"测试集: {len(test_df)} 条")
        
        print("\n训练集标签分布:")
        print(train_df['label'].value_counts())
        print("\n验证集标签分布:")
        print(val_df['label'].value_counts())
        print("\n测试集标签分布:")
        print(test_df['label'].value_counts())
        
        return train_df, val_df, test_df

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
    
    def augment_data(self, df, text_column='comment', label_column='label', 
                    target_per_class=6000, force_augment=True):
        """数据增强 - 强制增强以增加数据多样性"""
        print("\n=== 开始数据增强 ===")
        
        augmented_data = []
        label_counts = df[label_column].value_counts()
        
        print(f"增强前各类别数量:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} 条")
        
        for label in label_counts.index:
            label_data = df[df[label_column] == label]
            current_count = len(label_data)
            
            # 强制增强，即使数据足够也增加多样性
            needed = target_per_class - current_count
            
            print(f"\n增强 {label} 类别:")
            print(f"  现有: {current_count} 条, 目标: {target_per_class} 条, 需要增强: {needed} 条")
            
            if needed > 0 and len(label_data) > 0:
                augmented_count = 0
                progress_interval = max(1, needed // 10)
                
                for idx, (_, sample) in enumerate(label_data.iterrows()):
                    if augmented_count >= needed:
                        break
                    
                    original_text = sample[text_column]
                    attempts = 0
                    max_attempts_per_sample = 5
                    
                    while augmented_count < needed and attempts < max_attempts_per_sample:
                        attempts += 1
                        
                        # 使用多种增强方法
                        aug_method = random.choice(['deletion', 'swap', 'insertion', 'combined'])
                        
                        try:
                            if aug_method == 'deletion' and len(original_text) > 10:
                                augmented_text = self.random_deletion(original_text, p=0.15)
                            elif aug_method == 'swap' and len(original_text) > 8:
                                augmented_text = self.random_swap(original_text, n=2)
                            elif aug_method == 'insertion' and len(original_text) > 8:
                                augmented_text = self.random_insertion(original_text, n=1)
                            elif aug_method == 'combined' and len(original_text) > 15:
                                # 组合增强
                                temp_text = self.random_deletion(original_text, p=0.1)
                                augmented_text = self.random_swap(temp_text, n=1)
                            else:
                                continue
                            
                            # 质量控制
                            if (augmented_text != original_text and 
                                len(augmented_text) >= len(original_text) * 0.6 and
                                len(augmented_text) <= len(original_text) * 1.5 and
                                len(augmented_text) >= 5):
                                
                                new_sample = sample.copy()
                                new_sample[text_column] = augmented_text
                                new_sample['is_augmented'] = True
                                augmented_data.append(new_sample)
                                augmented_count += 1
                                
                                if augmented_count % progress_interval == 0:
                                    print(f"    已生成 {augmented_count}/{needed} 条")
                                break
                                
                        except Exception as e:
                            continue
                
                print(f"  实际生成: {augmented_count} 条")
        
        # 合并增强数据
        if augmented_data:
            augmented_df = pd.DataFrame(augmented_data)
            final_df = pd.concat([df, augmented_df], ignore_index=True)
            
            print(f"\n增强完成:")
            print(f"  原始数据: {len(df)} 条")
            print(f"  增强数据: {len(augmented_df)} 条")
            print(f"  总计: {len(final_df)} 条")
            
            print(f"\n增强后各类别数量:")
            final_counts = final_df[label_column].value_counts()
            for label in final_counts.index:
                print(f"  {label}: {final_counts[label]} 条")
        else:
            final_df = df
            print("未生成增强数据")
        
        return final_df

    def load_sentiment_words(self):
        """加载情感词典"""
        sentiment_words = {
            'positive': ['好', '很好', '非常好', '不错', '满意', '喜欢', '赞', '漂亮', '舒服', '划算', '优秀', '棒'],
            'negative': ['差', '很差', '不好', '不满意', '讨厌', '垃圾', '坑', '贵', '慢', '难用', '糟糕', '差劲']
        }
        return sentiment_words

    def preprocess_pipeline(self, file_path, text_column='comment', rating_column='rating', 
                          augment=True, target_sizes={'train': 12000, 'val': 3000, 'test': 3000}, 
                          output_dir='data_enhanced'):
        """完整的数据预处理流程"""
        
        # 1. 加载数据
        df = self.load_data(file_path)
        
        # 2. 基础清洗
        df = self.basic_clean(df, text_column)
        
        # 3. 分析原始数据中的emoji使用情况
        self.analyze_emoji_usage(df, text_column)
        
        # 4. 文本清洗（包含emoji处理）
        print("\n=== 文本清洗（包含emoji处理） ===")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # 5. 分析清洗后数据中的emoji使用情况
        print("\n=== 清洗后数据Emoji使用分析 ===")
        self.analyze_emoji_usage(df, 'cleaned_text')
        
        # 6. 处理标签
        df = self.process_labels(df, rating_column)
        
        # 7. 平衡抽样
        train_df, val_df, test_df = self.balanced_sampling(df, target_sizes)
        
        # 8. 数据增强（强制增强以增加多样性）
        if augment and 'label' in train_df.columns:
            print("\n=== 对训练集进行数据增强 ===")
            # 设置每个类别的目标数量为6000，强制增强
            train_df = self.augment_data(train_df, 'cleaned_text', 'label', 
                                       target_per_class=6000, force_augment=True)
        
        # 9. 保存处理后的数据
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, 'train_data.csv')
        val_path = os.path.join(output_dir, 'val_data.csv')
        test_path = os.path.join(output_dir, 'test_data.csv')
        
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        print(f"\n处理后的数据已保存到:")
        print(f"  训练集: {train_path}")
        print(f"  验证集: {val_path}")
        print(f"  测试集: {test_path}")
        
        # 10. 最终统计
        print("\n=== 预处理完成 ===")
        print(f"最终数据形状:")
        print(f"  训练集: {train_df.shape}")
        print(f"  验证集: {val_df.shape}")
        print(f"  测试集: {test_df.shape}")
        
        print("\n各数据集标签分布:")
        print("训练集:")
        print(train_df['label'].value_counts())
        print("\n验证集:")
        print(val_df['label'].value_counts())
        print("\n测试集:")
        print(test_df['label'].value_counts())
        
        return train_df, val_df, test_df

# 使用示例
if __name__ == "__main__":
    # 初始化预处理器
    preprocessor = DataPreprocessor()
    
    # 运行预处理流程（包含强制数据增强）
    train_data, val_data, test_data = preprocessor.preprocess_pipeline(
        file_path='yf_amazon/ratings.csv',
        text_column='comment',
        rating_column='rating',
        augment=True,
        target_sizes={
            'train': 12000,
            'val': 3000, 
            'test': 3000
        },
        output_dir='data_enhanced'  # 使用新的输出目录
    )