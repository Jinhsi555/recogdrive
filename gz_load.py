#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import numpy as np
import pickle

def read_gz_file(file_path):
    """
    读取 .gz 压缩文件
    
    Args:
        file_path (str): .gz 文件的路径
        
    Returns:
        data: 解压后的数据
    """
    try:
        with gzip.open(file_path, 'rb') as f:
            # 尝试使用 pickle 加载数据（如果是 Python 序列化对象）
            data = pickle.load(f)
            return data
    except Exception as e:
        print(f"使用 pickle 读取失败: {e}")
        
        # 如果不是 pickle 格式，尝试以文本方式读取
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = f.read()
                return data
        except Exception as e2:
            print(f"以文本方式读取也失败: {e2}")
            
            # 最后尝试以二进制方式读取原始数据
            try:
                with gzip.open(file_path, 'rb') as f:
                    raw_data = f.read()
                    return raw_data
            except Exception as e3:
                print(f"读取文件时发生错误: {e3}")
                return None

def analyze_data(data):
    """
    分析读取到的数据
    
    Args:
        data: 从文件中读取的数据
    """
    print(f"数据类型: {type(data)}")
    
    if isinstance(data, (list, tuple)):
        print(f"数据长度: {len(data)}")
        if len(data) > 0:
            print(f"第一个元素类型: {type(data[0])}")
            print(f"第一个元素: {data[0]}")
            
    elif isinstance(data, dict):
        print(f"字典键数量: {len(data)}")
        print(f"字典键: {list(data.keys())}")
        if len(data) > 0:
            first_key = list(data.keys())[0]
            print(f"第一个键值对: {first_key} => {data[first_key]}")
            
    elif isinstance(data, np.ndarray):
        print(f"Numpy 数组形状: {data.shape}")
        print(f"Numpy 数组类型: {data.dtype}")
        print(f"数组前5个元素: {data[:5]}")
        
    elif isinstance(data, str):
        print(f"字符串长度: {len(data)}")
        print(f"字符串前100字符: {data[:100]}")
        
    elif isinstance(data, bytes):
        print(f"字节数据长度: {len(data)}")
        print(f"字节数据前100字节: {data[:100]}")
        
    else:
        print(f"数据内容预览: {str(data)[:200]}")

if __name__ == "__main__":
    # 指定文件路径
    file_path = "/mnt/data/data/wlb/recogdrive/exp/recogdrive_agent_cache_train_no_hidden_state_with_worldmirror_qwen3vl/2021.06.09.20.26.11_veh-35_00247_00529/b3ad6c622b7250c0/qwen3vl_features.gz"
    
    print("正在读取文件...")
    data = read_gz_file(file_path)
    
    if data is not None:
        print("文件读取成功！")
        print("=" * 50)
        analyze_data(data)
    else:
        print("文件读取失败！")