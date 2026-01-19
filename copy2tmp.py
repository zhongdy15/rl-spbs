#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import time
from datetime import datetime

# 定义源目录路径
SOURCE_DIR = "/mnt/data1/zdy/rl-spbs-gitee/logs"


def main():
    parser = argparse.ArgumentParser(description="根据匹配字段复制日志文件夹到 /tmp")

    # 1. 第一个参数：目标文件夹的前缀，默认为 'logs'
    # 使用 nargs='?' 表示这是一个可选的单个参数
    parser.add_argument(
        "prefix",
        nargs='?',
        default="logs",
        help="目标文件夹的前缀名称 (默认为 logs)"
    )

    # 2. 后续参数：匹配字段列表
    # 使用 nargs='*' 表示接收后面所有的参数作为一个列表
    parser.add_argument(
        "patterns",
        nargs='*',
        help="匹配文件夹名称的字段列表 (支持多个，空格分隔，默认为 * 即复制所有)"
    )

    args = parser.parse_args()

    # 处理匹配模式逻辑：如果没有输入patterns，则默认为 ['*']
    match_patterns = args.patterns if args.patterns else ["*"]

    # --- 创建目标文件夹 ---
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dest_folder_name = f"{args.prefix}_{current_time}"
    dest_path = os.path.join("/tmp", dest_folder_name)

    try:
        os.makedirs(dest_path, exist_ok=True)
        print(f"[Info] 目标文件夹已创建: {dest_path}")
    except OSError as e:
        print(f"[Error] 创建文件夹失败: {e}")
        return

    # --- 检查源目录 ---
    if not os.path.exists(SOURCE_DIR):
        print(f"[Error] 源目录不存在: {SOURCE_DIR}")
        return

    # --- 遍历并复制 ---
    print(f"[Info] 正在筛选: {match_patterns} (逻辑关系: OR)")

    count = 0
    for item in os.listdir(SOURCE_DIR):
        src_item_path = os.path.join(SOURCE_DIR, item)

        if os.path.isdir(src_item_path):
            is_match = False

            # --- 核心匹配逻辑修改 ---
            if "*" in match_patterns:
                # 如果列表里有 *，则全匹配
                is_match = True
            else:
                # 检查文件名是否包含 match_patterns 中的任意一个字符串
                # any() 函数：只要有一个条件满足就返回 True
                if any(pattern in item for pattern in match_patterns):
                    is_match = True

            if is_match:
                dst_item_path = os.path.join(dest_path, item)
                try:
                    shutil.copytree(src_item_path, dst_item_path)
                    print(f"  -> 已复制: {item}")
                    count += 1
                except Exception as e:
                    print(f"  [Error] 复制 {item} 失败: {e}")

    print(f"-" * 30)
    print(f"[Success] 任务完成。共复制了 {count} 个文件夹到 {dest_path}")


if __name__ == "__main__":
    main()
