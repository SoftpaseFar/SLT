import pandas as pd
import json

# 假设Excel文件名为 'data.xlsx'，工作表名为 'Sheet1'
excel_file = '文件路径'
sheet_name = 'Sheet1'

# 读取Excel文件
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# 初始化一个空列表，用于存放转换后的JSON对象
questions = []

# 遍历每一行数据
for index, row in df.iterrows():
    # 提取题目相关信息
    if pd.notna(row['答案']) and (row['答案'] == '正确' or row['答案'] == '错误'):
        # 处理正确或错误的题目
        correct_answer = 'A' if row['答案'] == '正确' else 'B'
        question = {
            "题干": row['题干'],
            "答案": correct_answer,
            "选项": ['正确', '错误']
        }
    else:
        # 处理其他题目
        question = {
            "题干": row['题干'],
            "答案": row['答案'],
            "选项": [
                row['A'],
                row['B'],
                row['C'],
                row['D']
            ]
        }

    # 将当前题目信息添加到列表中
    questions.append(question)

# 将列表转换成JSON格式
json_data = json.dumps(questions, ensure_ascii=False, indent=2)

# 输出JSON格式的数据
print(json_data)
