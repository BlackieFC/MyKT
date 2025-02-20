import pymysql
import json
from collections import OrderedDict
import pandas as pd
import os
from datetime import datetime


def create_connection(host, user, password, db, charset):
    """
    连接到MySQL数据库
    """
    connection = None
    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db,
            charset=charset
        )
    except pymysql.MySQLError as e:
        print(f"错误: '{e}'")
    return connection


def get_table_columns(_table_name, _connection):
    """获取表的列名"""
    _query = f"SHOW COLUMNS FROM {_table_name}"
    with _connection.cursor() as _cursor:
        _cursor.execute(_query)
        _result = _cursor.fetchall()

        # 提取列名
        _column_names = [_column[0] for _column in _result]
        return _column_names


def synchronize_sheet(_mydb, _table_name ,_query, path_out=None, _columns=None):
    """
    同步表
    """
    with _mydb.cursor() as _cursor:
        _cols = get_table_columns(_table_name, _mydb)  # 获取表的列名
        _cursor.execute(_query)
        _result = _cursor.fetchall()

    # 生成Dataframe
    _result = pd.DataFrame(_result, columns=_columns)
    # 保存
    if path_out is not None:
        _result.to_excel(path_out, index=False)

    return _result


"""连接到MySQL数据库"""
# KET英语听力
# mydb = create_connection(
#     host="rm-2zez2bef4uh12j5f98o.mysql.rds.aliyuncs.com",
#     user="jingling_rds_rw",
#     password="2CqSlaUuQfjUVV5+g0ut",
#     db="jingling-data",
#     charset='utf8mb4'
#     )

# arithmetic
mydb = create_connection(
    host="rm-2ze18b9v2p70z655k.mysql.rds.aliyuncs.com",
    user="data_engine_asnyc_ro",
    password="RbcrsQdQWxFFhe7_",
    db="data_engine_asnyc",
    charset='utf8mb4'
    )


"""同步表"""
root_path = '.'

# # questions
# table_name = 'arithmetic_questions'
# query = f"select * from {table_name}"
# file_out = os.path.join(root_path, f'{table_name}.xlsx')
# df_exer = synchronize_sheet(mydb, table_name, query, file_out)
#
# # points
# table_name = 'arithmetic_points'
# query = f"select * from {table_name}"
# file_out = os.path.join(root_path, f'{table_name}.xlsx')
# df_point = synchronize_sheet(mydb, table_name, query, file_out)
#
# # units
# table_name = 'arithmetic_units'
# query = f"select * from {table_name}"
# file_out = os.path.join(root_path, f'{table_name}.xlsx')
# df_unit = synchronize_sheet(mydb, table_name, query, file_out)

# records
columns = ['id','tal_id','question_id','question','answers','correction_result','created_at']
table_name = 'arithmetic_route_records'
query = f"SELECT {', '.join(columns)} FROM {table_name}"  #  LIMIT 1000

print(f'start at {datetime.now()}')
df = synchronize_sheet(mydb,
                       table_name,
                       query,
                       None,
                       _columns=columns
                       )
df = df.rename(columns={'question_id': 'point_id', 'correction_result': 'correct'})  # 重命名指定列

print(f'saving at {datetime.now()}')
# df.to_excel(os.path.join(root_path, f'{table_name}.xlsx'), index=False)              # 保存（超出最大行数）
df.to_csv(os.path.join(root_path, f'{table_name}.csv'), index=False)
print(f'done at {datetime.now()}')
