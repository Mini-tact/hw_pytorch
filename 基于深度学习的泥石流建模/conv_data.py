import pymysql.cursors

config = {
          'host':'127.0.0.1',
          'port':3306,
          'user':'root',
          'password':'123',
          'database':'data_1',
          'charset':'utf8',
          'cursorclass':pymysql.cursors.Cursor,
          }

# 连接数据库
connection = pymysql.connect(config)
