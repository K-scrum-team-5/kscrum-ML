import pymysql
import configparser

# DB 정보
config = configparser.ConfigParser()
config.read('config.ini')
db_host = config['DB']['HOST']
db_user = config['DB']['USER']
db_pass = config['DB'].get("PASS", "default password")
db_name = config['DB'].get("NAME", False)

# query 정보
test_query = config['QUERY']['TEST']


# MySQL 데이터베이스 연결
db = pymysql.connect(
	user    = db_user,
        passwd  = db_pass,
    	host    = db_host,
    	db      = db_name,
    	charset = 'utf8'
)

# 데이터에 접근
cursor = db.cursor(pymysql.cursors.DictCursor)

# SQL query 작성

sql = test_query

# SQL query 실행
cursor.execute(sql)

# db 데이터 가져오기

results = cursor.fetchmany(10) # n개의 데이터 가져오기 

for row in results:
    print(row)

# 수정 사항 db에 저장
# db.commit()
 
# Database 닫기
db.close()



