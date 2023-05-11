import sqlite3
from IMD2SQL import dbPath


def main():
    conn = sqlite3.connect(dbPath())
    cursor = conn.cursor()
    cursor.execute("ALTER TABLE table_name MODIFY COLUMN Month DATETIME")
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
    
