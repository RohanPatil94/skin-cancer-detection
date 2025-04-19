# database.py
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_user(name, email, password):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    try:
        hashed_password = generate_password_hash(password)
        c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                 (name, email, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(email, password):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[3], password):
        return {'id': user[0], 'name': user[1], 'email': user[2]}
    return None

def get_user_by_id(user_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    
    if user:
        return {'id': user[0], 'name': user[1], 'email': user[2]}
    return None