#!/usr/bin/env python3

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import os

class DatabaseManager:
    def __init__(self, db_path: str = "soil_analysis.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    full_name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    profile_picture TEXT,
                    bio TEXT
                )
            ''')
            
            # Analysis history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    location_name VARCHAR(200) NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    analysis_data TEXT NOT NULL,
                    primary_crop VARCHAR(50),
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # User preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE,
                    preferred_units VARCHAR(20) DEFAULT 'metric',
                    default_location VARCHAR(200),
                    email_notifications BOOLEAN DEFAULT 1,
                    map_style VARCHAR(20) DEFAULT 'terrain',
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create default user if none exists
            cursor.execute('SELECT COUNT(*) FROM users')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO users (username, email, full_name, bio)
                    VALUES (?, ?, ?, ?)
                ''', ('demo_user', 'demo@soilanalysis.com', 'Demo User', 
                      'Agricultural enthusiast exploring soil analysis technology'))
                
                user_id = cursor.lastrowid
                cursor.execute('''
                    INSERT INTO user_preferences (user_id) VALUES (?)
                ''', (user_id,))
            
            conn.commit()

    def create_user(self, username: str, email: str, full_name: str = None, bio: str = None) -> Optional[int]:
        """Create a new user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (username, email, full_name, bio)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, full_name, bio))
                
                user_id = cursor.lastrowid
                
                # Create default preferences
                cursor.execute('''
                    INSERT INTO user_preferences (user_id) VALUES (?)
                ''', (user_id,))
                
                conn.commit()
                return user_id
        except sqlite3.IntegrityError:
            return None

    def get_user(self, user_id: int = None, username: str = None) -> Optional[Dict]:
        """Get user by ID or username"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            elif username:
                cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            else:
                return None
                
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_default_user(self) -> Dict:
        """Get the default demo user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users LIMIT 1')
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            conn.commit()

    def save_analysis(self, user_id: int, location_name: str, latitude: float, 
                     longitude: float, analysis_data: Dict, primary_crop: str, 
                     confidence: float) -> int:
        """Save analysis result to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analyses 
                (user_id, location_name, latitude, longitude, analysis_data, primary_crop, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, location_name, latitude, longitude, 
                  json.dumps(analysis_data), primary_crop, confidence))
            
            analysis_id = cursor.lastrowid
            conn.commit()
            return analysis_id

    def get_user_analyses(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get user's analysis history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM analyses 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            analyses = []
            for row in cursor.fetchall():
                analysis = dict(row)
                analysis['analysis_data'] = json.loads(analysis['analysis_data'])
                analyses.append(analysis)
            
            return analyses

    def get_analysis_locations(self, user_id: int) -> List[Dict]:
        """Get all analyzed locations for map display"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, location_name, latitude, longitude, primary_crop, 
                       confidence, created_at
                FROM analyses 
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            
            return [dict(row) for row in cursor.fetchall()]

    def get_user_preferences(self, user_id: int) -> Optional[Dict]:
        """Get user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM user_preferences WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_user_preferences(self, user_id: int, preferences: Dict):
        """Update user preferences"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build dynamic update query based on provided preferences
            fields = []
            values = []
            for key, value in preferences.items():
                if key in ['preferred_units', 'default_location', 'email_notifications', 'map_style']:
                    fields.append(f"{key} = ?")
                    values.append(value)
            
            if fields:
                values.append(user_id)
                query = f"UPDATE user_preferences SET {', '.join(fields)} WHERE user_id = ?"
                cursor.execute(query, values)
                conn.commit()

    def get_dashboard_stats(self, user_id: int) -> Dict:
        """Get dashboard statistics for user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total analyses
            cursor.execute('SELECT COUNT(*) FROM analyses WHERE user_id = ?', (user_id,))
            total_analyses = cursor.fetchone()[0]
            
            # Unique locations
            cursor.execute('''
                SELECT COUNT(DISTINCT location_name) FROM analyses WHERE user_id = ?
            ''', (user_id,))
            unique_locations = cursor.fetchone()[0]
            
            # Most recommended crop
            cursor.execute('''
                SELECT primary_crop, COUNT(*) as count 
                FROM analyses WHERE user_id = ? 
                GROUP BY primary_crop 
                ORDER BY count DESC 
                LIMIT 1
            ''', (user_id,))
            result = cursor.fetchone()
            most_recommended_crop = result[0] if result else 'None'
            
            # Recent analysis date
            cursor.execute('''
                SELECT created_at FROM analyses 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (user_id,))
            result = cursor.fetchone()
            last_analysis = result[0] if result else None
            
            return {
                'total_analyses': total_analyses,
                'unique_locations': unique_locations,
                'most_recommended_crop': most_recommended_crop,
                'last_analysis': last_analysis
            }

if __name__ == "__main__":
    # Test the database
    db = DatabaseManager()
    print("Database initialized successfully!")
    
    # Test user creation
    user = db.get_default_user()
    print(f"Default user: {user}")
    
    # Test stats
    stats = db.get_dashboard_stats(user['id'])
    print(f"Dashboard stats: {stats}")
