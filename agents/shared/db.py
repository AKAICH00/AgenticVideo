"""
Database module for agent-database communication.

Usage:
    from shared.db import get_db, Database
    
    db = get_db()  # Singleton
    rows = await db.fetch_all("SELECT * FROM video_campaigns WHERE status = 'new'")
"""

import os
from typing import Optional, List, Any
from contextlib import asynccontextmanager
import asyncio

# Use synchronous psycopg2 wrapped in async for simplicity
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "")


class Database:
    """Async-compatible database wrapper."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or DATABASE_URL
        self._conn = None
    
    def _get_connection(self):
        """Get or create a database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.database_url)
            self._conn.autocommit = False
        return self._conn
    
    async def fetch_all(self, query: str, params: Optional[List[Any]] = None) -> List[dict]:
        """Execute query and return all rows as dicts."""
        def _exec():
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    result = cur.fetchall()
                    conn.commit()
                    return [dict(row) for row in result]
            except Exception as e:
                conn.rollback()
                raise
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _exec)
    
    async def fetch_one(self, query: str, params: Optional[List[Any]] = None) -> Optional[dict]:
        """Execute query and return first row as dict."""
        def _exec():
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    result = cur.fetchone()
                    conn.commit()
                    return dict(result) if result else None
            except Exception as e:
                conn.rollback()
                raise
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _exec)
    
    async def execute(self, query: str, params: Optional[List[Any]] = None):
        """Execute query without returning results."""
        def _exec():
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _exec)
    
    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None


# Singleton instance
_db_instance: Optional[Database] = None


def get_db() -> Database:
    """Get the singleton Database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


# Legacy compatibility functions
def get_db_connection():
    """Legacy: get raw psycopg2 connection."""
    return get_db()._get_connection()


def execute_query(query, params=None, fetch_one=False, fetch_all=False):
    """Legacy: execute query synchronously."""
    conn = get_db()._get_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        if fetch_one:
            result = cur.fetchone()
            conn.commit()
            return result
        if fetch_all:
            result = cur.fetchall()
            conn.commit()
            return result
        conn.commit()
        return None


def fetch_new_campaigns():
    """Legacy: fetch new campaigns."""
    return execute_query(
        "SELECT * FROM video_campaigns WHERE status = 'new'", 
        fetch_all=True
    )


def update_campaign_status(campaign_id, status, error=None):
    """Legacy: update campaign status."""
    query = "UPDATE video_campaigns SET status = %s"
    params = [status]
    if error:
        query += ", error_log = %s"
        params.append(error)
    query += " WHERE id = %s"
    params.append(campaign_id)
    execute_query(query, tuple(params))
