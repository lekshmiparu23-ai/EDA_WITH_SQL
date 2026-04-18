"""
db_connector.py — SQL Database Integration for DataLens
Supports MySQL and PostgreSQL connections.
"""

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, inspect
import pymysql

def get_mysql_engine(host, port, user, password, database):
    """
    Create SQLAlchemy engine for MySQL connection.
    
    Parameters:
        host, port, user, password, database: MySQL credentials
    
    Returns:
        SQLAlchemy engine or None on failure
    """
    try:
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(url, connect_args={"connect_timeout": 10})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"MySQL Connection Failed: {e}")
        return None

def get_postgres_engine(host, port, user, password, database):
    """Create SQLAlchemy engine for PostgreSQL."""
    try:
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(url, connect_args={"connect_timeout": 10})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"PostgreSQL Connection Failed: {e}")
        return None

def list_tables(engine):
    """Return list of table names in the connected database."""
    try:
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception as e:
        st.error(f"Error fetching tables: {e}")
        return []

def load_table(engine, table_name, limit=10000):
    """
    Load a full table as DataFrame.
    
    Parameters:
        engine: SQLAlchemy engine
        table_name (str): Table to load
        limit (int): Max rows to fetch (default 10000)
    
    Returns:
        pd.DataFrame
    """
    try:
        # Proper quoting for table name based on engine dialect
        if engine.dialect.name == 'mysql':
            query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
        elif engine.dialect.name == 'postgresql':
            query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
        else:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error loading table: {e}")
        return None

def run_custom_query(engine, query):
    """
    Execute a custom SQL SELECT query and return DataFrame.
    
    Parameters:
        engine: SQLAlchemy engine
        query (str): SQL SELECT statement
    
    Returns:
        pd.DataFrame or None
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Query Error: {e}")
        return None
