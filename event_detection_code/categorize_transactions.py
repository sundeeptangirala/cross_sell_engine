"""
TRANSACTION CATEGORIZATION PIPELINE
====================================

Complete pipeline to:
1. Extract transactions from SQL Server
2. Apply hybrid categorization
3. Store results back to database

Uses correct field names:
- UCIC (customer_id)
- DEPTRANS_POST_DATE (transaction_date)
- DEPTRANS_AMOUNT (amount)
- CLEANSED (description)
- DEPTRANS_PAYEE (contains MCC after ~)

Author: Sundeep Tangirala
Date: 2026-02-06
"""

import pandas as pd
import pyodbc
from hybrid_categorizer import HybridTransactionCategorizer
from datetime import datetime
import sys


def extract_mcc_from_payee(payee_string):
    """
    Extract MCC code from DEPTRANS_PAYEE field
    MCC is the last 4 characters after the last '~'
    
    Example: "WALMART~5411" → "5411"
    """
    if not payee_string or pd.isna(payee_string):
        return None
    
    try:
        # Clean the string
        cleaned = str(payee_string).replace(' ', '').replace('-', '~').replace('_', '~')
        
        # Find last occurrence of ~
        if '~' in cleaned:
            parts = cleaned.split('~')
            mcc = parts[-1]
            
            # Validate it's 4 digits
            if len(mcc) == 4 and mcc.isdigit():
                return mcc
    except:
        pass
    
    return None


def load_transactions(connection_string, start_date='2025-01-01', limit=None):
    """
    Load transactions from SQL Server
    
    Args:
        connection_string: SQL Server connection string
        start_date: Only load transactions after this date
        limit: Limit number of rows (for testing)
    
    Returns:
        DataFrame with transactions
    """
    
    print("Connecting to SQL Server...")
    conn = pyodbc.connect(connection_string)
    
    # Build query
    query = f"""
        SELECT 
            UCIC as customer_id,
            DEPTRANS_POST_DATE as transaction_date,
            DEPTRANS_AMOUNT as amount,
            CLEANSED as description,
            DEPTRANS_TYPE_DESC as txn_type,
            DEPTRANS_PAYEE as payee_raw,
            PROCESS_DATE
        FROM SEGEDW.SEG_E0.SAT_SEG_DEP_TRAN
        WHERE DEPTRANS_POST_DATE >= '{start_date}'
            AND PROCESS_DATE > '12-12-2025'
            AND DEPTRANS_AMOUNT > 0
            AND CLEANSED IS NOT NULL
            AND DEPTRANS_PAYEE NOT LIKE '%*%'
    """
    
    if limit:
        query = f"SELECT TOP {limit} * FROM ({query}) AS subquery"
    
    print(f"Loading transactions from {start_date}...")
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} transactions")
    
    # Extract MCC from DEPTRANS_PAYEE
    print("Extracting MCC codes...")
    df['mcc'] = df['payee_raw'].apply(extract_mcc_from_payee)
    
    mcc_count = df['mcc'].notna().sum()
    mcc_pct = (mcc_count / len(df) * 100)
    print(f"MCC coverage: {mcc_count:,} / {len(df):,} ({mcc_pct:.1f}%)")
    
    return df


def categorize_transactions(df, categorizer):
    """
    Apply hybrid categorization to transaction DataFrame
    
    Args:
        df: DataFrame with transactions
        categorizer: HybridTransactionCategorizer instance
    
    Returns:
        DataFrame with added category columns
    """
    
    print("\nApplying hybrid categorization...")
    
    # Categorize using the hybrid approach
    df = categorizer.categorize_dataframe(
        df, 
        description_col='description',
        mcc_col='mcc'
    )
    
    return df


def save_to_database(df, connection_string, table_name='DATASCIENCE.tier3_transaction_categories'):
    """
    Save categorized transactions to database
    
    Args:
        df: Categorized DataFrame
        connection_string: SQL connection
        table_name: Target table name
    """
    
    print(f"\nSaving results to {table_name}...")
    
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    
    # Create table if not exists
    create_table_sql = f"""
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'tier3_transaction_categories')
    BEGIN
        CREATE TABLE {table_name} (
            customer_id VARCHAR(50),
            transaction_date DATE,
            amount DECIMAL(18,2),
            description VARCHAR(500),
            txn_type VARCHAR(50),
            mcc VARCHAR(4),
            category VARCHAR(50),
            category_confidence DECIMAL(5,2),
            categorization_method VARCHAR(20),
            categorized_date DATETIME DEFAULT GETDATE()
        );
        
        CREATE INDEX idx_customer ON {table_name}(customer_id);
        CREATE INDEX idx_category ON {table_name}(category);
        CREATE INDEX idx_txn_date ON {table_name}(transaction_date);
    END
    """
    
    cursor.execute(create_table_sql)
    conn.commit()
    
    # Insert data in batches
    batch_size = 1000
    total_rows = len(df)
    
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Prepare insert statement
        insert_sql = f"""
            INSERT INTO {table_name} 
            (customer_id, transaction_date, amount, description, txn_type, 
             mcc, category, category_confidence, categorization_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Prepare data
        data = [
            (
                row['customer_id'],
                row['transaction_date'],
                row['amount'],
                row['description'],
                row['txn_type'],
                row['mcc'],
                row['category'],
                row['category_confidence'],
                row['categorization_method']
            )
            for _, row in batch.iterrows()
        ]
        
        cursor.executemany(insert_sql, data)
        conn.commit()
        
        if (i + batch_size) % 10000 == 0:
            print(f"  Inserted {i + batch_size:,} / {total_rows:,} rows...")
    
    print(f"Successfully saved {total_rows:,} transactions!")
    
    conn.close()


def generate_category_report(df):
    """
    Generate summary report of categorization results
    """
    
    print("\n" + "="*70)
    print("CATEGORIZATION REPORT")
    print("="*70)
    
    # Overall statistics
    print(f"\nTotal Transactions: {len(df):,}")
    print(f"Date Range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"Total Amount: ${df['amount'].sum():,.2f}")
    
    # By category
    print("\nTop 20 Categories:")
    category_stats = df.groupby('category').agg({
        'customer_id': 'count',
        'amount': 'sum',
        'category_confidence': 'mean'
    }).round(2)
    category_stats.columns = ['Transactions', 'Total_Amount', 'Avg_Confidence']
    category_stats = category_stats.sort_values('Transactions', ascending=False).head(20)
    print(category_stats)
    
    # By method
    print("\nBy Categorization Method:")
    method_stats = df.groupby('categorization_method').agg({
        'customer_id': 'count',
        'category_confidence': 'mean'
    }).round(2)
    method_stats.columns = ['Count', 'Avg_Confidence']
    print(method_stats)
    
    # Coverage
    categorized = (df['category'] != 'unknown').sum()
    coverage = (categorized / len(df) * 100)
    print(f"\nOverall Coverage: {coverage:.1f}%")
    print(f"  Categorized: {categorized:,}")
    print(f"  Unknown: {len(df) - categorized:,}")
    
    print("="*70)


def main():
    """
    Main execution pipeline
    """
    
    print("="*70)
    print("TRANSACTION CATEGORIZATION PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=VSDSESOLDEV03;DATABASE=SEGEDW;Trusted_Connection=yes;"
    
    # STEP 1: Initialize categorizer
    print("STEP 1: Initializing hybrid categorizer...")
    categorizer = HybridTransactionCategorizer()
    
    # STEP 2: Load transactions
    print("\nSTEP 2: Loading transactions...")
    
    # For testing, limit to 10,000 transactions
    # Remove limit for production
    df = load_transactions(
        connection_string,
        start_date='2025-01-01',
        limit=10000  # Remove this for full run
    )
    
    # STEP 3: Train ML model (if not already trained)
    print("\nSTEP 3: Training ML model...")
    
    # Load MCC-labeled data for training
    training_df = load_transactions(
        connection_string,
        start_date='2024-01-01',
        limit=50000  # Sample for training
    )
    
    # Filter to only rows with MCC
    training_df = training_df[training_df['mcc'].notna()]
    
    if len(training_df) > 0:
        categorizer.train_ml_model(transactions_df=training_df)
        # Save model for future use
        categorizer.save_model('/home/claude/categorizer_model.pkl')
    else:
        print("No MCC-labeled data found for training. Skipping ML training.")
    
    # STEP 4: Categorize transactions
    print("\nSTEP 4: Categorizing transactions...")
    df = categorize_transactions(df, categorizer)
    
    # STEP 5: Generate report
    print("\nSTEP 5: Generating report...")
    generate_category_report(df)
    
    # STEP 6: Save to database
    print("\nSTEP 6: Saving to database...")
    save_to_database(df, connection_string)
    
    # STEP 7: Save to CSV (for review)
    output_file = f'/home/claude/categorized_transactions_{datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults also saved to: {output_file}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
