#!/usr/bin/env python3

import asyncio
import server
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

async def test_infer_schema():
    """Test the infer_schema tool with comprehensive data types and patterns."""
    
    print("🔍 Testing Infer Schema Tool")
    print("=" * 60)
    
    # Create comprehensive test dataset with various patterns
    print("Creating comprehensive test dataset...")
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        # Numeric columns
        'id_sequential': range(1, 101),  # Should be detected as ID
        'price': np.random.lognormal(3, 1, 100),  # Numeric with precision/scale
        'quantity': np.random.randint(1, 100, 100),  # Integer
        'rating': np.random.uniform(1.0, 5.0, 100),  # Float with decimal places
        
        # String columns with patterns
        'email': [f'user{i}@example.com' for i in range(100)],
        'phone': [f'+1-555-{100 + i:04d}' for i in range(100)],
        'website': [f'https://site{i}.com' for i in range(100)],
        'uuid_col': [f'{i:08x}-{i:04x}-{i:04x}-{i:04x}-{i:012x}' for i in range(100)],
        'postal_code': [f'{10000 + i:05d}' for i in range(100)],
        'ip_address': [f'192.168.{i//25}.{i%25}' for i in range(100)],
        'credit_card': [f'4532 1234 5678 {1000 + i:04d}' for i in range(100)],
        
        # Date patterns
        'date_iso': pd.date_range('2023-01-01', periods=100).strftime('%Y-%m-%d'),
        'date_us': pd.date_range('2023-01-01', periods=100).strftime('%m/%d/%Y'),
        
        # Categorical with different cardinalities
        'category_low': np.random.choice(['A', 'B', 'C'], 100),  # Low cardinality
        'category_high': [f'cat_{i%50}' for i in range(100)],  # High cardinality
        
        # Columns with missing data
        'text_with_missing': ['text_' + str(i) if i % 10 != 0 else None for i in range(100)],
        'numeric_with_missing': [float(i) if i % 8 != 0 else None for i in range(100)],
        
        # Edge cases
        'all_unique': [f'unique_{i}' for i in range(100)],  # 100% unique
        'mostly_null': [i if i < 5 else None for i in range(100)],  # 95% null
        'constant': ['same_value'] * 100,  # No variance
    })
    
    test_data.to_csv('test_schema_data.csv', index=False)
    
    # Load the data
    load_result = await server.load_data('test_schema_data.csv', 'schema_test')
    print(f"✅ {load_result}")
    
    # Run schema inference
    print("\nRunning schema inference...")
    result = await server.infer_schema('schema_test')
    
    print(f"\n📊 SCHEMA INFERENCE RESULTS")
    print("=" * 60)
    print(f"Dataset: {result.dataset_name}")
    print(f"Shape: {result.total_rows}x{result.total_columns}")
    print(f"Analyzed columns: {len(result.columns)}")
    
    # Analyze each column type
    column_types = {}
    pattern_counts = {}
    id_columns = []
    
    print(f"\n📋 COLUMN-BY-COLUMN ANALYSIS")
    print("-" * 60)
    
    for i, col in enumerate(result.columns):
        print(f"{i+1:2d}. {col.name:<20} | Type: {col.type:<8} | Nullable: {col.nullable}")
        
        # Count types
        column_types[col.type] = column_types.get(col.type, 0) + 1
        
        # Count patterns
        if col.pattern:
            pattern_counts[col.pattern] = pattern_counts.get(col.pattern, 0) + 1
            print(f"    Pattern: {col.pattern}")
        
        # Check for additional metadata (these are added dynamically)
        if hasattr(col, 'is_id_column') and col.is_id_column:
            id_columns.append(col.name)
            print(f"    🔑 ID Column detected")
        
        if hasattr(col, 'uniqueness_ratio'):
            print(f"    Uniqueness: {col.uniqueness_ratio:.3f}")
        
        if col.min_value is not None and col.max_value is not None:
            print(f"    Range: {col.min_value:.3f} to {col.max_value:.3f}")
        
        if col.max_length is not None:
            print(f"    Max Length: {col.max_length}")
        
        if col.sample_values:
            sample_str = str(col.sample_values[:3]).replace("'", "")
            print(f"    Samples: {sample_str}...")
        
        if hasattr(col, 'data_quality'):
            dq = col.data_quality
            print(f"    Missing: {dq['missing_percentage']:.1f}%, Duplicates: {dq['duplicate_percentage']:.1f}%")
        
        print()
    
    # Summary statistics
    print(f"📊 SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Column Types:")
    for col_type, count in column_types.items():
        print(f"  {col_type}: {count}")
    
    print(f"\nPattern Detection:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count}")
    
    print(f"\nID Columns: {id_columns}")
    
    # Check if YAML file was created
    yaml_path = Path('reports') / f'schema_contract_schema_test.yaml'
    if yaml_path.exists():
        print(f"\n📄 YAML CONTRACT ANALYSIS")
        print("-" * 40)
        with open(yaml_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        print(f"YAML file created: {yaml_path}")
        print(f"Dataset name: {yaml_content.get('dataset_name')}")
        print(f"Version: {yaml_content.get('version')}")
        print(f"Created at: {yaml_content.get('created_at')}")
        
        stats = yaml_content.get('statistics', {})
        print(f"Statistics:")
        print(f"  Total rows: {stats.get('total_rows')}")
        print(f"  Total columns: {stats.get('total_columns')}")
        print(f"  ID columns: {stats.get('id_columns')}")
        print(f"  High cardinality: {stats.get('high_cardinality_columns')}")
        print(f"  Data quality score: {stats.get('data_quality_score')}%")
        
        # Show a sample column contract
        columns = yaml_content.get('columns', [])
        if columns:
            print(f"\nSample Column Contract (first column):")
            print(yaml.dump(columns[0], default_flow_style=False, indent=2))
    
    # Test error cases
    print(f"\n🧪 TESTING ERROR CASES")
    print("-" * 40)
    
    try:
        await server.infer_schema('nonexistent_dataset')
        print("❌ Expected KeyError for nonexistent dataset")
    except KeyError as e:
        print(f"✅ Correctly handled nonexistent dataset: {e}")
    
    # Test with empty dataset
    empty_data = pd.DataFrame()
    empty_data.to_csv('test_empty.csv', index=False)
    
    try:
        await server.load_data('test_empty.csv', 'empty')
        empty_result = await server.infer_schema('empty')
        print(f"✅ Empty dataset handled: {len(empty_result.columns)} columns")
    except Exception as e:
        print(f"⚠️  Empty dataset error: {e}")

if __name__ == "__main__":
    asyncio.run(test_infer_schema()) 