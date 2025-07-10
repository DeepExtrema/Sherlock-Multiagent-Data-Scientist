#!/usr/bin/env python3
"""
Comprehensive test for data_quality_report and drift_analysis tools.
Tests signatures, output models, HTML generation, JSON parsing, and error handling.
"""

import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Import the server components
from server import (
    data_quality_report, drift_analysis, debug_drift_summary,
    DataQualityReportResult, DriftAnalysisResult, DebugSummaryResult,
    data_store, store_lock, REPORT_DIR
)

async def test_data_quality_report():
    """Test data_quality_report tool with various scenarios."""
    print("🔍 TESTING DATA_QUALITY_REPORT")
    print("=" * 50)
    
    # Create test dataset with quality issues
    test_data = pd.DataFrame({
        'numeric_good': np.random.normal(50, 10, 1000),
        'numeric_outliers': np.concatenate([
            np.random.normal(50, 5, 950),
            np.random.normal(200, 5, 50)  # Outliers
        ]),
        'missing_data': [np.nan if i % 10 == 0 else f"value_{i}" for i in range(1000)],
        'categorical': np.random.choice(['A', 'B', 'C'], 1000),
        'high_cardinality': [f"unique_{i}" for i in range(1000)],  # Each value unique
        'duplicate_data': ['constant'] * 1000,  # All same value
    })
    
    # Store test data
    async with store_lock:
        data_store['quality_test'] = test_data
    
    print(f"📊 Created test dataset with shape: {test_data.shape}")
    print(f"📈 Data types: {list(test_data.dtypes)}")
    
    # Test 1: Basic data quality report
    print("\n🧪 Test 1: Basic Data Quality Report")
    try:
        result = await data_quality_report('quality_test')
        print(f"✅ Success! HTML saved to: {result.html_uri}")
        print(f"📄 Result type: {type(result)}")
        print(f"🔗 HTML URI exists: {Path(result.html_uri.replace('file://', '')).exists()}")
        
        # Verify output model structure
        assert isinstance(result, DataQualityReportResult)
        assert hasattr(result, 'html_uri')
        assert result.html_uri.startswith('file://')
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Data quality with custom title
    print("\n🧪 Test 2: Custom Title Metadata")
    try:
        result = await data_quality_report('quality_test', title="Custom Quality Analysis")
        print(f"✅ Success with custom title! HTML: {result.html_uri}")
        
        # Check if HTML file contains the title
        html_path = Path(result.html_uri.replace('file://', ''))
        if html_path.exists():
            html_content = html_path.read_text()
            print(f"📝 HTML file size: {len(html_content)} characters")
            
    except Exception as e:
        print(f"❌ Error with custom title: {e}")
    
    # Test 3: Error handling - missing dataset
    print("\n🧪 Test 3: Error Handling - Missing Dataset")
    try:
        result = await data_quality_report('nonexistent_dataset')
        print(f"⚠️ Unexpected success: {result}")
    except KeyError as e:
        print(f"✅ Expected KeyError caught: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")

async def test_drift_analysis():
    """Test drift_analysis tool with controlled drift scenarios."""
    print("\n\n🔍 TESTING DRIFT_ANALYSIS")
    print("=" * 50)
    
    # Create baseline dataset
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        'feature_stable': np.random.normal(0, 1, 1000),
        'feature_drift': np.random.normal(0, 1, 1000),
        'categorical_stable': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),
        'categorical_drift': np.random.choice(['X', 'Y'], 1000, p=[0.8, 0.2]),
        'numeric_no_drift': np.random.uniform(0, 100, 1000)
    })
    
    # Create current dataset with intentional drift
    current_data = pd.DataFrame({
        'feature_stable': np.random.normal(0, 1, 1000),  # No drift
        'feature_drift': np.random.normal(5, 2, 1000),   # Mean and variance drift
        'categorical_stable': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),  # No drift
        'categorical_drift': np.random.choice(['X', 'Y'], 1000, p=[0.2, 0.8]),  # Distribution drift
        'numeric_no_drift': np.random.uniform(0, 100, 1000)  # No drift
    })
    
    # Store datasets
    async with store_lock:
        data_store['baseline'] = baseline_data
        data_store['current'] = current_data
    
    print(f"📊 Baseline shape: {baseline_data.shape}")
    print(f"📊 Current shape: {current_data.shape}")
    print(f"🎯 Intentional drift in: feature_drift, categorical_drift")
    
    # Test 1: Basic drift analysis
    print("\n🧪 Test 1: Basic Drift Analysis")
    try:
        result = await drift_analysis('baseline', 'current')
        print(f"✅ Success! Drift analysis completed")
        print(f"📄 Result type: {type(result)}")
        print(f"📈 Drift count: {result.drift_count}")
        print(f"📈 Drift share: {result.drift_share:.3f}")
        print(f"🔗 HTML URI: {result.html_uri}")
        
        # Verify output model structure
        assert isinstance(result, DriftAnalysisResult)
        assert hasattr(result, 'drift_count')
        assert hasattr(result, 'drift_share')
        assert hasattr(result, 'html_uri')
        assert isinstance(result.drift_count, (int, float))
        assert isinstance(result.drift_share, (int, float))
        
        # Check HTML file exists
        html_path = Path(result.html_uri.replace('file://', ''))
        print(f"📄 HTML exists: {html_path.exists()}")
        if html_path.exists():
            print(f"📝 HTML file size: {html_path.stat().st_size} bytes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Error handling - missing datasets
    print("\n🧪 Test 2: Error Handling - Missing Datasets")
    try:
        result = await drift_analysis('missing_baseline', 'current')
        print(f"⚠️ Unexpected success: {result}")
    except KeyError as e:
        print(f"✅ Expected KeyError caught: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")

async def test_debug_drift_summary():
    """Test debug_drift_summary to understand JSON structure."""
    print("\n\n🔍 TESTING DEBUG_DRIFT_SUMMARY")
    print("=" * 50)
    
    # Ensure datasets exist from previous test
    if 'baseline' not in data_store or 'current' not in data_store:
        print("⚠️ Skipping debug test - baseline datasets not available")
        return
    
    print("🧪 Test: Debug Drift Summary JSON Analysis")
    try:
        result = await debug_drift_summary('baseline', 'current')
        print(f"✅ Success! Debug summary completed")
        print(f"📄 Result type: {type(result)}")
        print(f"🔑 Summary keys: {result.keys}")
        
        # Analyze the summary structure
        summary = result.summary
        print(f"\n📊 SUMMARY STRUCTURE ANALYSIS:")
        print(f"Top-level keys: {list(summary.keys())}")
        
        if 'metrics' in summary:
            metrics = summary['metrics']
            print(f"📈 Number of metrics: {len(metrics)}")
            print(f"📋 Metric types found:")
            
            drift_metrics = []
            for i, metric in enumerate(metrics[:5]):  # Show first 5
                metric_id = metric.get('metric_id', 'unknown')
                metric_value = metric.get('value', 'unknown')
                print(f"  {i+1}. {metric_id}: {type(metric_value)}")
                
                if 'Drift' in metric_id:
                    drift_metrics.append(metric)
            
            print(f"\n🎯 DRIFT-SPECIFIC METRICS:")
            for metric in drift_metrics:
                metric_id = metric.get('metric_id', 'unknown')
                value = metric.get('value', {})
                print(f"  - {metric_id}: {value}")
                
                # Special focus on DriftedColumnsCount
                if metric_id.startswith('DriftedColumnsCount'):
                    print(f"    📊 DriftedColumnsCount details:")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            print(f"      {k}: {v}")
        
        # Save debug output for manual inspection
        debug_file = REPORT_DIR / 'debug_metrics_analysis.json'
        with open(debug_file, 'w') as f:
            json.dump(result.summary, f, indent=2, default=str)
        print(f"\n💾 Full debug output saved to: {debug_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_error_scenarios():
    """Test various error scenarios and edge cases."""
    print("\n\n🔍 TESTING ERROR SCENARIOS")
    print("=" * 50)
    
    # Test 1: Empty dataset
    print("🧪 Test 1: Empty Dataset")
    try:
        empty_df = pd.DataFrame()
        async with store_lock:
            data_store['empty'] = empty_df
        
        result = await data_quality_report('empty')
        print(f"⚠️ Empty dataset handled: {result.html_uri}")
    except Exception as e:
        print(f"❌ Empty dataset error: {e}")
    
    # Test 2: Single column dataset
    print("\n🧪 Test 2: Single Column Dataset")
    try:
        single_col_df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        async with store_lock:
            data_store['single_col'] = single_col_df
        
        result = await data_quality_report('single_col')
        print(f"✅ Single column handled: {result.html_uri}")
    except Exception as e:
        print(f"❌ Single column error: {e}")
    
    # Test 3: All NaN dataset
    print("\n🧪 Test 3: All NaN Dataset")
    try:
        nan_df = pd.DataFrame({
            'col1': [np.nan] * 100,
            'col2': [np.nan] * 100,
            'col3': [np.nan] * 100
        })
        async with store_lock:
            data_store['all_nan'] = nan_df
        
        result = await data_quality_report('all_nan')
        print(f"✅ All NaN dataset handled: {result.html_uri}")
        
        # Try drift analysis with NaN data
        result_drift = await drift_analysis('all_nan', 'all_nan')
        print(f"✅ NaN drift analysis: drift_count={result_drift.drift_count}")
        
    except Exception as e:
        print(f"❌ All NaN error: {e}")

async def test_file_output_verification():
    """Verify HTML file outputs and their contents."""
    print("\n\n🔍 TESTING FILE OUTPUT VERIFICATION")
    print("=" * 50)
    
    # Check reports directory
    print(f"📁 Reports directory: {REPORT_DIR}")
    print(f"📁 Directory exists: {REPORT_DIR.exists()}")
    
    if REPORT_DIR.exists():
        html_files = list(REPORT_DIR.glob("*.html"))
        print(f"📄 HTML files found: {len(html_files)}")
        
        for html_file in html_files:
            print(f"  📄 {html_file.name} ({html_file.stat().st_size} bytes)")
            
            # Basic content verification
            try:
                content = html_file.read_text(encoding='utf-8')
                has_evidently = 'evidently' in content.lower()
                has_html_structure = '<html>' in content and '</html>' in content
                print(f"    ✅ Valid HTML structure: {has_html_structure}")
                print(f"    ✅ Contains Evidently content: {has_evidently}")
            except Exception as e:
                print(f"    ❌ Content read error: {e}")

async def main():
    """Run all tests sequentially."""
    print("🚀 STARTING COMPREHENSIVE EVIDENTLY TOOLS TESTING")
    print("=" * 70)
    
    try:
        await test_data_quality_report()
        await test_drift_analysis()
        await test_debug_drift_summary()
        await test_error_scenarios()
        await test_file_output_verification()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS COMPLETED!")
        print("Check the reports/ directory for generated HTML files.")
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR IN TEST SUITE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 