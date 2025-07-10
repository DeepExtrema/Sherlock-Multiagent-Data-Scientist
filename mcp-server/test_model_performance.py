#!/usr/bin/env python3
"""
Comprehensive test for model_performance_report tool.
Tests regression vs classification paths, metric extraction, HTML generation, and error handling.
"""

import asyncio
import json
import numpy as np
from pathlib import Path

# Import the server components
from server import (
    model_performance_report, debug_perf_summary,
    ModelPerformanceReportResult, DebugSummaryResult,
    REPORT_DIR
)

async def test_regression_performance():
    """Test regression model performance with Evidently integration."""
    print("🔍 TESTING REGRESSION PERFORMANCE")
    print("=" * 50)
    
    # Generate synthetic regression data with realistic patterns
    np.random.seed(42)
    n_samples = 1000
    
    # True values following a pattern
    y_true = np.random.normal(50, 15, n_samples)
    
    # Predictions with some error but correlated
    noise = np.random.normal(0, 5, n_samples)
    y_pred = y_true * 0.8 + 10 + noise  # Systematic bias + noise
    
    print(f"📊 Generated {n_samples} regression samples")
    print(f"📈 True values range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"📈 Pred values range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    # Test 1: Basic regression performance
    print("\n🧪 Test 1: Basic Regression Performance")
    try:
        result = await model_performance_report(
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(),
            model_type="regression"
        )
        
        print(f"✅ Success! Regression analysis completed")
        print(f"📄 Result type: {type(result)}")
        print(f"📊 Metrics keys: {list(result.metrics.keys())}")
        print(f"🔗 HTML URI: {result.html_uri}")
        
        # Analyze extracted metrics
        metrics = result.metrics
        print(f"\n📈 EXTRACTED METRICS:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value} ({type(value)})")
        
        # Verify HTML file exists
        html_path = Path(result.html_uri.replace('file://', ''))
        print(f"📄 HTML exists: {html_path.exists()}")
        if html_path.exists():
            print(f"📝 HTML file size: {html_path.stat().st_size} bytes")
        
        # Verify result structure
        assert isinstance(result, ModelPerformanceReportResult)
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'html_uri')
        assert isinstance(result.metrics, dict)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Different regression types
    print("\n🧪 Test 2: Various Regression Type Names")
    for reg_type in ["reg", "regression", "regressor"]:
        try:
            result = await model_performance_report(
                y_true=y_true[:100].tolist(),
                y_pred=y_pred[:100].tolist(), 
                model_type=reg_type
            )
            print(f"✅ '{reg_type}' handled as regression: {len(result.metrics)} metrics")
        except Exception as e:
            print(f"❌ '{reg_type}' failed: {e}")

async def test_classification_performance():
    """Test classification model performance with sklearn fallback."""
    print("\n\n🔍 TESTING CLASSIFICATION PERFORMANCE")
    print("=" * 50)
    
    # Generate synthetic classification data
    np.random.seed(42)
    n_samples = 1000
    
    # True labels (0, 1, 2 for 3-class problem)
    y_true = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    
    # Predictions with some accuracy but not perfect
    y_pred = y_true.copy()
    # Add some errors (flip ~20% of predictions)
    error_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    y_pred[error_indices] = np.random.choice([0, 1, 2], len(error_indices))
    
    print(f"📊 Generated {n_samples} classification samples")
    print(f"📈 True label distribution: {np.bincount(y_true)}")
    print(f"📈 Pred label distribution: {np.bincount(y_pred)}")
    print(f"🎯 Accuracy preview: {(y_true == y_pred).mean():.3f}")
    
    # Test 1: Basic classification performance
    print("\n🧪 Test 1: Basic Classification Performance")
    try:
        result = await model_performance_report(
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(),
            model_type="classification"
        )
        
        print(f"✅ Success! Classification analysis completed")
        print(f"📄 Result type: {type(result)}")
        print(f"📊 Metrics keys: {list(result.metrics.keys())}")
        print(f"🔗 HTML URI: {result.html_uri}")
        
        # Analyze sklearn-computed metrics
        metrics = result.metrics
        print(f"\n📈 SKLEARN METRICS:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value} ({type(value)})")
        
        # Check HTML content (should be simple, not Evidently)
        html_path = Path(result.html_uri.replace('file://', ''))
        print(f"📄 HTML exists: {html_path.exists()}")
        if html_path.exists():
            content = html_path.read_text()
            print(f"📝 HTML file size: {len(content)} characters")
            print(f"📝 Contains Evidently styling: {'evidently' in content.lower()}")
            print(f"📝 Contains basic HTML: {'<html>' in content}")
            
            # Show snippet of HTML content
            lines = content.split('\n')
            print(f"📝 HTML Preview (first 3 lines):")
            for i, line in enumerate(lines[:3]):
                print(f"    {i+1}: {line[:60]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Binary classification
    print("\n🧪 Test 2: Binary Classification")
    try:
        # Binary case
        y_true_binary = np.random.choice([0, 1], 500, p=[0.6, 0.4])
        y_pred_binary = y_true_binary.copy()
        # Add errors
        error_indices = np.random.choice(500, size=100, replace=False)
        y_pred_binary[error_indices] = 1 - y_pred_binary[error_indices]
        
        result = await model_performance_report(
            y_true=y_true_binary.tolist(),
            y_pred=y_pred_binary.tolist(),
            model_type="classification"
        )
        
        print(f"✅ Binary classification: {len(result.metrics)} metrics")
        print(f"📊 Binary accuracy: {result.metrics.get('accuracy', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"❌ Binary classification error: {e}")

async def test_debug_performance_summary():
    """Test debug performance summary for JSON structure analysis."""
    print("\n\n🔍 TESTING DEBUG PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    y_true = np.random.normal(0, 1, 100).tolist()
    y_pred = (np.array(y_true) + np.random.normal(0, 0.1, 100)).tolist()
    
    # Test 1: Debug regression summary
    print("🧪 Test 1: Debug Regression Summary")
    try:
        result = await debug_perf_summary(
            y_true=y_true,
            y_pred=y_pred,
            model_type="regression"
        )
        
        print(f"✅ Debug regression completed")
        print(f"🔑 Summary keys: {result.keys}")
        
        # Analyze the summary structure
        summary = result.summary
        print(f"\n📊 REGRESSION DEBUG ANALYSIS:")
        print(f"Top-level keys: {list(summary.keys())}")
        
        if 'metrics' in summary:
            metrics = summary['metrics']
            print(f"📈 Number of metrics: {len(metrics)}")
            print(f"📋 First 5 metrics:")
            
            for i, metric in enumerate(metrics[:5]):
                metric_id = metric.get('metric_id', 'unknown')
                metric_value = metric.get('value', 'unknown')
                print(f"  {i+1}. {metric_id}: {type(metric_value)}")
        
        # Save debug output
        debug_file = REPORT_DIR / 'debug_regression_analysis.json'
        with open(debug_file, 'w') as f:
            json.dump(result.summary, f, indent=2, default=str)
        print(f"💾 Debug output saved to: {debug_file}")
        
    except Exception as e:
        print(f"❌ Debug regression error: {e}")
    
         # Test 2: Debug classification summary
          print("\n🧪 Test 2: Debug Classification Summary")
     try:
         y_true_class = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
         y_pred_class = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
         
         result = await debug_perf_summary(
             y_true=y_true_class,
             y_pred=y_pred_class,
             model_type="classification"
         )
         
         print(f"✅ Debug classification completed")
         print(f"🔑 Summary keys: {result.keys}")
         
     except Exception as e:
         print(f"❌ Debug classification error: {e}")

async def test_error_scenarios():
    """Test error handling scenarios."""
    print("\n\n🔍 TESTING ERROR SCENARIOS")
    print("=" * 50)
    
    # Test 1: Length mismatch
    print("🧪 Test 1: Length Mismatch Error")
    try:
        result = await model_performance_report(
            y_true=[1, 2, 3, 4, 5],
            y_pred=[1, 2, 3],  # Different length
            model_type="regression"
        )
        print(f"⚠️ Unexpected success: {result}")
    except ValueError as e:
        print(f"✅ Expected ValueError caught: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
    
    # Test 2: Empty lists
    print("\n🧪 Test 2: Empty Lists")
    try:
        result = await model_performance_report(
            y_true=[],
            y_pred=[],
            model_type="classification"
        )
        print(f"⚠️ Empty lists handled: {result}")
    except Exception as e:
        print(f"❌ Empty lists error: {e}")
    
    # Test 3: Single value
    print("\n🧪 Test 3: Single Value")
    try:
        result = await model_performance_report(
            y_true=[1.0],
            y_pred=[1.1],
            model_type="regression"
        )
        print(f"✅ Single value handled: {len(result.metrics)} metrics")
    except Exception as e:
        print(f"❌ Single value error: {e}")
    
    # Test 4: Mixed types
    print("\n🧪 Test 4: Mixed Int/Float Types")
    try:
        result = await model_performance_report(
            y_true=[1, 2.5, 3, 4.2],  # Mixed int/float
            y_pred=[1.1, 2.3, 3.1, 4.0],
            model_type="regression"
        )
        print(f"✅ Mixed types handled: {len(result.metrics)} metrics")
    except Exception as e:
        print(f"❌ Mixed types error: {e}")

async def test_metric_structure_analysis():
    """Analyze the detailed structure of returned metrics."""
    print("\n\n🔍 TESTING METRIC STRUCTURE ANALYSIS")
    print("=" * 50)
    
    # Generate predictable data for analysis
    y_true_reg = [1.0, 2.0, 3.0, 4.0, 5.0] * 20  # 100 samples
    y_pred_reg = [1.1, 2.1, 2.9, 4.1, 4.9] * 20  # Close but not perfect
    
    y_true_class = [0, 1, 2] * 33 + [0]  # 100 samples, 3 classes
    y_pred_class = [0, 1, 2] * 30 + [0, 1, 1, 0, 1, 1, 0, 1, 2, 0]  # Some errors
    
    # Test 1: Regression metrics structure
    print("🧪 Test 1: Regression Metrics Structure")
    try:
        result = await model_performance_report(
            y_true=y_true_reg,
            y_pred=y_pred_reg,
            model_type="regression"
        )
        
        print(f"📊 REGRESSION METRICS STRUCTURE:")
        for key, value in result.metrics.items():
            print(f"  {key}:")
            print(f"    Type: {type(value)}")
            print(f"    Value: {value}")
            if isinstance(value, dict):
                print(f"    Subkeys: {list(value.keys())}")
        
    except Exception as e:
        print(f"❌ Regression metrics error: {e}")
    
    # Test 2: Classification metrics structure
    print("\n🧪 Test 2: Classification Metrics Structure")
    try:
        result = await model_performance_report(
            y_true=y_true_class,
            y_pred=y_pred_class,
            model_type="classification"
        )
        
        print(f"📊 CLASSIFICATION METRICS STRUCTURE:")
        for key, value in result.metrics.items():
            print(f"  {key}:")
            print(f"    Type: {type(value)}")
            print(f"    Value: {value}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"    Length: {len(value)}")
        
    except Exception as e:
        print(f"❌ Classification metrics error: {e}")

async def test_html_content_analysis():
    """Analyze the generated HTML content in detail."""
    print("\n\n🔍 TESTING HTML CONTENT ANALYSIS")
    print("=" * 50)
    
    # Generate test data
    y_true_reg = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred_reg = [1.1, 2.1, 2.9, 4.1, 4.9]
    
    y_true_class = [0, 1, 2, 0, 1]
    y_pred_class = [0, 1, 1, 0, 1]
    
    # Test 1: Regression HTML analysis
    print("🧪 Test 1: Regression HTML Content")
    try:
        result = await model_performance_report(
            y_true=y_true_reg,
            y_pred=y_pred_reg,
            model_type="regression"
        )
        
        html_path = Path(result.html_uri.replace('file://', ''))
        if html_path.exists():
            content = html_path.read_text()
            print(f"📄 Regression HTML Analysis:")
            print(f"  File size: {len(content)} characters")
            print(f"  Lines: {len(content.split('\n'))}")
            print(f"  Contains 'evidently': {'evidently' in content.lower()}")
            print(f"  Contains 'regression': {'regression' in content.lower()}")
            print(f"  Contains CSS: {'<style>' in content or '.css' in content}")
            print(f"  Contains JavaScript: {'<script>' in content or '.js' in content}")
            
    except Exception as e:
        print(f"❌ Regression HTML error: {e}")
    
    # Test 2: Classification HTML analysis
    print("\n🧪 Test 2: Classification HTML Content")
    try:
        result = await model_performance_report(
            y_true=y_true_class,
            y_pred=y_pred_class,
            model_type="classification"
        )
        
        html_path = Path(result.html_uri.replace('file://', ''))
        if html_path.exists():
            content = html_path.read_text()
            print(f"📄 Classification HTML Analysis:")
            print(f"  File size: {len(content)} characters")
            print(f"  Lines: {len(content.split('\n'))}")
            print(f"  Contains 'evidently': {'evidently' in content.lower()}")
            print(f"  Contains 'classification': {'classification' in content.lower()}")
            print(f"  Simple structure: {content.count('<') < 20}")
            
            # Show the actual content since it's manually generated
            print(f"📝 Manual HTML Content:")
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    print(f"    {i+1}: {line}")
            
    except Exception as e:
        print(f"❌ Classification HTML error: {e}")

async def main():
    """Run all tests sequentially."""
    print("🚀 STARTING MODEL PERFORMANCE REPORT TESTING")
    print("=" * 70)
    
    try:
        await test_regression_performance()
        await test_classification_performance()
        await test_debug_performance_summary()
        await test_error_scenarios()
        await test_metric_structure_analysis()
        await test_html_content_analysis()
        
        print("\n" + "=" * 70)
        print("🎉 ALL MODEL PERFORMANCE TESTS COMPLETED!")
        print("Check the reports/ directory for generated HTML files.")
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR IN TEST SUITE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 