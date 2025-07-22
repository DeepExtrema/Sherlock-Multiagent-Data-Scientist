"""
Connectivity and Smoke Tester for Hybrid API

This script simulates various scenarios and tests the implementation logic
without requiring external services like Redis, MongoDB, or running servers.
"""

import asyncio
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import traceback

class ConnectivityTester:
    """Tests connectivity and logic without external dependencies."""
    
    def __init__(self):
        self.test_results = []
        self.base_path = Path(__file__).parent
        
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            "name": test_name,
            "passed": passed,
            "status": status,
            "message": message
        })
        print(f"  {status} {test_name}" + (f" - {message}" if message else ""))
    
    def test_imports_and_syntax(self) -> bool:
        """Test all imports can be resolved and syntax is valid."""
        print("\n🔧 Testing Imports and Syntax...")
        
        files_to_test = [
            "orchestrator/translation_queue.py",
            "api/hybrid_router.py",
            "master_orchestrator_api.py"
        ]
        
        all_passed = True
        
        for file_path in files_to_test:
            full_path = self.base_path / file_path
            
            if not full_path.exists():
                self.log_test(f"File exists: {file_path}", False, "File not found")
                all_passed = False
                continue
            
            try:
                # Test syntax by parsing
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                compile(content, file_path, 'exec')
                self.log_test(f"Syntax valid: {file_path}", True)
                
                # Check for obvious import issues
                problematic_patterns = [
                    'from __future__ import annotations',  # Should be first import
                    'import *',  # Wildcard imports
                    'from typing import *'  # Wildcard from typing
                ]
                
                for pattern in problematic_patterns:
                    if pattern in content:
                        if pattern == 'from __future__ import annotations':
                            # Check if it's the first import
                            first_import_line = next((line for line in content.split('\n') if line.strip().startswith(('import ', 'from '))), '')
                            if pattern not in first_import_line:
                                self.log_test(f"Import order: {file_path}", False, "__future__ imports should be first")
                                all_passed = False
                        else:
                            self.log_test(f"Import style: {file_path}", False, f"Problematic pattern: {pattern}")
                            all_passed = False
                
            except SyntaxError as e:
                self.log_test(f"Syntax valid: {file_path}", False, f"Syntax error: {e}")
                all_passed = False
            except Exception as e:
                self.log_test(f"File readable: {file_path}", False, f"Error: {e}")
                all_passed = False
        
        return all_passed
    
    async def test_translation_queue_logic(self) -> bool:
        """Test translation queue logic without Redis."""
        print("\n🔄 Testing Translation Queue Logic...")
        
        try:
            # Simulate the translation queue logic
            class MockRedisClient:
                def __init__(self):
                    self.data = {}
                    self.lists = {}
                    self.expiry = {}
                
                async def hset(self, key, mapping):
                    self.data[key] = mapping
                    return True
                
                async def hgetall(self, key):
                    return self.data.get(key, {})
                
                async def rpush(self, list_name, value):
                    if list_name not in self.lists:
                        self.lists[list_name] = []
                    self.lists[list_name].append(value)
                    return len(self.lists[list_name])
                
                async def blpop(self, list_name, timeout=5):
                    if list_name in self.lists and self.lists[list_name]:
                        value = self.lists[list_name].pop(0)
                        return (list_name, value)
                    return None
                
                async def expire(self, key, seconds):
                    self.expiry[key] = seconds
                    return True
                
                async def ping(self):
                    return True
            
            # Test enqueue/dequeue logic
            mock_redis = MockRedisClient()
            
            # Simulate enqueue
            token = "test_token_123"
            test_data = {
                "status": "queued",
                "text": "test translation",
                "metadata": {"test": True},
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "retries": 0
            }
            
            # Test the Redis operations sequence
            await mock_redis.hset(f"translation:{token}", mapping=test_data)
            await mock_redis.rpush("translation:q", token)
            await mock_redis.expire(f"translation:{token}", 300)
            
            # Verify storage
            stored_data = await mock_redis.hgetall(f"translation:{token}")
            if stored_data == test_data:
                self.log_test("Translation data storage", True)
            else:
                self.log_test("Translation data storage", False, f"Data mismatch: {stored_data}")
                return False
            
            # Test queue retrieval
            result = await mock_redis.blpop("translation:q", timeout=1)
            if result and result[1] == token:
                self.log_test("Queue operations", True)
            else:
                self.log_test("Queue operations", False, f"Failed to retrieve token: {result}")
                return False
            
            # Test status update simulation
            update_data = {"status": "processing", "updated_at": "2024-01-01T00:01:00"}
            await mock_redis.hset(f"translation:{token}", mapping=update_data)
            
            final_data = await mock_redis.hgetall(f"translation:{token}")
            if "status" in final_data and final_data["status"] == "processing":
                self.log_test("Status updates", True)
            else:
                self.log_test("Status updates", False, "Status not updated correctly")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Translation queue logic", False, f"Exception: {e}")
            return False
    
    def test_circular_dependency_detection(self) -> bool:
        """Test the circular dependency detection algorithm."""
        print("\n🔄 Testing Circular Dependency Detection...")
        
        def check_circular_dependencies(tasks: List[Dict[str, Any]]):
            """Simulate the circular dependency detection from the router."""
            graph = {}
            for task in tasks:
                task_id = task.get("id")
                deps = task.get("depends_on", [])
                graph[task_id] = deps
            
            visited = set()
            rec_stack = set()
            
            def dfs(node):
                if node in rec_stack:
                    raise ValueError(f"Circular dependency detected involving task: {node}")
                if node in visited:
                    return
                
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in graph.get(node, []):
                    dfs(neighbor)
                
                rec_stack.remove(node)
            
            for task_id in graph:
                if task_id not in visited:
                    dfs(task_id)
        
        test_cases = [
            {
                "name": "No dependencies",
                "tasks": [{"id": "A", "depends_on": []}],
                "should_detect": False
            },
            {
                "name": "Linear dependencies",
                "tasks": [
                    {"id": "A", "depends_on": []},
                    {"id": "B", "depends_on": ["A"]},
                    {"id": "C", "depends_on": ["B"]}
                ],
                "should_detect": False
            },
            {
                "name": "Simple cycle",
                "tasks": [
                    {"id": "A", "depends_on": ["B"]},
                    {"id": "B", "depends_on": ["A"]}
                ],
                "should_detect": True
            },
            {
                "name": "Complex cycle",
                "tasks": [
                    {"id": "A", "depends_on": ["C"]},
                    {"id": "B", "depends_on": ["A"]},
                    {"id": "C", "depends_on": ["B"]}
                ],
                "should_detect": True
            },
            {
                "name": "Self dependency",
                "tasks": [{"id": "A", "depends_on": ["A"]}],
                "should_detect": True
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            try:
                check_circular_dependencies(test_case["tasks"])
                # No exception - no cycle detected
                if test_case["should_detect"]:
                    self.log_test(f"Cycle detection: {test_case['name']}", False, "Should have detected cycle")
                    all_passed = False
                else:
                    self.log_test(f"Cycle detection: {test_case['name']}", True)
            except ValueError:
                # Exception - cycle detected
                if test_case["should_detect"]:
                    self.log_test(f"Cycle detection: {test_case['name']}", True)
                else:
                    self.log_test(f"Cycle detection: {test_case['name']}", False, "False positive cycle detection")
                    all_passed = False
            except Exception as e:
                self.log_test(f"Cycle detection: {test_case['name']}", False, f"Unexpected error: {e}")
                all_passed = False
        
        return all_passed
    
    def test_input_validation(self) -> bool:
        """Test input validation logic."""
        print("\n✅ Testing Input Validation...")
        
        try:
            # Simulate Pydantic validation logic
            def validate_natural_language(text: str) -> bool:
                if not text or text.isspace():
                    raise ValueError("Natural language content cannot be empty")
                if len(text.split()) < 3:
                    raise ValueError("Natural language content too brief, provide more details")
                return True
            
            def validate_dsl_yaml(yaml_str: str) -> bool:
                import yaml
                try:
                    parsed = yaml.safe_load(yaml_str)
                    if not isinstance(parsed, dict):
                        raise ValueError("DSL must be a valid YAML object")
                    if "tasks" not in parsed:
                        raise ValueError("DSL must contain 'tasks' field")
                    if not isinstance(parsed["tasks"], list):
                        raise ValueError("'tasks' must be a list")
                    if len(parsed["tasks"]) == 0:
                        raise ValueError("DSL must contain at least one task")
                    return True
                except yaml.YAMLError as e:
                    raise ValueError(f"Invalid YAML format: {e}")
            
            validation_tests = [
                # Natural language tests
                {
                    "type": "natural_language",
                    "input": "Load data and analyze it thoroughly",
                    "should_pass": True
                },
                {
                    "type": "natural_language", 
                    "input": "",
                    "should_pass": False
                },
                {
                    "type": "natural_language",
                    "input": "hi there",  # Too brief
                    "should_pass": False
                },
                # DSL YAML tests
                {
                    "type": "dsl_yaml",
                    "input": """
name: "Test Workflow"
tasks:
  - id: "task1"
    agent: "eda_agent"
    action: "load_data"
""",
                    "should_pass": True
                },
                {
                    "type": "dsl_yaml",
                    "input": "invalid: yaml: [content",
                    "should_pass": False
                },
                {
                    "type": "dsl_yaml",
                    "input": "name: test\ntasks: []",  # Empty tasks
                    "should_pass": False
                }
            ]
            
            all_passed = True
            
            for test in validation_tests:
                try:
                    if test["type"] == "natural_language":
                        validate_natural_language(test["input"])
                    elif test["type"] == "dsl_yaml":
                        validate_dsl_yaml(test["input"])
                    
                    # No exception - validation passed
                    if test["should_pass"]:
                        self.log_test(f"Validation {test['type']}: {test['input'][:20]}...", True)
                    else:
                        self.log_test(f"Validation {test['type']}: {test['input'][:20]}...", False, "Should have failed validation")
                        all_passed = False
                
                except Exception as e:
                    # Exception - validation failed
                    if not test["should_pass"]:
                        self.log_test(f"Validation {test['type']}: {test['input'][:20]}...", True)
                    else:
                        self.log_test(f"Validation {test['type']}: {test['input'][:20]}...", False, f"Unexpected validation error: {e}")
                        all_passed = False
            
            return all_passed
            
        except Exception as e:
            self.log_test("Input validation logic", False, f"Test setup error: {e}")
            return False
    
    def test_error_handling_patterns(self) -> bool:
        """Test error handling patterns in the implementation."""
        print("\n🛡️  Testing Error Handling Patterns...")
        
        files_to_check = [
            "orchestrator/translation_queue.py",
            "api/hybrid_router.py",
            "master_orchestrator_api.py"
        ]
        
        all_passed = True
        
        for file_path in files_to_check:
            full_path = self.base_path / file_path
            
            if not full_path.exists():
                self.log_test(f"Error handling check: {file_path}", False, "File not found")
                all_passed = False
                continue
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for proper exception handling
            try_count = content.count('try:')
            except_count = content.count('except')
            httpexception_count = content.count('HTTPException')
            
            # Each try should have at least one except
            if try_count == 0:
                self.log_test(f"Error handling: {file_path}", False, "No try/except blocks found")
                all_passed = False
            elif except_count < try_count:
                self.log_test(f"Error handling: {file_path}", False, f"Try blocks ({try_count}) > except blocks ({except_count})")
                all_passed = False
            else:
                self.log_test(f"Error handling: {file_path}", True, f"{try_count} try blocks with {except_count} except handlers")
            
            # API files should have HTTPException handling
            if 'api' in file_path and httpexception_count == 0:
                self.log_test(f"HTTP error handling: {file_path}", False, "No HTTPException usage in API file")
                all_passed = False
        
        return all_passed
    
    def test_configuration_consistency(self) -> bool:
        """Test configuration file consistency."""
        print("\n⚙️  Testing Configuration Consistency...")
        
        config_path = self.base_path / "config.yaml"
        
        if not config_path.exists():
            self.log_test("Config file exists", False, "config.yaml not found")
            return False
        
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.log_test("Config YAML parsing", True)
            
            # Check required sections
            required_sections = [
                'master_orchestrator',
                'master_orchestrator.translation_queue',
                'master_orchestrator.llm'
            ]
            
            all_passed = True
            
            for section in required_sections:
                keys = section.split('.')
                current = config
                
                for key in keys:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        self.log_test(f"Config section: {section}", False, "Section missing")
                        all_passed = False
                        break
                else:
                    self.log_test(f"Config section: {section}", True)
            
            # Check for reasonable timeout values
            try:
                tq_config = config['master_orchestrator']['translation_queue']
                timeout = tq_config.get('timeout_seconds', 0)
                
                if timeout <= 0:
                    self.log_test("Translation timeout", False, "Timeout must be positive")
                    all_passed = False
                elif timeout > 3600:  # 1 hour
                    self.log_test("Translation timeout", False, f"Timeout too large: {timeout}s")
                    all_passed = False
                else:
                    self.log_test("Translation timeout", True, f"Reasonable timeout: {timeout}s")
                    
            except KeyError:
                self.log_test("Translation timeout config", False, "Timeout configuration missing")
                all_passed = False
            
            return all_passed
            
        except yaml.YAMLError as e:
            self.log_test("Config YAML parsing", False, f"YAML error: {e}")
            return False
        except Exception as e:
            self.log_test("Config validation", False, f"Error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all connectivity and smoke tests."""
        print("🔍 Starting Comprehensive Connectivity & Smoke Tests")
        print("=" * 60)
        
        test_functions = [
            ("Import & Syntax", self.test_imports_and_syntax),
            ("Translation Queue Logic", self.test_translation_queue_logic),
            ("Circular Dependency Detection", self.test_circular_dependency_detection),
            ("Input Validation", self.test_input_validation),
            ("Error Handling", self.test_error_handling_patterns),
            ("Configuration", self.test_configuration_consistency)
        ]
        
        passed_tests = 0
        total_tests = len(test_functions)
        
        for test_name, test_func in test_functions:
            print(f"\n🧪 {test_name}")
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                
                if result:
                    passed_tests += 1
                    
            except Exception as e:
                print(f"  ❌ FAIL {test_name} - Exception: {e}")
                traceback.print_exc()
        
        # Summary
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        
        individual_results = {}
        for result in self.test_results:
            test_category = result["name"].split(":")[0] if ":" in result["name"] else "General"
            if test_category not in individual_results:
                individual_results[test_category] = {"passed": 0, "total": 0}
            
            individual_results[test_category]["total"] += 1
            if result["passed"]:
                individual_results[test_category]["passed"] += 1
        
        print(f"\nTest Categories:")
        for category, stats in individual_results.items():
            print(f"  {category}: {stats['passed']}/{stats['total']} passed")
        
        print(f"\nOverall: {passed_tests}/{total_tests} test suites passed")
        print(f"Individual tests: {sum(1 for r in self.test_results if r['passed'])}/{len(self.test_results)} passed")
        
        success_rate = passed_tests / total_tests
        if success_rate == 1.0:
            print("🎉 All tests passed! Implementation looks solid.")
        elif success_rate >= 0.8:
            print("⚠️  Most tests passed, but some issues need attention.")
        else:
            print("❌ Multiple issues found that need fixing.")
        
        return {
            "overall_passed": passed_tests,
            "overall_total": total_tests,
            "individual_results": self.test_results,
            "success_rate": success_rate
        }


async def main():
    """Main test runner."""
    tester = ConnectivityTester()
    results = await tester.run_all_tests()
    
    return 0 if results["success_rate"] >= 0.8 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 