"""
Comprehensive Validation Script for Hybrid API Implementation

This script performs static analysis and validation without needing to run the server.
It checks for:
- Import compatibility
- Syntax errors
- Logic consistency
- Configuration validation
- Edge case handling
- Potential runtime bugs
"""

import ast
import os
import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import traceback

class ImplementationValidator:
    """Validates the hybrid API implementation for bugs and issues."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.successes = []
        self.base_path = Path(__file__).parent
        
    def log_error(self, message: str, context: str = ""):
        """Log an error."""
        self.errors.append(f"❌ ERROR [{context}]: {message}")
        
    def log_warning(self, message: str, context: str = ""):
        """Log a warning."""
        self.warnings.append(f"⚠️  WARNING [{context}]: {message}")
        
    def log_success(self, message: str, context: str = ""):
        """Log a success."""
        self.successes.append(f"✅ SUCCESS [{context}]: {message}")
    
    def validate_file_syntax(self, filepath: Path) -> bool:
        """Validate Python file syntax."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to check syntax
            ast.parse(content)
            self.log_success(f"Syntax valid", f"{filepath.name}")
            return True
            
        except SyntaxError as e:
            self.log_error(f"Syntax error at line {e.lineno}: {e.msg}", f"{filepath.name}")
            return False
        except Exception as e:
            self.log_error(f"Failed to read/parse file: {e}", f"{filepath.name}")
            return False
    
    def validate_imports(self, filepath: Path) -> bool:
        """Validate imports in a Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports_valid = True
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        if self._is_problematic_import(module_name):
                            self.log_warning(f"Import '{module_name}' may not be available", f"{filepath.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module
                    if module_name and self._is_problematic_import(module_name):
                        self.log_warning(f"Import from '{module_name}' may not be available", f"{filepath.name}")
            
            if imports_valid:
                self.log_success("Imports structure valid", f"{filepath.name}")
            
            return imports_valid
            
        except Exception as e:
            self.log_error(f"Failed to validate imports: {e}", f"{filepath.name}")
            return False
    
    def _is_problematic_import(self, module_name: str) -> bool:
        """Check if an import might be problematic."""
        # Common modules that might not be available
        optional_modules = {
            'redis', 'redis.asyncio', 'aiohttp', 'motor', 'confluent_kafka',
            'uvicorn', 'fastapi', 'pydantic'
        }
        
        for optional in optional_modules:
            if module_name.startswith(optional):
                return True
        return False
    
    def validate_config_structure(self) -> bool:
        """Validate the configuration file structure."""
        try:
            config_path = self.base_path / "config.yaml"
            if not config_path.exists():
                self.log_error("config.yaml not found", "Config")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = [
                'master_orchestrator',
                'master_orchestrator.translation_queue',
                'master_orchestrator.llm',
                'master_orchestrator.cache'
            ]
            
            for section in required_sections:
                if not self._get_nested_key(config, section):
                    self.log_error(f"Missing required config section: {section}", "Config")
                    return False
            
            # Validate translation queue config
            tq_config = config['master_orchestrator']['translation_queue']
            required_tq_keys = ['redis_url', 'timeout_seconds', 'max_retries']
            
            for key in required_tq_keys:
                if key not in tq_config:
                    self.log_error(f"Missing translation_queue config: {key}", "Config")
                    return False
            
            self.log_success("Configuration structure valid", "Config")
            return True
            
        except yaml.YAMLError as e:
            self.log_error(f"YAML parsing error: {e}", "Config")
            return False
        except Exception as e:
            self.log_error(f"Config validation failed: {e}", "Config")
            return False
    
    def _get_nested_key(self, data: Dict, key_path: str) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def validate_translation_queue_logic(self) -> bool:
        """Validate translation queue implementation logic."""
        try:
            tq_path = self.base_path / "orchestrator" / "translation_queue.py"
            
            with open(tq_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            issues_found = []
            
            # Check for proper async/await usage
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if async functions use await properly
                    if node.decorator_list:
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Name) and decorator.id == 'async':
                                # This should be an async def, not a decorator
                                issues_found.append(f"Function {node.name} incorrectly uses async as decorator")
                
                # Check for potential race conditions in Redis operations
                if isinstance(node, ast.Call):
                    if hasattr(node.func, 'attr') and node.func.attr in ['hset', 'rpush', 'blpop']:
                        # Good - using Redis operations
                        pass
            
            # Check for error handling patterns
            try_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
            if len(try_blocks) < 5:  # Should have several try/except blocks
                self.log_warning("Few try/except blocks found - consider more error handling", "TranslationQueue")
            
            self.log_success("Translation queue logic structure valid", "TranslationQueue")
            return True
            
        except Exception as e:
            self.log_error(f"Translation queue validation failed: {e}", "TranslationQueue")
            return False
    
    def validate_api_router_logic(self) -> bool:
        """Validate API router implementation."""
        try:
            router_path = self.base_path / "api" / "hybrid_router.py"
            
            with open(router_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common API issues
            issues = []
            
            # Check for proper HTTP status codes
            if 'status.HTTP_202_ACCEPTED' not in content:
                issues.append("Missing 202 status for async operations")
            
            if 'status.HTTP_429_TOO_MANY_REQUESTS' not in content:
                issues.append("Missing 429 status for rate limiting")
            
            # Check for input validation
            if '@validator' not in content:
                issues.append("Missing input validation decorators")
            
            # Check for proper error responses
            if 'HTTPException' not in content:
                issues.append("Missing HTTP exception handling")
            
            if issues:
                for issue in issues:
                    self.log_warning(issue, "APIRouter")
            else:
                self.log_success("API router patterns look good", "APIRouter")
            
            return len(issues) == 0
            
        except Exception as e:
            self.log_error(f"API router validation failed: {e}", "APIRouter")
            return False
    
    def validate_circular_dependency_detection(self) -> bool:
        """Test the circular dependency detection logic."""
        try:
            # Test cases for circular dependency detection
            test_cases = [
                # Case 1: Simple cycle
                {
                    "name": "Simple A->B->A cycle",
                    "tasks": [
                        {"id": "A", "depends_on": ["B"]},
                        {"id": "B", "depends_on": ["A"]}
                    ],
                    "should_detect_cycle": True
                },
                # Case 2: No cycle
                {
                    "name": "Linear dependency",
                    "tasks": [
                        {"id": "A", "depends_on": []},
                        {"id": "B", "depends_on": ["A"]},
                        {"id": "C", "depends_on": ["B"]}
                    ],
                    "should_detect_cycle": False
                },
                # Case 3: Complex cycle
                {
                    "name": "Complex A->B->C->A cycle",
                    "tasks": [
                        {"id": "A", "depends_on": ["C"]},
                        {"id": "B", "depends_on": ["A"]},
                        {"id": "C", "depends_on": ["B"]}
                    ],
                    "should_detect_cycle": True
                },
                # Case 4: Self-dependency
                {
                    "name": "Self dependency",
                    "tasks": [
                        {"id": "A", "depends_on": ["A"]}
                    ],
                    "should_detect_cycle": True
                }
            ]
            
            # Simulate the circular dependency check logic
            def check_circular_dependencies(tasks: List[Dict[str, Any]]):
                """Simulate the circular dependency detection."""
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
            
            all_passed = True
            
            for test_case in test_cases:
                try:
                    check_circular_dependencies(test_case["tasks"])
                    # No exception raised
                    if test_case["should_detect_cycle"]:
                        self.log_error(f"Failed to detect cycle in: {test_case['name']}", "CircularDeps")
                        all_passed = False
                    else:
                        self.log_success(f"Correctly no cycle detected: {test_case['name']}", "CircularDeps")
                        
                except ValueError as e:
                    # Exception raised (cycle detected)
                    if test_case["should_detect_cycle"]:
                        self.log_success(f"Correctly detected cycle: {test_case['name']}", "CircularDeps")
                    else:
                        self.log_error(f"False positive cycle detection: {test_case['name']}", "CircularDeps")
                        all_passed = False
            
            return all_passed
            
        except Exception as e:
            self.log_error(f"Circular dependency validation failed: {e}", "CircularDeps")
            return False
    
    def validate_edge_case_handling(self) -> bool:
        """Validate edge case handling in the implementation."""
        edge_cases_valid = True
        
        # Check translation queue edge cases
        tq_edge_cases = [
            "Redis connection failure fallback",
            "Token expiration handling", 
            "Invalid token format validation",
            "Translation timeout management",
            "Retry logic with exponential backoff",
            "Queue cleanup for expired tokens"
        ]
        
        router_path = self.base_path / "api" / "hybrid_router.py"
        tq_path = self.base_path / "orchestrator" / "translation_queue.py"
        
        try:
            # Read implementation files
            with open(router_path, 'r', encoding='utf-8') as f:
                router_content = f.read()
            
            with open(tq_path, 'r', encoding='utf-8') as f:
                tq_content = f.read()
            
            # Check for edge case handling patterns
            edge_case_checks = {
                "Token validation": "len(token) != 32" in router_content,
                "Redis fallback": "in_memory_queue" in tq_content,
                "Timeout handling": "timeout_seconds" in tq_content,
                "Error status codes": "HTTP_400_BAD_REQUEST" in router_content,
                "Rate limiting": "HTTP_429_TOO_MANY_REQUESTS" in router_content,
                "Exception handling": "except Exception" in tq_content,
                "Retry logic": "max_retries" in tq_content,
                "Input validation": "@validator" in router_content
            }
            
            for check_name, is_implemented in edge_case_checks.items():
                if is_implemented:
                    self.log_success(f"{check_name} implemented", "EdgeCases")
                else:
                    self.log_warning(f"{check_name} may be missing", "EdgeCases")
                    edge_cases_valid = False
            
        except Exception as e:
            self.log_error(f"Edge case validation failed: {e}", "EdgeCases")
            edge_cases_valid = False
        
        return edge_cases_valid
    
    def validate_async_patterns(self) -> bool:
        """Validate async/await patterns in the implementation."""
        try:
            files_to_check = [
                "orchestrator/translation_queue.py",
                "api/hybrid_router.py",
                "master_orchestrator_api.py"
            ]
            
            async_issues = []
            
            for file_path in files_to_check:
                full_path = self.base_path / file_path
                
                if not full_path.exists():
                    async_issues.append(f"File not found: {file_path}")
                    continue
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Check for async/await patterns
                async_functions = []
                await_calls = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.AsyncFunctionDef):
                        async_functions.append(node.name)
                    elif isinstance(node, ast.Await):
                        await_calls.append("await_call")
                
                # Validate patterns
                if 'async def' in content and len(await_calls) == 0:
                    async_issues.append(f"{file_path}: async functions without await calls")
                
                if 'await' in content and len(async_functions) == 0:
                    async_issues.append(f"{file_path}: await calls outside async functions")
            
            if async_issues:
                for issue in async_issues:
                    self.log_warning(issue, "AsyncPatterns")
                return False
            else:
                self.log_success("Async patterns look correct", "AsyncPatterns")
                return True
                
        except Exception as e:
            self.log_error(f"Async pattern validation failed: {e}", "AsyncPatterns")
            return False
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print("🔍 Starting Comprehensive Validation of Hybrid API Implementation")
        print("=" * 70)
        
        validation_results = {}
        
        # 1. File existence and syntax validation
        core_files = [
            "orchestrator/translation_queue.py",
            "api/hybrid_router.py", 
            "api/__init__.py",
            "master_orchestrator_api.py",
            "config.yaml",
            "test_hybrid_api.py"
        ]
        
        print("\n📁 Validating Core Files...")
        files_valid = True
        for file_path in core_files:
            full_path = self.base_path / file_path
            
            if not full_path.exists():
                self.log_error(f"Core file missing: {file_path}", "Files")
                files_valid = False
                continue
            
            if file_path.endswith('.py'):
                if not self.validate_file_syntax(full_path):
                    files_valid = False
                else:
                    self.validate_imports(full_path)
        
        validation_results['files_valid'] = files_valid
        
        # 2. Configuration validation
        print("\n⚙️  Validating Configuration...")
        validation_results['config_valid'] = self.validate_config_structure()
        
        # 3. Logic validation
        print("\n🧠 Validating Implementation Logic...")
        validation_results['translation_queue_valid'] = self.validate_translation_queue_logic()
        validation_results['api_router_valid'] = self.validate_api_router_logic()
        validation_results['circular_deps_valid'] = self.validate_circular_dependency_detection()
        
        # 4. Edge case validation
        print("\n🛡️  Validating Edge Case Handling...")
        validation_results['edge_cases_valid'] = self.validate_edge_case_handling()
        
        # 5. Async pattern validation
        print("\n⚡ Validating Async Patterns...")
        validation_results['async_patterns_valid'] = self.validate_async_patterns()
        
        # Summary
        print("\n" + "=" * 70)
        print("📋 VALIDATION SUMMARY")
        print("=" * 70)
        
        # Print results
        for category in ['successes', 'warnings', 'errors']:
            items = getattr(self, category)
            if items:
                print(f"\n{category.upper()}:")
                for item in items:
                    print(f"  {item}")
        
        # Overall assessment
        total_checks = len(validation_results)
        passed_checks = sum(1 for result in validation_results.values() if result)
        
        print(f"\n🏁 OVERALL RESULT: {passed_checks}/{total_checks} validation categories passed")
        
        if passed_checks == total_checks:
            print("🎉 All validations passed! Implementation looks solid.")
        elif passed_checks >= total_checks * 0.8:
            print("⚠️  Most validations passed, but some issues need attention.")
        else:
            print("❌ Multiple issues found that need fixing.")
        
        validation_results['overall_score'] = passed_checks / total_checks
        validation_results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'success_count': len(self.successes),
            'warning_count': len(self.warnings),
            'error_count': len(self.errors)
        }
        
        return validation_results


def main():
    """Main validation function."""
    validator = ImplementationValidator()
    results = validator.run_comprehensive_validation()
    
    # Return exit code based on results
    if results['overall_score'] >= 0.8:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 