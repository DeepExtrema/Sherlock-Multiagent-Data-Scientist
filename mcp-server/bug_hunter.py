"""
Bug Hunter for Hybrid API Implementation

Specialized script to detect specific bugs and runtime issues:
- Race conditions in async code
- Memory leaks in queues
- Invalid state transitions
- Resource cleanup issues
- Error handling gaps
- Logic inconsistencies
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Set
import yaml

class BugHunter:
    """Specialized bug detection for the hybrid API implementation."""
    
    def __init__(self):
        self.bugs_found = []
        self.potential_issues = []
        self.base_path = Path(__file__).parent
        
    def log_bug(self, severity: str, message: str, file: str, line: int = None):
        """Log a bug with severity level."""
        location = f"{file}:{line}" if line else file
        self.bugs_found.append(f"🐛 {severity.upper()} [{location}]: {message}")
        
    def log_potential_issue(self, message: str, file: str, line: int = None):
        """Log a potential issue."""
        location = f"{file}:{line}" if line else file
        self.potential_issues.append(f"⚠️  POTENTIAL [{location}]: {message}")
    
    def check_translation_queue_bugs(self) -> List[str]:
        """Check for bugs in translation queue implementation."""
        bugs = []
        tq_path = self.base_path / "orchestrator" / "translation_queue.py"
        
        if not tq_path.exists():
            return ["Translation queue file not found"]
        
        with open(tq_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Bug 1: Check for race conditions in Redis operations
        redis_operations = re.findall(r'await.*\.(hset|rpush|blpop|expire)', content)
        if len(redis_operations) > 3:
            # Multiple Redis operations might have race conditions
            for i, line in enumerate(lines):
                if 'await' in line and any(op in line for op in ['hset', 'rpush']):
                    if i + 1 < len(lines) and 'await' in lines[i + 1]:
                        self.log_potential_issue(
                            "Potential race condition: consecutive Redis operations without transaction",
                            "translation_queue.py", i + 1
                        )
        
        # Bug 2: Check for memory leaks in in-memory queue
        if 'in_memory_tokens' in content:
            if 'del' not in content and 'pop' not in content:
                self.log_bug(
                    "HIGH", 
                    "Potential memory leak: in_memory_tokens dict never cleaned up",
                    "translation_queue.py"
                )
        
        # Bug 3: Check for proper error handling in critical paths
        critical_methods = ['enqueue', 'get_status', 'update_status', 'pop_next']
        for method in critical_methods:
            method_pattern = rf'async def {method}\('
            if re.search(method_pattern, content):
                # Find the method body
                method_start = content.find(f'async def {method}(')
                if method_start != -1:
                    # Look for try/except in this method
                    method_section = content[method_start:method_start + 1000]  # Rough method boundary
                    if 'try:' not in method_section or 'except Exception' not in method_section:
                        self.log_bug(
                            "MEDIUM",
                            f"Missing comprehensive error handling in {method}() method",
                            "translation_queue.py"
                        )
        
        # Bug 4: Check for token collision possibilities
        if 'uuid4().hex' in content:
            # UUID4 is good, but check if there's any validation
            if 'token in' not in content:  # Check for existence before use
                self.log_potential_issue(
                    "No token collision detection before storing",
                    "translation_queue.py"
                )
        
        # Bug 5: Check for proper async context management
        if 'asyncio.Queue' in content:
            if 'queue.put' in content and 'await queue.put' not in content:
                self.log_bug(
                    "HIGH",
                    "asyncio.Queue.put() calls without await - this will cause runtime errors",
                    "translation_queue.py"
                )
        
        return bugs
    
    def check_api_router_bugs(self) -> List[str]:
        """Check for bugs in API router implementation."""
        bugs = []
        router_path = self.base_path / "api" / "hybrid_router.py"
        
        if not router_path.exists():
            return ["API router file not found"]
        
        with open(router_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Bug 1: Check for SQL injection in validation (though unlikely with Pydantic)
        if 'eval(' in content or 'exec(' in content:
            self.log_bug(
                "CRITICAL",
                "Dangerous eval/exec usage found - potential code injection",
                "hybrid_router.py"
            )
        
        # Bug 2: Check for response data leakage
        internal_fields = ['error_details', 'metadata', 'correlation_id']
        for field in internal_fields:
            pattern = rf'"{field}".*:.*\{{[^}}]*\}}'
            if re.search(pattern, content):
                self.log_potential_issue(
                    f"Internal field '{field}' might expose sensitive data in API responses",
                    "hybrid_router.py"
                )
        
        # Bug 3: Check for missing rate limiting on all endpoints
        endpoint_patterns = [r'@router\.post\(["\']([^"\']+)["\']', r'@router\.get\(["\']([^"\']+)["\']']
        endpoints = []
        for pattern in endpoint_patterns:
            endpoints.extend(re.findall(pattern, content))
        
        rate_limited_count = content.count('rate_limiter.check')
        if len(endpoints) > rate_limited_count:
            self.log_bug(
                "MEDIUM",
                f"Found {len(endpoints)} endpoints but only {rate_limited_count} rate limit checks",
                "hybrid_router.py"
            )
        
        # Bug 4: Check for CORS issues or missing security headers
        if 'cors' not in content.lower() and 'origin' not in content.lower():
            self.log_potential_issue(
                "No CORS handling visible in router - may cause browser issues",
                "hybrid_router.py"
            )
        
        # Bug 5: Check for improper error response formats
        error_responses = re.findall(r'HTTPException\([^)]+\)', content)
        for i, response in enumerate(error_responses):
            if 'detail=' not in response:
                self.log_bug(
                    "LOW",
                    f"HTTPException missing detail field in error response #{i+1}",
                    "hybrid_router.py"
                )
        
        return bugs
    
    def check_master_orchestrator_bugs(self) -> List[str]:
        """Check for bugs in master orchestrator integration."""
        bugs = []
        mo_path = self.base_path / "master_orchestrator_api.py"
        
        if not mo_path.exists():
            return ["Master orchestrator file not found"]
        
        with open(mo_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Bug 1: Check for proper component lifecycle management
        lifespan_pattern = r'@asynccontextmanager\s+async def lifespan'
        if not re.search(lifespan_pattern, content):
            self.log_bug(
                "HIGH",
                "Missing asynccontextmanager for proper component lifecycle",
                "master_orchestrator_api.py"
            )
        
        # Bug 2: Check for resource cleanup in lifespan
        cleanup_keywords = ['stop', 'close', 'shutdown', 'cleanup']
        cleanup_count = sum(1 for keyword in cleanup_keywords if keyword in content)
        
        component_count = content.count('= None')  # Global component declarations
        if component_count > cleanup_count:
            self.log_potential_issue(
                "Some components may not be properly cleaned up in lifespan",
                "master_orchestrator_api.py"
            )
        
        # Bug 3: Check for circular imports
        imports = re.findall(r'from\s+(\S+)\s+import', content)
        local_imports = [imp for imp in imports if not imp.startswith(('fastapi', 'typing', 'datetime'))]
        
        for imp in local_imports:
            if 'api' in imp and 'orchestrator' in imp:
                self.log_potential_issue(
                    f"Potential circular import: {imp}",
                    "master_orchestrator_api.py"
                )
        
        # Bug 4: Check for missing error handling in lifespan
        lifespan_start = content.find('async def lifespan(')
        if lifespan_start != -1:
            lifespan_section = content[lifespan_start:lifespan_start + 3000]
            if 'except Exception' not in lifespan_section:
                self.log_bug(
                    "MEDIUM",
                    "Lifespan function lacks comprehensive error handling",
                    "master_orchestrator_api.py"
                )
        
        return bugs
    
    def check_config_consistency_bugs(self) -> List[str]:
        """Check for configuration consistency issues."""
        bugs = []
        config_path = self.base_path / "config.yaml"
        
        if not config_path.exists():
            return ["Config file not found"]
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return [f"YAML parsing error: {e}"]
        
        # Bug 1: Check for inconsistent timeout values
        timeouts = []
        
        def extract_timeouts(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if 'timeout' in key.lower():
                        timeouts.append((new_prefix, value))
                    extract_timeouts(value, new_prefix)
        
        extract_timeouts(config)
        
        # Check for logical timeout relationships
        translation_timeout = None
        task_timeout = None
        
        for path, value in timeouts:
            if 'translation' in path and 'timeout' in path:
                translation_timeout = value
            elif 'task' in path and 'timeout' in path:
                task_timeout = value
        
        if translation_timeout and task_timeout:
            if translation_timeout >= task_timeout:
                self.log_bug(
                    "MEDIUM",
                    f"Translation timeout ({translation_timeout}s) >= task timeout ({task_timeout}s) - may cause conflicts",
                    "config.yaml"
                )
        
        # Bug 2: Check for missing Redis configuration
        redis_configs = []
        
        def find_redis_configs(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if 'redis' in key.lower():
                        redis_configs.append((new_prefix, value))
                    find_redis_configs(value, new_prefix)
        
        find_redis_configs(config)
        
        unique_redis_urls = set()
        for path, value in redis_configs:
            if isinstance(value, str) and value.startswith('redis://'):
                unique_redis_urls.add(value)
        
        if len(unique_redis_urls) > 1:
            self.log_potential_issue(
                f"Multiple Redis URLs configured: {unique_redis_urls} - ensure they're intentional",
                "config.yaml"
            )
        
        # Bug 3: Check for development vs production settings
        if 'localhost' in str(config):
            self.log_potential_issue(
                "Configuration contains localhost URLs - not suitable for production",
                "config.yaml"
            )
        
        return bugs
    
    def check_test_coverage_gaps(self) -> List[str]:
        """Check for gaps in test coverage."""
        gaps = []
        test_path = self.base_path / "test_hybrid_api.py"
        
        if not test_path.exists():
            return ["Test file not found"]
        
        with open(test_path, 'r', encoding='utf-8') as f:
            test_content = f.read()
        
        # Check for missing test scenarios
        critical_scenarios = {
            'Redis connection failure': 'redis.*fail',
            'Translation timeout': 'timeout',
            'Invalid YAML in DSL': 'invalid.*yaml',
            'Circular dependencies': 'circular',
            'Rate limiting': 'rate.*limit',
            'Token expiration': 'token.*expir',
            'Concurrent requests': 'concurrent',
            'Large payloads': 'large|big|size',
            'Network failures': 'network.*fail',
            'Database connection loss': 'database.*fail|mongo.*fail'
        }
        
        for scenario, pattern in critical_scenarios.items():
            if not re.search(pattern, test_content, re.IGNORECASE):
                self.log_potential_issue(
                    f"Missing test coverage for: {scenario}",
                    "test_hybrid_api.py"
                )
        
        return gaps
    
    def check_performance_issues(self) -> List[str]:
        """Check for potential performance issues."""
        issues = []
        
        # Check all Python files for performance anti-patterns
        python_files = [
            "orchestrator/translation_queue.py",
            "api/hybrid_router.py",
            "master_orchestrator_api.py"
        ]
        
        for file_path in python_files:
            full_path = self.base_path / file_path
            if not full_path.exists():
                continue
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Performance Issue 1: Synchronous operations in async functions
            if 'time.sleep(' in content:
                self.log_bug(
                    "HIGH",
                    "time.sleep() found in async context - use asyncio.sleep() instead",
                    file_path
                )
            
            # Performance Issue 2: Blocking operations without timeout
            blocking_patterns = [
                r'\.get\(\s*[^,)]*\s*\)',  # dict.get() without timeout
                r'\.blpop\([^,)]*\)',      # Redis blpop without timeout
            ]
            
            for pattern in blocking_patterns:
                matches = re.findall(pattern, content)
                if matches and 'timeout' not in ''.join(matches):
                    self.log_potential_issue(
                        f"Blocking operation without timeout: {matches[0]}",
                        file_path
                    )
            
            # Performance Issue 3: Memory inefficient operations
            if 'scan_iter' in content and 'async for' not in content:
                self.log_potential_issue(
                    "Redis scan_iter without async iteration - may block event loop",
                    file_path
                )
            
            # Performance Issue 4: No connection pooling mentioned
            if 'redis.from_url' in content and 'pool' not in content:
                self.log_potential_issue(
                    "Redis connection without explicit pool configuration",
                    file_path
                )
        
        return issues
    
    def run_comprehensive_bug_hunt(self) -> Dict[str, Any]:
        """Run all bug detection checks."""
        print("🔍 Starting Comprehensive Bug Hunt")
        print("=" * 50)
        
        # Run all bug checks
        print("\n🐛 Checking Translation Queue...")
        self.check_translation_queue_bugs()
        
        print("🐛 Checking API Router...")
        self.check_api_router_bugs()
        
        print("🐛 Checking Master Orchestrator...")
        self.check_master_orchestrator_bugs()
        
        print("🐛 Checking Configuration...")
        self.check_config_consistency_bugs()
        
        print("🐛 Checking Test Coverage...")
        self.check_test_coverage_gaps()
        
        print("🐛 Checking Performance Issues...")
        self.check_performance_issues()
        
        # Summary
        print("\n" + "=" * 50)
        print("🏁 BUG HUNT RESULTS")
        print("=" * 50)
        
        if self.bugs_found:
            print(f"\n🐛 BUGS FOUND ({len(self.bugs_found)}):")
            for bug in self.bugs_found:
                print(f"  {bug}")
        
        if self.potential_issues:
            print(f"\n⚠️  POTENTIAL ISSUES ({len(self.potential_issues)}):")
            for issue in self.potential_issues:
                print(f"  {issue}")
        
        if not self.bugs_found and not self.potential_issues:
            print("\n🎉 No bugs or issues found! Code looks clean.")
        
        return {
            'bugs_found': len(self.bugs_found),
            'potential_issues': len(self.potential_issues),
            'bug_details': self.bugs_found,
            'issue_details': self.potential_issues,
            'status': 'clean' if not self.bugs_found else 'issues_found'
        }


def main():
    """Main bug hunting function."""
    hunter = BugHunter()
    results = hunter.run_comprehensive_bug_hunt()
    
    # Return exit code based on findings
    if results['bugs_found'] == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 