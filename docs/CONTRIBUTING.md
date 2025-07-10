# 🤝 **Contributing Guide**

Welcome to the Deepline community! We're excited to have you contribute to the future of data science and MLOps automation.

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.12 or higher
- Git for version control
- Basic understanding of data science workflows
- Familiarity with MCP (Model Context Protocol)

### **Development Setup**
```powershell
# 1. Fork the repository on GitHub
# 2. Clone your fork locally
git clone https://github.com/your-username/deepline.git
cd deepline/mcp-server

# 3. Create development environment
python -m venv deepline-dev
deepline-dev\Scripts\activate

# 4. Install development dependencies
pip install -r requirements-python313.txt
pip install pytest black ruff mypy

# 5. Run tests to verify setup
python -m pytest test_*.py -v
```

---

## 📋 **Development Workflow**

### **1. Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

### **2. Make Changes**
- Follow the code style guidelines
- Write comprehensive tests
- Update documentation as needed
- Ensure all tests pass

### **3. Code Quality Checks**
```powershell
# Format code
black .

# Lint code
ruff check --fix .

# Type checking
mypy server.py

# Run tests
python -m pytest test_*.py -v
```

### **4. Submit Pull Request**
- Write clear commit messages
- Reference related issues
- Describe changes in PR description
- Ensure CI checks pass

---

## 🎯 **Contribution Areas**

### **🔧 Core Server Development**
- **Tools enhancement**: Improve existing analysis tools
- **New tools**: Add advanced analytics capabilities
- **Performance optimization**: Speed up data processing
- **Error handling**: Improve robustness and error messages

### **📊 Data Analysis Features**
- **Visualization**: New plot types and interactive charts
- **Statistical methods**: Advanced statistical tests
- **ML algorithms**: New outlier detection methods
- **Quality metrics**: Enhanced data quality assessments

### **🛡️ Quality & Testing**
- **Unit tests**: Comprehensive test coverage
- **Integration tests**: End-to-end workflow testing
- **Performance tests**: Benchmarking and optimization
- **Edge case testing**: Robust error handling

### **📖 Documentation**
- **User guides**: Step-by-step tutorials
- **API documentation**: Detailed tool references
- **Examples**: Real-world use cases
- **Video tutorials**: Visual learning resources

---

## 📝 **Code Style Guidelines**

### **Python Style**
```python
# Use Black formatting (automatic)
# Follow PEP 8 conventions
# Use type hints

from typing import Dict, List, Optional, Any
import asyncio

async def analyze_data(
    dataset_name: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze dataset with specified options.
    
    Args:
        dataset_name: Name of the loaded dataset
        options: Optional configuration parameters
        
    Returns:
        Analysis results dictionary
        
    Raises:
        KeyError: If dataset not found
        ValueError: If invalid options provided
    """
    # Implementation here
    pass
```

### **Documentation Style**
- Use clear, concise language
- Include code examples
- Provide expected outputs
- Add troubleshooting tips

### **Commit Message Format**
```
feat: add new outlier detection method using LOF

- Implement Local Outlier Factor algorithm
- Add contamination parameter support
- Include visualization for detected outliers
- Update documentation with examples

Closes #123
```

---

## 🧪 **Testing Guidelines**

### **Test Structure**
```python
import pytest
import asyncio
from server import load_data, detect_outliers

class TestOutlierDetection:
    """Test suite for outlier detection functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        return "test_data.csv"
    
    @pytest.mark.asyncio
    async def test_iqr_outlier_detection(self, sample_data):
        """Test IQR outlier detection method."""
        # Load test data
        await load_data(sample_data, "test_dataset")
        
        # Test outlier detection
        result = await detect_outliers("test_dataset", method="iqr")
        
        # Assertions
        assert "outliers" in result
        assert "counts" in result
        assert result["method_used"] == "iqr"
    
    @pytest.mark.asyncio
    async def test_invalid_method(self, sample_data):
        """Test error handling for invalid method."""
        await load_data(sample_data, "test_dataset")
        
        with pytest.raises(ValueError, match="Invalid method"):
            await detect_outliers("test_dataset", method="invalid")
```

### **Test Categories**
- **Unit tests**: Individual function testing
- **Integration tests**: Multi-tool workflows
- **Performance tests**: Speed and memory usage
- **Edge case tests**: Error conditions and boundary values

---

## 🔄 **Pull Request Process**

### **1. Pre-submission Checklist**
- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Error handling implemented
- [ ] Performance considerations addressed

### **2. PR Description Template**
```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] README updated
- [ ] API documentation updated
- [ ] Examples added/updated

## Breaking Changes
None / List any breaking changes
```

### **3. Review Process**
- **Code review**: Technical accuracy and style
- **Testing review**: Coverage and quality
- **Documentation review**: Clarity and completeness
- **Performance review**: Speed and memory impact

---

## 🐛 **Bug Reports**

### **Bug Report Template**
```markdown
## Bug Description
Clear description of the issue

## Steps to Reproduce
1. Load dataset X
2. Run command Y
3. Error occurs

## Expected Behavior
What should have happened

## Actual Behavior
What actually happened

## Environment
- OS: Windows 10/11
- Python: 3.12.x
- Deepline version: x.x.x
- Dependencies: (pip freeze output)

## Additional Context
Screenshots, logs, or other relevant information
```

### **Bug Priority Levels**
- **Critical**: System crashes, data corruption
- **High**: Major functionality broken
- **Medium**: Minor functionality issues
- **Low**: Cosmetic issues, enhancements

---

## 💡 **Feature Requests**

### **Feature Request Template**
```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Other approaches evaluated

## Additional Context
Examples, mockups, or related issues
```

### **Feature Evaluation Criteria**
- **User impact**: How many users benefit?
- **Technical complexity**: Implementation difficulty
- **Performance impact**: System resource usage
- **Maintenance burden**: Long-term support needs

---

## 📚 **Development Resources**

### **Key Technologies**
- **MCP Framework**: [Documentation](https://modelcontextprotocol.io)
- **FastAPI**: [Documentation](https://fastapi.tiangolo.com)
- **Pandas**: [Documentation](https://pandas.pydata.org)
- **Evidently**: [Documentation](https://docs.evidentlyai.com)

### **Development Tools**
- **IDE**: Visual Studio Code with Python extension
- **Testing**: pytest for test framework
- **Linting**: ruff for code quality
- **Formatting**: black for code style
- **Type checking**: mypy for static analysis

### **Useful Commands**
```powershell
# Run specific test
python -m pytest test_outlier_detection.py -v

# Run with coverage
python -m pytest --cov=server test_*.py

# Profile performance
python -m cProfile -o profile.stats server.py

# Check memory usage
python -m memory_profiler server.py
```

---

## 🎖️ **Recognition**

### **Contributor Levels**
- **First-time contributor**: Welcome package and mentorship
- **Regular contributor**: Recognition in releases
- **Core contributor**: Commit access and decision-making
- **Maintainer**: Full project responsibility

### **Contribution Recognition**
- **Contributors file**: All contributors listed
- **Release notes**: Major contributions highlighted
- **Community showcase**: Featured contributions
- **Conference talks**: Speaking opportunities

---

## 📞 **Getting Help**

### **Development Questions**
- **Discord**: #development channel
- **GitHub Discussions**: Technical questions
- **Email**: dev@deepline.ai
- **Office Hours**: Weekly video calls

### **Mentorship Program**
- **New contributor onboarding**
- **Paired programming sessions**
- **Code review guidance**
- **Architecture discussions**

---

## 📄 **License & CLA**

### **Contributor License Agreement**
- **Required for all contributions**
- **Covers future license changes**
- **Protects contributors and project**
- **Simple online signing process**

### **License Structure**
- **SDK/Client components**: Apache 2.0
- **Core server**: BUSL 1.1 (converts to Apache 2.0 in 2027)
- **Documentation**: Creative Commons

---

**🎉 Thank you for contributing to Deepline! Together, we're building the future of data science automation.**

*Have questions? Don't hesitate to reach out via GitHub Discussions or our Discord community.* 