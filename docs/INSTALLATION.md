# 🛠️ **Installation Guide**

## **Windows Installation (Recommended)**

### **Prerequisites**
- Windows 10 or Windows 11
- Python 3.12 or higher
- Claude Desktop app installed
- PowerShell or Command Prompt

### **Step 1: Python Installation**
```powershell
# Check if Python is installed
python --version

# If not installed, download from https://python.org
# Make sure to check "Add Python to PATH" during installation
```

### **Step 2: Project Setup**
```powershell
# Create workspace directory
mkdir C:\deepline-workspace
cd C:\deepline-workspace

# Clone the repository
git clone https://github.com/your-org/deepline.git
cd deepline\mcp-server
```

### **Step 3: Dependency Installation**
```powershell
# Install all required packages
pip install -r requirements-python313.txt

# Verify installation
python -c "import pandas, numpy, evidently, missingno; print('✅ All dependencies installed')"
```

### **Step 4: Configuration**
```powershell
# Run setup verification
python verify_setup.py

# This will:
# ✅ Check Python version
# ✅ Verify all dependencies
# ✅ Configure Claude Desktop
# ✅ Test basic functionality
```

### **Step 5: Launch Server**
```powershell
# Start the MCP server
python launch_server.py

# You should see:
# 🚀 Deepline MCP Server starting...
# 📊 17 tools loaded successfully
# 🔗 Ready for Claude Desktop connection
```

### **Step 6: Connect to Claude Desktop**
1. **Open Claude Desktop** (restart if it was already open)
2. **Look for "deepline-eda" in the MCP servers list**
3. **Test with**: `"Load the iris.csv dataset and show me basic info"`

---

## **Alternative Installation Methods**

### **Virtual Environment (Isolated)**
```powershell
# Create virtual environment
python -m venv deepline-env
deepline-env\Scripts\activate

# Install dependencies
pip install -r requirements-python313.txt

# Remember to activate environment each time:
# deepline-env\Scripts\activate
```

### **Development Installation**
```powershell
# Install with development tools
pip install -r requirements-python313.txt
pip install pytest black ruff mypy

# Run tests to verify
python -m pytest test_*.py -v
```

### **System-Wide Installation**
```powershell
# Install globally (not recommended for production)
pip install --user -r requirements-python313.txt
```

---

## **Dependency Details**

### **Core Dependencies**
```
mcp[cli]==1.10.1              # MCP framework
pandas==2.3.0                 # Data manipulation
numpy==2.1.3                  # Numerical computing
evidently==0.7.9               # Data quality monitoring
scikit-learn==1.7.0            # Machine learning
matplotlib==3.10.0             # Visualization
```

### **Optional Dependencies**
```
pytest==7.4.3                 # Testing framework
black==*                      # Code formatting
ruff==*                       # Code linting
mypy==*                       # Type checking
```

---

## **Verification Steps**

### **Test 1: Python Environment**
```powershell
python --version
# Expected: Python 3.12.0 or higher
```

### **Test 2: Dependencies**
```powershell
python -c "import pandas, numpy, evidently, missingno, sklearn; print('✅ Core dependencies OK')"
```

### **Test 3: Server Launch**
```powershell
python launch_server.py
# Should start without errors
```

### **Test 4: Tool Loading**
```powershell
python -c "from server import mcp; print(f'✅ {len(mcp.tools)} tools loaded')"
```

### **Test 5: Claude Desktop Integration**
```powershell
python verify_setup.py
# Should show ✅ for all checks
```

---

## **Troubleshooting**

### **Common Issues**

#### **"Python not found"**
```powershell
# Solution: Install Python from https://python.org
# Make sure "Add Python to PATH" is checked
```

#### **"pip not found"**
```powershell
# Solution: Reinstall Python with pip included
# Or install pip separately:
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

#### **"Module not found" errors**
```powershell
# Solution: Install missing dependencies
pip install -r requirements-python313.txt

# Or install individually:
pip install pandas numpy evidently missingno scikit-learn matplotlib
```

#### **"Permission denied" errors**
```powershell
# Solution 1: Run as administrator
# Right-click PowerShell → "Run as administrator"

# Solution 2: Use user installation
pip install --user -r requirements-python313.txt
```

#### **"Claude Desktop not connecting"**
```powershell
# Solution: Check configuration
python verify_setup.py

# Manually check config location:
# %APPDATA%\Claude\claude_desktop_config.json
```

### **Debug Mode**
```powershell
# Enable debug logging
$env:DEBUG = "true"
python launch_server.py
```

---

## **Next Steps**

After successful installation:

1. **📖 Read the [User Guide](USER_GUIDE.md)**
2. **🧪 Try the [Examples](EXAMPLES.md)**
3. **🔧 Customize [Configuration](CONFIGURATION.md)**
4. **🚀 Start analyzing your data!**

---

## **Need Help?**

- **Check logs**: `reports/` directory
- **Run diagnostics**: `python verify_setup.py`
- **GitHub Issues**: [Report problems](https://github.com/your-org/deepline/issues)
- **Documentation**: [Full docs](../README.md) 