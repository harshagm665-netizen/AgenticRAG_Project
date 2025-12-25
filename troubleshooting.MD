# 🔧 Quick Troubleshooting Guide

## Common Issues & Solutions

### 1. **"ModuleNotFoundError" errors**
```powershell
# Solution: Install missing packages
pip install -r requirements.txt

# If still failing, try upgrading:
pip install --upgrade langchain langchain-community langchain-openai
```

---

### 2. **"The api_key client option must be set"**
```powershell
# Solution A: Check your .env file exists and has:
OPENAI_API_KEY=your_actual_key_here

# Solution B: Set in PowerShell:
$env:OPENAI_API_KEY = "your_actual_key_here"
streamlit run app.py
```

---

### 3. **"Error during retrieval: 'VectorStoreRetriever' object has no attribute..."**
**Status:** ✅ FIXED in latest rag_tools.py
- Uses `retriever.invoke()` instead of deprecated methods
- Make sure you have the latest version of files

---

### 4. **No documents showing after upload**
**Check:**
- File formats are PDF, TXT, or MD
- Files aren't corrupted
- Look for error messages in red
- Check temp_data folder was created

**Solution:**
```powershell
# If temp_data folder has permission issues:
rm -r temp_data
mkdir temp_data
```

---

### 5. **Vector store build fails**
**Common causes:**
- API key not set or invalid
- No internet connection
- Vocareum endpoint down

**Test your connection:**
```python
# Run this in Python to test:
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openai.vocareum.com/v1"
)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="test"
)
print("Success!" if response else "Failed")
```

---

### 6. **Guardrails not triggering**
**Check:**
- `safety.py` exists in project folder
- Query contains exact trigger words: "confidential", "secret", etc.
- Case-insensitive matching is working

**Test safety module:**
```python
from safety import check_safety
print(check_safety("This is confidential"))  # Should return (False, 'confidential')
print(check_safety("This is safe"))  # Should return (True, None)
```

---

### 7. **Filter tool not working**
**Common issues:**
- Filename doesn't match exactly (case-sensitive)
- File wasn't uploaded
- Metadata wasn't set correctly

**Debug:**
```python
# Check metadata in Python console:
docs = st.session_state.docs
for doc in docs[:5]:
    print(doc.metadata)
```

---

### 8. **Slow response times**
**Causes:**
- Large documents (many chunks to process)
- Slow API endpoint
- High k value (retrieving too many chunks)

**Solutions:**
- Reduce chunk_size (less chunks overall)
- Reduce k value in retriever (currently set to 3)
- Use faster embedding model

---

### 9. **Streamlit keeps restarting**
**Cause:** File watcher detecting changes

**Solution:**
```powershell
# Run with --server.fileWatcherType none
streamlit run app.py --server.fileWatcherType none
```

---

### 10. **Port 8501 already in use**
```powershell
# Find and kill process using port 8501:
netstat -ano | findstr :8501
taskkill /PID <process_id> /F

# Or use different port:
streamlit run app.py --server.port 8502
```

---

## Verification Checklist

Before starting demo, verify:

```powershell
# 1. Check Python version (should be 3.8+)
python --version

# 2. Check virtual environment is active
# Should see (.venv) in prompt

# 3. Check all files exist
ls *.py
# Should see: app.py, rag_tools.py, safety.py

# 4. Check .env file
cat .env
# Should show: OPENAI_API_KEY=...

# 5. Test import
python -c "from rag_tools import rag_retriever_tool; print('OK')"

# 6. Test Streamlit
streamlit --version
```

---

## Emergency Reset

If everything breaks:

```powershell
# 1. Stop Streamlit
Ctrl+C

# 2. Clear cache
rm -r .streamlit
rm -r temp_data

# 3. Reinstall packages
pip uninstall -y langchain langchain-community langchain-openai
pip install -r requirements.txt

# 4. Restart
streamlit run app.py
```

---

## Getting Help

If issues persist:

1. **Check error message carefully** - It usually tells you what's wrong
2. **Google the exact error** - LangChain errors are well-documented
3. **Check LangChain docs**: https://python.langchain.com/docs/
4. **Verify API access** - Test with simple OpenAI call
5. **Check file versions** - Make sure you have latest rag_tools.py and app.py

---

## Contact Information

For demo-specific questions:
- Review DEMO_SCRIPT.md
- Check QUICKSTART.md for setup
- Review SETUP.md for configuration

Good luck! 🚀