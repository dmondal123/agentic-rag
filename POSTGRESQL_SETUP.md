# PostgreSQL Required Setup Guide

## ✅ **Changes Made**

The ExecutionAgent now **requires a PostgreSQL connection** - no more fallback to mock documents.

### **Removed:**
- ❌ Mock document store initialization
- ❌ Fallback retrieval methods  
- ❌ `DocumentChunk` class (unused)
- ❌ All fallback error handling

### **Now Required:**
- ✅ PostgreSQL database connection
- ✅ Proper `.env` configuration
- ✅ Database schema setup

## 🔧 **Setup Your Environment**

### **1. Create `.env` file:**
```bash
# PostgreSQL Configuration (adjust for your setup)
PGHOST=localhost
PGPORT=5433
PGUSER=postgres
PGPASSWORD=password  
PGDATABASE=pdfrag

# Required: OpenAI API Key
OPENAI_API_KEY=your_actual_openai_key_here

# Retrieval Settings
USE_HYDE_RETRIEVAL=true
ENABLE_SPARSE_CONTEXT=true
MODEL_NAME=gpt-4o
```

### **2. Ensure Your PostgreSQL Container is Running:**
```bash
# Check container status
docker ps | grep pgvector

# Should show: pgvector-container running with port 5432->5433
```

### **3. Verify Database Schema:**
Your PostgreSQL database needs the `documents` table:

```sql
-- Connect to your database
docker exec -it pgvector-container psql -U postgres -d pdfrag

-- Check if table exists
\dt

-- If not, create it:
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    source TEXT,
    content_type TEXT,
    content TEXT,
    embedding VECTOR(1536),
    metadata JSONB
);

-- Create index for vector search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

## 🚨 **Error Handling**

The system will now **fail fast** if PostgreSQL is unavailable:

```python
RuntimeError: ExecutionAgent requires PostgreSQL connection. 
Failed to initialize PostgreSQL retriever: connection to server at "localhost", port 5433 failed
```

This is **intentional** - you'll know immediately if your database is down.

## 🧪 **Test Your Setup**

```bash
# Test PostgreSQL connection directly
docker exec pgvector-container pg_isready -h localhost -p 5432 -U postgres

# Should return: localhost:5432 - accepting connections
```

## 📊 **Benefits of This Approach**

1. **Production Ready**: No silent failures or mock data
2. **Fail Fast**: Immediate feedback if database is down  
3. **Cleaner Code**: No complex fallback logic
4. **Predictable**: Always uses real PostgreSQL data
5. **Performance**: Direct database access, no overhead

## 🔄 **Migration Complete**

Your system now:
- ✅ **Always** connects to PostgreSQL
- ✅ **Fails explicitly** if database unavailable
- ✅ Uses **real document retrieval** with custom reranking
- ✅ Supports both **HyDE** and **Direct** retrieval modes
- ✅ Has **clean, production-ready** code

Make sure your PostgreSQL container is running and properly configured before starting the system! 🚀
