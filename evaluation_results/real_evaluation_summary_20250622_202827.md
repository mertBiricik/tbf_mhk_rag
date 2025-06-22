# 🏀 REAL Basketball RAG System Evaluation Report

**Evaluation Date:** 2025-06-22 20:28:27  
**Total Queries:** 10  
**Successful Evaluations:** 10  
**Total Evaluation Time:** 62.13 seconds

## 📊 Overall Performance Summary

### ⏱️ Timing Performance
- **Average Retrieval Time:** 0.11s
- **Average RAG Generation:** 5.22s  
- **Average Base LLM Generation:** 0.88s
- **Average Total RAG Time:** 5.33s

### 🔍 Retrieval Quality
- **Average Source Coverage:** 0.0%
- **Average Documents Retrieved:** 5.0

### 📝 Answer Quality Comparison

| Metric | RAG | Base LLM | RAG Advantage |
|--------|-----|----------|---------------|
| **Citations/Sources** | 4.0 | 0.2 | 100.0% |
| **Relevance Score** | 0.9 | 0.8 | 30.0% |
| **Confidence Level** | Higher | Lower | 0.0% |

## 📈 Performance by Category

### Fouls
- **Queries:** 2
- **Source Coverage:** 0.0%
- **RAG Citation Win Rate:** 100.0%

### Timing
- **Queries:** 1
- **Source Coverage:** 0.0%
- **RAG Citation Win Rate:** 100.0%

### Rule Changes
- **Queries:** 1
- **Source Coverage:** 0.0%
- **RAG Citation Win Rate:** 100.0%

### Substitutions
- **Queries:** 1
- **Source Coverage:** 0.0%
- **RAG Citation Win Rate:** 100.0%

### Court Specifications
- **Queries:** 2
- **Source Coverage:** 0.0%
- **RAG Citation Win Rate:** 100.0%

### Game Flow
- **Queries:** 1
- **Source Coverage:** 0.0%
- **RAG Citation Win Rate:** 100.0%

### Officiating
- **Queries:** 1
- **Source Coverage:** 0.0%
- **RAG Citation Win Rate:** 100.0%

### Technology
- **Queries:** 1
- **Source Coverage:** 0.0%
- **RAG Citation Win Rate:** 100.0%

## 🎯 Key Findings

### RAG System Advantages:
- **100.0%** of answers had better source citations
- **30.0%** of answers were more relevant to the query
- **0.0%** of answers showed higher confidence (less uncertainty)

### Performance Insights:
- Average retrieval covers **0.0%** of expected sources
- RAG system provides **4.0x** more citations than base LLM
- Total response time averages **5.33 seconds**

## 🔬 Detailed Query Results

### Example 1: 5 faul yapan oyuncuya ne olur?

**Category:** fouls | **Difficulty:** easy

**RAG Answer:** 5 faul yapan oyuncu, oyunu derhal terk edecektir. (Kaynak 1 - basketbol_oyun_kurallari_degisiklikleri_2024.txt)

Madde 40.1: "5 faul yapmış olan bir oyuncu hakem tarafından bilgilendirilecek ve oyunu ...

**Base LLM Answer:** Oyuncu saha dışına çıkarılır ve bir dakika sürecek ceza suresi başlar.

**Metrics:**
- Retrieval Time: 0.66s
- Source Coverage: 0.0%
- RAG Citations: 4
- Base Citations: 0

---

### Example 2: Şut saati kaç saniyedir?

**Category:** timing | **Difficulty:** easy

**RAG Answer:** Şut saati 14 saniye göstermektedir. (Kaynak 1 - basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023), 29/50-14)

**Base LLM Answer:** 20 saniye.

**Metrics:**
- Retrieval Time: 0.03s
- Source Coverage: 0.0%
- RAG Citations: 3
- Base Citations: 0

---

### Example 3: Teknik faul ile kişisel faul arasındaki fark nedir?

**Category:** fouls | **Difficulty:** medium

**RAG Answer:** Teknik faul ile kişisel faul arasındaki fark, temassız faullerdir. Teknik faul davranışsal nitelikteki temassız faulleri ifade ederken, kişisel faul ise top canlı ya da ölü iken bir oyuncunun rakibine...

**Base LLM Answer:** Teknik faul, oyuncunun eylemleri nedeniyle hakem tarafından verilen bir ceza türüdür. Kişisel faul ise, rakip oyuncu veya hakemle fiziksel temas sonucu verilen ceza türüdür.

**Metrics:**
- Retrieval Time: 0.05s
- Source Coverage: 0.0%
- RAG Citations: 5
- Base Citations: 0

---

