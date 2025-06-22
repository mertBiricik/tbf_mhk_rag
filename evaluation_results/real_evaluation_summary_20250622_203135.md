# 🏀 REAL Basketball RAG System Evaluation Report

**Evaluation Date:** 2025-06-22 20:31:35  
**Total Queries:** 10  
**Successful Evaluations:** 10  
**Total Evaluation Time:** 57.36 seconds

## 📊 Overall Performance Summary

### ⏱️ Timing Performance
- **Average Retrieval Time:** 0.12s
- **Average RAG Generation:** 4.48s  
- **Average Base LLM Generation:** 1.13s
- **Average Total RAG Time:** 4.60s

### 🔍 Retrieval Quality
- **Average Source Coverage:** 0.0%
- **Average Documents Retrieved:** 5.0

### 📝 Answer Quality Comparison

| Metric | RAG | Base LLM | RAG Advantage |
|--------|-----|----------|---------------|
| **Citations/Sources** | 3.9 | 0.2 | 100.0% |
| **Relevance Score** | 1.1 | 1.0 | 30.0% |
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
- RAG system provides **3.9x** more citations than base LLM
- Total response time averages **4.60 seconds**

## 🔬 Detailed Query Results

### Query 1: 5 faul yapan oyuncuya ne olur?

**Category:** fouls | **Difficulty:** easy

#### 🤖 RAG Answer:
```
5 faul yapan oyuncu, oyunu derhal terk edecektir. (Kaynak 1 - basketbol_oyun_kurallari_degisiklikleri_2024.txt)

Madde 40.1 - 5 faul yapmış olan bir oyuncu hakem tarafından bilgilendirilecek ve oyunu derhal terk edecektir.

Bu oyuncuya verilecek başka herhangi bir teknik ya da diskalifiye edici faul veya sportmenlik dışı davranış faulü, başantrenöre 'B' olarak kaydedilecek ve buna göre cezalandırılacaktır. (Kaynak 3 - basketbol_oyun_kurallari_resmi_yorumlar_2023.txt)

Madde 37-8
```

#### 🧠 Base LLM Answer:
```
Oyuncu saha dışına çıkarılır ve bir dakika sürecek ceza suresi başlar.
```

#### 📊 Performance Metrics:
- **Retrieval Time:** 0.83s
- **RAG Generation Time:** 6.53s  
- **Base Generation Time:** 0.50s
- **Source Coverage:** 0.0%
- **RAG Citations:** 5
- **Base Citations:** 0
- **Documents Retrieved:** 5

#### 📚 Sources Used:
- **1.** basketbol_oyun_kurallari_degisiklikleri_2024.txt (2024) - changes
- **2.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **3.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **4.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **5.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations

---

### Query 2: Şut saati kaç saniyedir?

**Category:** timing | **Difficulty:** easy

#### 🤖 RAG Answer:
```
Şut saati 14 saniye göstermektedir. (Kaynak 1 - basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023), 29/50-14)
```

#### 🧠 Base LLM Answer:
```
20 saniye.
```

#### 📊 Performance Metrics:
- **Retrieval Time:** 0.04s
- **RAG Generation Time:** 1.93s  
- **Base Generation Time:** 0.17s
- **Source Coverage:** 0.0%
- **RAG Citations:** 3
- **Base Citations:** 0
- **Documents Retrieved:** 5

#### 📚 Sources Used:
- **1.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **2.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **3.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **4.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **5.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations

---

### Query 3: Teknik faul ile kişisel faul arasındaki fark nedir?

**Category:** fouls | **Difficulty:** medium

#### 🤖 RAG Answer:
```
Teknik faul ile kişisel faul arasındaki fark, temassız faullerdir. Teknik faul davranışsal nitelikteki temassız faulleri ifade ederken, kişisel faul ise top canlı ya da ölü iken bir oyuncunun rakibine yaptığı kural dışı teması ifade eder.

Kaynak 1 - basketbol_oyun_kurallari_2022.txt (2022):
Madde 34.1.1
Tanım: Kişisel faul, top canlı ya da ölüyken bir oyuncunun rakibine yaptığı kural dışı temastır.
Madde 34.2.1
Tanım: Teknik faul, davranışsal nitelikteki temassız faulleri ifade eder.

Kaynak 3 - basketbol_oyun_kurallari_2022.txt (2022):
B.8.3.4
Teknik faul, davranışsal nitelikteki temassız faullerdir.
Madde 36.2.3 ve Madde 36.2.4
Diskalifiye edici teknik faul, oyuncuya veya başantrenöre verilebilir.

Kaynak 5 - basketbol_oyun_kurallari_2022.txt (2022):
Bir oyuncu ya da bir başantrenör Madde 36.2.3 veya Madde 36.2.4 ‘e göre diskalifiye edilirse, bu teknik faul olarak cezalandırılacak tek faul olacak ve diskalifiyeden dolayı hiçbir ek ceza uygulanmayacaktır.
```

#### 🧠 Base LLM Answer:
```
Teknik faul, oyuncunun eylemleri nedeniyle hakem tarafından verilen bir ceza türüdür. Kişisel faul ise, rakip oyuncuya fiziksel temas veya müdahalede bulunmak suretiyle yapılan hatalardır.
```

#### 📊 Performance Metrics:
- **Retrieval Time:** 0.05s
- **RAG Generation Time:** 8.11s  
- **Base Generation Time:** 1.07s
- **Source Coverage:** 0.0%
- **RAG Citations:** 4
- **Base Citations:** 0
- **Documents Retrieved:** 5

#### 📚 Sources Used:
- **1.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **2.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **3.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **4.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **5.** basketbol_oyun_kurallari_2022.txt (2022) - rules

---

### Query 4: 2024 yılında hangi kurallar değişti?

**Category:** rule_changes | **Difficulty:** medium

#### 🤖 RAG Answer:
```
2024 yılında, 44. madde ile Düzeltilebilir Hata'da yapılan değişikliklerden dolayı oyuna dahil olan tüm katılımcılar için ortak bir ifade oluşturuldu (Kaynak 3 - basketbol_oyun_kurallari_degisiklikleri_2024.txt).

Ayrıca, faul kısıtlamaları ve oyundan diskalifiyeler ile ilgili değişikler yapıldı. Bir oyuncu 5 faul yapmışsa oyunu derhal terk edecektir (Kaynak 4 - basketbol_oyun_kurallari_degisiklikleri_2024.txt).
```

#### 🧠 Base LLM Answer:
```
2024 yılında FIBA, basketbolun geleceğini şekillendirmek için yeni bir dizi kural ve uygulamayı tanıttı. Bu değişikliklerin bazıları:

- 3'lü hücumda, topu kaptırma veya elden kaybetme gibi durumlarda, topun yeniden başlatılması için "hedefe yakın" kavramı kaldırıldı.
- Oyuncuların basketbol topunu elle tutarken, topun elle tutulduğu pozisyonda, topun elle tutulan elin bilekten daha yukarıda olması gerekliliği kaldırıldı.
- Topu elle tutan oyuncu, topu elle tuttuğu pozisyonda, topun elle tutulan elin bilekten daha aşağıda olması durumunda, topu elle tutan oyuncunun basketbol topunu elle tuttuğunda, topun elle tutulduğu pozisyonda, topun elle tutulan elin bilekten daha yukarıda olması gerekliliği kaldırıldı.
- Oyuncuların basketbol topunu elle tutarken, topun elle tutulduğu pozisyonda, topun elle tutulan elin bilekten daha aşağıda olması durumunda, topu elle tutan oyuncunun basketbol topunu elle tuttuğunda, topun elle tutulduğu pozisyonda, topun elle tutulan elin bilekten daha yukarıda olması gerekliliği
```

#### 📊 Performance Metrics:
- **Retrieval Time:** 0.04s
- **RAG Generation Time:** 3.36s  
- **Base Generation Time:** 6.41s
- **Source Coverage:** 0.0%
- **RAG Citations:** 4
- **Base Citations:** 2
- **Documents Retrieved:** 5

#### 📚 Sources Used:
- **1.** basketbol_oyun_kurallari_degisiklikleri_2024.txt (2024) - changes
- **2.** basketbol_oyun_kurallari_degisiklikleri_2024.txt (2024) - changes
- **3.** basketbol_oyun_kurallari_degisiklikleri_2024.txt (2024) - changes
- **4.** basketbol_oyun_kurallari_degisiklikleri_2024.txt (2024) - changes
- **5.** basketbol_oyun_kurallari_degisiklikleri_2024.txt (2024) - changes

---

### Query 5: Oyuncu değişimi sırasında mola alınabilir mi?

**Category:** substitutions | **Difficulty:** hard

#### 🤖 RAG Answer:
```
Hayır, oyuncu değişimi sırasında mola alınamaz. (Kaynak 1 - basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023), 5-15)

Belge numarası: 5-15
```

#### 🧠 Base LLM Answer:
```
Hayır, oyuncu değişimi sırasında mola alınamaz.
```

#### 📊 Performance Metrics:
- **Retrieval Time:** 0.03s
- **RAG Generation Time:** 2.02s  
- **Base Generation Time:** 0.41s
- **Source Coverage:** 0.0%
- **RAG Citations:** 4
- **Base Citations:** 0
- **Documents Retrieved:** 5

#### 📚 Sources Used:
- **1.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **2.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **3.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **4.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **5.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations

---

### Query 6: Basketbol sahasının boyutları nelerdir?

**Category:** court_specifications | **Difficulty:** easy

#### 🤖 RAG Answer:
```
Saha (Diyagram 1) düz, engelsiz sert yüzeyli, sınır çizgisinin iç kenarından ölçüldüğünde 28 m uzunluğunda, 15 m genişliğinde olacaktır. (Kaynak 1 - basketbol_oyun_kurallari_2022.txt (2022), Madde 2.1)

Zemin, engellerden arındırılmış en az 2 m'lik başka bir ilave sınır çizgisi ile çevrili saha alanını içerecektir (Diyagram 2). Bu nedenle zemin en az 32 m uzunluğunda ve 19 m genişliğinde olmalıdır. (Kaynak 1 - basketbol_oyun_kurallari_2022.txt (2022), Madde 2.2)
```

#### 🧠 Base LLM Answer:
```
Basketbol sahalarının uzunluğu 28 metredir, genişliği ise 15 metredir.
```

#### 📊 Performance Metrics:
- **Retrieval Time:** 0.07s
- **RAG Generation Time:** 4.85s  
- **Base Generation Time:** 0.50s
- **Source Coverage:** 0.0%
- **RAG Citations:** 4
- **Base Citations:** 0
- **Documents Retrieved:** 5

#### 📚 Sources Used:
- **1.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **2.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **3.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **4.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **5.** basketbol_oyun_kurallari_2022.txt (2022) - rules

---

### Query 7: Üçlük çizgisi mesafesi ne kadardır?

**Category:** court_specifications | **Difficulty:** easy

#### 🤖 RAG Answer:
```
Üçlük çizgisi mesafesi 6.75 m'dir (Kaynak 1 - basketbol_oyun_kurallari_2022.txt (2022), 2.5.4)

Dış kenarı, kenar çizgilerin iç kenarından 0.90 m olan, dip çizgiden ve dip çizgiye dik 2 paralel
çizgi.
Rakibin sepetinin tam merkezinin altındaki zemin üzerindeki noktadan yayın dış kenarına
6.75 m yarıçaplı bir yay.
```

#### 🧠 Base LLM Answer:
```
Cevap: 23,75 feet (7,24 metre)
```

#### 📊 Performance Metrics:
- **Retrieval Time:** 0.04s
- **RAG Generation Time:** 5.89s  
- **Base Generation Time:** 0.37s
- **Source Coverage:** 0.0%
- **RAG Citations:** 3
- **Base Citations:** 0
- **Documents Retrieved:** 5

#### 📚 Sources Used:
- **1.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **2.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **3.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **4.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **5.** basketbol_oyun_kurallari_2022.txt (2022) - rules

---

### Query 8: Hangi durumda oyun uzatmaya gider?

**Category:** game_flow | **Difficulty:** medium

#### 🤖 RAG Answer:
```
Dördüncü çeyrekte skor eşitse oyun uzatmaya gider. (Kaynak 2 - basketbol_oyun_kurallari_2022.txt (2022), Madde 8)

8.5, 8.6 ve 8.7 maddeleri oyun arası başlama ve bitme durumlarını açıklamaktadır.

Dördüncü çeyrekte skor eşitse oyun uzatmalarla devam edecektir. (Kaynak 2 - basketbol_oyun_kurallari_2022.txt (2022), Madde 8)

İçerde ve dışarıda 2 maçlı toplam sayı serili yarışma sisteminde, iki maçın toplam skoru 2. maçın sonunda eşitse bu maç uzatmalarla devam edecektir. (Kaynak 2 - basketbol_oyun_kurallari_2022.txt (2022), Madde 8)
```

#### 🧠 Base LLM Answer:
```
Oyunun normal süresi bittikten sonra, iki takım da aynı sayıda puanla eşit ise ve de maçın normal süresi bitiminde eşitlik bozulmamışsa, maça uzatma devresi eklenir.
```

#### 📊 Performance Metrics:
- **Retrieval Time:** 0.04s
- **RAG Generation Time:** 5.31s  
- **Base Generation Time:** 1.03s
- **Source Coverage:** 0.0%
- **RAG Citations:** 4
- **Base Citations:** 0
- **Documents Retrieved:** 5

#### 📚 Sources Used:
- **1.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **2.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **3.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **4.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **5.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations

---

### Query 9: Hakem kararlarına itiraz edilebilir mi?

**Category:** officiating | **Difficulty:** hard

#### 🤖 RAG Answer:
```
Hakem kararlarına itiraz edilebilir. (Kaynak 1 - basketbol_oyun_kurallari_degisiklikleri_2024.txt, C.1)

İtiraz nedenleri, oyunun bitiminden en geç 15 dakika sonra kaptan tarafından başhakeme bildirilecek ve maç kağıdındaki “İtiraz durumunda kaptanın imzası” bölümü imzalanacaktır. (Kaynak 2 - basketbol_oyun_kurallari_2022.txt, a)

Takım, oyunun bitiminden en geç 1 saat sonra itiraz nedenlerini başhakeme yazılı olarak sunacaktır. (Kaynak 2 - basketbol_oyun_kurallari_2022.txt, b)
```

#### 🧠 Base LLM Answer:
```
Hakem kararlarına itiraz edilemez.
```

#### 📊 Performance Metrics:
- **Retrieval Time:** 0.04s
- **RAG Generation Time:** 4.50s  
- **Base Generation Time:** 0.28s
- **Source Coverage:** 0.0%
- **RAG Citations:** 4
- **Base Citations:** 0
- **Documents Retrieved:** 5

#### 📚 Sources Used:
- **1.** basketbol_oyun_kurallari_degisiklikleri_2024.txt (2024) - changes
- **2.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **3.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **4.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **5.** basketbol_oyun_kurallari_2022.txt (2022) - rules

---

### Query 10: Video hakem sistemi ne zaman kullanılır?

**Category:** technology | **Difficulty:** hard

#### 🤖 RAG Answer:
```
Video hakem sistemi, oyunun herhangi bir anında veya herhangi bir şiddet eylemi sırasında oyuncuların katılımını belirlemek için kullanılır. (Kaynak 2 - basketbol_oyun_kurallari_resmi_yorumlar_2023.txt)

Belge numarası: F-3.3.17
```

#### 🧠 Base LLM Answer:
```
Video hakem sistemi, maçın normal akışını bozmayacak şekilde ve hakemlerin görüşüne katkıda bulunacak şekilde kullanılır.
```

#### 📊 Performance Metrics:
- **Retrieval Time:** 0.06s
- **RAG Generation Time:** 2.28s  
- **Base Generation Time:** 0.61s
- **Source Coverage:** 0.0%
- **RAG Citations:** 4
- **Base Citations:** 0
- **Documents Retrieved:** 5

#### 📚 Sources Used:
- **1.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **2.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **3.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations
- **4.** basketbol_oyun_kurallari_2022.txt (2022) - rules
- **5.** basketbol_oyun_kurallari_resmi_yorumlar_2023.txt (2023) - interpretations

---

