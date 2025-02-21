import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.linear_model import LinearRegression
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from contextlib import contextmanager

# 🌍 Çevresel değişkenleri yükle
load_dotenv(override=True)

# API anahtarını önce .env'den almayı deneyelim
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Eğer .env'den alınamazsa, bir üst dizindeki .env'yi kontrol edelim
if not GOOGLE_API_KEY:
    load_dotenv(dotenv_path="../.env", override=True)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Son kontrol ve hata mesajı
if not GOOGLE_API_KEY:
    st.error("Google API anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin.")
    st.stop()

# 🎯 Agent Tipleri
class AgentType:
    DATA = "data"
    CAUSAL = "causal"
    QA = "qa"

# 📌 LLM Modelini Tek Yerde Tanımla
@st.cache_resource  # Cache the model loading
def load_llm_model():
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.05,
        google_api_key=GOOGLE_API_KEY,
        max_output_tokens=2048,
        top_p=0.9,
        top_k=20,
        timeout=120,
        retry_max_attempts=3,
        retry_min_wait=1,
        cache=False
        
    )
    return llm

# 📌 TEMPLATE'LER
DATA_ANALYSIS_TEMPLATE = """Sen deneyimli bir veri bilimci ve ekonomist olarak, verilen veri setini kullanarak kapsamlı ve görsel analizler yapacaksın.


VERİ SETİ HAKKINDA:
- Toplam Ülke Sayısı: {total_countries}
- Yıl Aralığı: {year_range}
- Veri Setindeki Metrik Sayısı: {metrics_count}
- Mevcut Metrikler: {metrics}
- Bölgeler: {regions}
- Global Mutluluk Ortalaması: {global_mean:.2f}
- Ortalama GDP Per Capita: {mean_gdp_per_capita:.2f}
- Ortalama Yaşam Beklentisi: {mean_life_expectancy:.2f}
- Ortalama İşsizlik Oranı: {mean_unemployment_rate:.2f}%
- Ortalama İnternet Kullanım Oranı: {mean_internet_users_percent:.2f}%
- G20 Üyesi Ülke Sayısı: {g20_count}
- OECD Üyesi Ülke Sayısı: {oecd_count}
- BRICS Üyesi Ülke Sayısı: {brics_count}
- En Mutlu Ülke: {happiest}
- En Mutsuz Ülke: {unhappiest}


TEMEL PRENSİPLER:

1. **STRATEJİK ANALİZ KATMANLARI**
   - Veriye ekonomik teori ve sosyal dinamikler lensinden bakış
   - Makro-mikro etkileşimlerin değerlendirilmesi
   - Disiplinlerarası perspektif entegrasyonu

2. **DERİN İÇGÖRÜ GELİŞTİRME**
   - Paradoksal ilişkilerin ortaya çıkarılması
   - Zaman serisi anomalilerinin yorumlanması
   - Yapısal kırılma noktalarının analizi
   - Benchmarking ile performans skalası oluşturma

3. **UZMAN YORUM MODELİ**
   - Ekonomik Göstergelerin Sosyal Etki Matrisi
   - Politikaların Çoklu Senaryo Simülasyonu
   - Regresyon Temelli Nedensellik Çerçevesi
   - Küresel Trendlerle Uyum Analizi

4. **GÖRSEL NARRATİF**
   - Heatmap ile çoklu parametre etkileşimleri
   - Radar grafiklerle çok boyutlu performans karşılaştırması
   - Boxplot ile bölgesel dağılım anomalileri
   - Zaman eksenli çoklu gösterge overlays

RAPORLAMA YAPISI:

📊 **Kritik Performans Değerlendirmesi**
- Göstergelerin sistemik önem derecelendirmesi
- Küresel sıralamadaki konumun jeopolitik etkileri
- Anahtar performans açıklarının kök neden analizi

📈 **Dinamik Trend Yorumlaması**
- Dönemsel volatilite kaynaklarının tespiti
- Trendlerin küresel makroekonomik döngülerle ilişkisi
- Sürdürülebilirlik endeksi projeksiyonları

🌍 **Yapısal Karşılaştırma Analitiği**
- Bölgesel liderlerle yetenek gap analizi
- Demografik farklılaşmanın sosyoekonomik etkisi
- Kurumsal kapasite-başarı korelasyon haritası

🔍 **Nedensel İlişki Mimarisi**
- Çoklu regresyonla dominant faktör tespiti
- Gecikmeli etki (lag effect) modellenmesi
- Eşik değerlerinin (threshold) politika etkisi

💡 **Stratejik Öngörü Çerçevesi**
- Senaryo temelli optimizasyon modeli
- Politika çarpan etkisi simülasyonları
- Kaynak tahsisi için öncelik matrisi

🧠 **Uzman Perspektifi**
- "Bu trend sosyal sermayede neyi gösteriyor?"
- "Ekonomik göstergelerin sosyal refaha yansıma mekanizması"
- "Yapısal reformlar için kritik kaldıraç noktaları"
- "Küresel şoklara karşı direnç analizi"

Görsel Entegrasyon:
- [Interaktif Dashboard: Tüm metriklerin real-time ilişkisi]
- [Bubble Chart: GDP/Mutluluk/Nüfus dinamikleri]
- [Parallel Coordinates: Çok boyutlu ülke profilleme]

Soru: {question}

Yanıtını verirken mutlaka veri setindeki gerçek değerleri kullan ve görsellerle destekle. Her sayısal değer ve trend veri setinden gelmeli."""

FINAL_CAUSAL_ANALYSIS_TEMPLATE = """
Sen NOBEL ÖDÜLLÜ UZMAN bir Veri bilimci, sosyal bilimci ve mutluluk araştırmacısısın. Aşağıdaki soruya, verisetindeki {variables} değişkenlerine dayanarak, tamamen veri odaklı, sayısal verilerle desteklenmiş ve derin içgörülerle zenginleştirilmiş kapsamlı bir analiz yapacaksın. Yanıtın; dış kaynaklara veya ek varsayımlara yer vermeden, sadece mevcut veriler üzerinden oluşturulmalı ve okuyucunun ilgisini çekecek akıcı bir dille sunulmalıdır.

(Not: Soru içerisinde "neden", "niye", "sebebi", "etkisi", "faktör" gibi tetikleyici kelimeler geçerse bu şablon aktif hale gelir. Eğer veri yetersizse, "Veri setimizde bu konuya ilişkin yeterli bilgi bulunmamaktadır" ifadesini kullan.)

TEMEL PRENSİPLER:

1. VERİ ODANLI ANALİZ:
   - Yanıtını, sadece {variables} içerisindeki sayısal veriler, istatistiksel hesaplamalar (ör. korelasyon, p-değerleri, trendler) ve karşılaştırmalar üzerinden oluştur.
   - Dış kaynaklara veya ek varsayımlara yer vermeden, mevcut veri noktalarına sıkı sıkıya bağlı kal.

2. DERİN İÇGÖRÜ VE UZMAN ANALİZİ:
   - Elde ettiğin sayısal bulguların arkasındaki nedenleri, etki mekanizmalarını ve stratejik sonuçları açık ve net cümlelerle ifade et.
   - Her bulgunun, hangi politika ya da uygulamalara işaret ettiğini ve toplumsal dinamiklere nasıl yansıdığını yorumla.
   - Geleceğe yönelik öngörüler, stratejik çıkarımlar ve öneriler ekleyerek, verinin pratik anlamını ortaya koy.

3. MODÜLER VE ESNEK YAPI:
   - Yanıtın belirli bölümlerini (ör. uzman yorumları, ülke/bölge özel analizi, görselleştirme) koşullara bağlı modüller şeklinde sun. Örneğin, verinin yetersiz olduğu durumlarda ilgili modülleri atlayarak "Veri setimizde bu konuya ilişkin yeterli bilgi bulunmamaktadır" uyarısı ver.
   - İhtiyaca göre dinamik yer tutucular (örn. country, year_range) ekleyerek yanıtı daha uyarlanabilir hale getir.

4. YAPILANDIRILMIŞ YANIT:
   🔍 **VERİSEL BULGULAR VE SAYISAL ÖZET:**
      - [Faktör 1] ile mutluluk: r=[değer], p=[değer].  
        Açıklama: Bu bulgu, [Faktör 1]'in artışının mutluluk üzerinde güçlü ve anlamlı bir etkisi olduğunu gösterir.
      - [Faktör 2] ile mutluluk: r=[değer], p=[değer].  
        Açıklama: Bu değer, [Faktör 2]'deki değişimin doğrudan mutluluk düzeyine yansıdığını ortaya koyar.
      - (Varsa ek sayısal bulgular ve hesaplamalar eklenebilir.)

   💡 **DERİN İÇGÖRÜ VE STRATEJİK ANALİZ:**
      - "[Faktör 1]'deki 1 birimlik artışın, mutluluk skorunu yaklaşık [Y] birim artırdığı gözlemlenmiştir. Bu, [ilgili sosyal dinamik/politik alan] üzerinde önemli bir etki yaratmaktadır."
      - "[Faktör 2]'deki artış, [belirtilen stratejik sonuç] ile ilişkilendirilmiştir. Bu durum, [ilgili uygulama veya politika] açısından değerli çıkarımlar sunmaktadır."
      - Bu bölümde, elde edilen veriler ışığında geleceğe yönelik öngörüler, stratejik öneriler ve potansiyel politika tavsiyeleri de yer almalıdır.

   🌍 **ÜLKE/BÖLGE ÖZEL ANALİZİ VE KARŞILAŞTIRMALI BAKIŞ:**
      - Ülke veya bölge özelinde güncel durum, sıralama ve performans kriterlerini, sayısal verilerle destekleyerek açıkla.
      - Benzer ülkeler veya bölgeler arasındaki farkları verilerle kıyaslayarak, ilgili örnekler ve karşılaştırmalar sun.

   📈 **GÖRSEL DESTEK (OPSİYONEL):**
      - Eğer uygunsa, analizini desteklemek için [görsel X: <tip> <ülke/bölge> <metrik>] formatında en fazla 2 görsel ekle.
      - Görseller, verisetindeki trendleri, karşılaştırmaları veya ilişkileri netleştirmelidir.

   ⚠️ **VERİ YETERSİZLİĞİ DURUMUNDA:**
      - Eğer mevcut veri seti, soruya ilişkin yeterli bilgi sağlamıyorsa, yanıtında "Veri setimizde bu konuya ilişkin yeterli bilgi bulunmamaktadır" ifadesini kullan.

Kontrol Listesi (YANIT ÜRETİMİNDE DİKKAT EDİLMESİ GEREKEN NOKTALAR):
   - Veri setindeki tüm ilgili değişkenler (ör. {variables}) kullanıldı mı?
   - İstatistiksel hesaplamalar (r, p-değerleri vb.) net ve doğru biçimde belirtildi mi?
   - Uzman yorumları, veriye dayalı, tutarlı ve stratejik öngörüler sunuyor mu?
   - Yanıt, verisetinin dışına çıkmadan, sadece mevcut veri üzerinden oluşturuldu mu?
   - Görseller, analizle uyumlu ve açıklayıcı şekilde entegre edildi mi?

Soru: {question}

NOT:
   - Yanıt tamamen {variables} içerisindeki verilere dayanmalıdır.
   - Dış kaynak veya ek varsayım kullanılmadan, yalnızca mevcut veri noktaları üzerinden cevap oluştur.
   - Yanıt, akıcı, ilgi çekici ve okuyucuyu sıkmadan, sayısal verilerle desteklenmiş derin analiz ve stratejik öngörüler içermelidir.
"""







GENERAL_QA_TEMPLATE = """
Sen deneyimli bir veri bilimci, ekonomist ve mutluluk araştırmacısısın. Verilen veri setindeki {variables} değişkenlerini esas alarak, soruları detaylı, sayısal ve anlamlı bir şekilde yanıtlayacaksın. Analizlerini görsellerle destekleyebilirsin. Yanıtların, dış kaynaklara veya varsayımlara gitmeden, yalnızca mevcut veri seti bilgilerine dayalı olmalıdır.

TEMEL PRENSİPLER:

1. VERİ ODAKLI YAKLAŞIM:
   - Analizlerini veri setindeki gerçek verilere dayandır.
   - Önemli sayısal bulguları (ör. korelasyon, p-değerleri, trendler) açıkça vurgula.
   - İstatistiksel analizler ve karşılaştırmalar yap; örnek hesaplamalarla destekle.
   - Anlamlı trendleri, kalıpları ve ilişkileri belirle.
   - Yanıtın, verisetinin dışına çıkmadan sadece mevcut veriler üzerinden oluşturulmalı.

2. BÜTÜNCÜL DEĞERLENDİRME:
   - Çoklu faktörleri ve ilişkileri incele.
   - Farklı açılardan karşılaştırmalar yap (ör. bölgesel, global, zaman içindeki değişim).
   - Karşılaştırmalı grafikler, tablolar ve diğer görsellerle destekle.

3. ANLAMLI İÇGÖRÜ GELİŞTİRME:
   - Verilerden derin çıkarımlar yap ve kritik noktaları belirt.
   - Beklenmedik sonuçları, önemli ilişkileri ve kalıpları cümleler halinde açıkla.
   - İstatistiksel bulguları, mantıksal çıkarımlarla yorumla.

4. UZMAN YORUM MODELİ:
   - Verilere dayalı uzman görüşlerini ekle.
   - Her bir sayısal bulgunun arkasındaki olası nedenleri tartış; örnek olaylarla destekle.
   - Geleceğe yönelik projeksiyonlar, politika önerileri ve stratejik çıkarımlar sun.
   - Yanıtın, verisetindeki {variables} bilgilerine tamamen bağlı kalmalıdır.

5. STRATEJİK ANALİZ KATMANLARI:
   - Ülke veya bölge özel analizinde, güncel durum, sıralama ve performans kriterlerini detaylandır.
   - Başarı ve başarısızlık hikayeleri ile örnek olaylara yer ver.
   - Karşılaştırmalı analizler yap; benzer ülkeler veya bölgeler arasındaki farkları ortaya koy.
   - Stratejik öneriler ve uzun vadeli öngörüler ekle.

GÖRSELLEŞTİRME SEÇENEKLERİ:
   - 📈 Trend Grafikleri: Zaman serisi analizleri, büyüme eğrileri, karşılaştırmalı trendler.
   - 📊 Karşılaştırma Grafikleri: Bar grafikleri, kutu grafikleri, radar grafikleri.
   - 🗺️ Coğrafi Görselleştirmeler: Bölgesel karşılaştırmalar, küresel dağılımlar, sıcaklık haritaları.
   - 📉 İlişki Grafikleri: Saçılım grafikleri, korelasyon matrisleri, ağaç haritaları.
   - Görsel isteklerini şu formatta belirt:
         [görsel X: <tip> <ülke/bölge> <metrik>]
   - Maksimum 2 görsel kullanılmalı.

YANIT ÜRETİMİNDE DİKKAT EDİLMESİ GEREKEN NOKTALAR (Kontrol Listesi):
   - Veri setindeki tüm ilgili değişkenler kullanıldı mı?
   - İstatistiksel hesaplamalar (ör. r, p-değerleri) açıkça belirtildi mi?
   - Görseller analizle uyumlu ve açıklayıcı mı?
   - Uzman yorumları veriye dayalı, tutarlı ve mantıklı mı?
   - Yanıt, verisetinin dışına çıkmadan, sadece mevcut bilgiler üzerinden üretildi mi?

Soru: {question}

NOT:
   - Yanıt, verisetindeki {variables} bilgilerine tamamen bağlı olmalı.
   - Dış kaynak veya ek varsayım kullanmadan, yalnızca mevcut veri noktaları üzerinden cevap oluştur.
   - Yanıtın akıcı, anlaşılır, sayısal ve veri odaklı olmasına özen göster.
"""





# 🚀 Multi-Agent Sistemi

@st.cache_data
def calculate_analysis_inputs(df):
    return {
        "total_countries": int(df['country_name'].nunique()),
        "year_range": f"{df['year'].min()} - {df['year'].max()}",
        "metrics_count": len(df.columns),
        "metrics": ", ".join(df.columns),
        "regions": ", ".join(sorted(set(df['regional_indicator'].unique()))),
        "global_mean": float(df['life_ladder'].mean()),
        "mean_gdp_per_capita": float(df['gdp_per_capita'].mean()),
        "mean_life_expectancy": float(df['life_expectancy'].mean()),
        "mean_unemployment_rate": float(df['unemployment_rate'].mean()),
        "mean_internet_users_percent": float(df['internet_users_percent'].mean()),
        "g20_count": int(df['g20_member'].sum()),
        "oecd_count": int(df['oecd_member'].sum()),
        "brics_count": int(df['brics_member'].sum()),
        "happiest": df.loc[df['life_ladder'].idxmax(), 'country_name'],
        "unhappiest": df.loc[df['life_ladder'].idxmin(), 'country_name'],
        "variables": ", ".join(df.columns)
    }

class MultiAgentSystem:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analysis_inputs = calculate_analysis_inputs(df)

        # Diğer gerekli hesaplamalar ve agent yapılandırmaları burada yapılabilir.
        self.agents = {
            AgentType.DATA: self._create_data_agent(),
            AgentType.CAUSAL: self._create_causal_agent(),
            AgentType.QA: self._create_qa_agent()
        }

    @st.cache_data(ttl=600)  # 10 dakika önbellekle
    def _calculate_trend_analysis(self, metric):
        """Zaman serisi trend analizini önbelleğe alarak hesapla."""
        if metric not in self.df.columns:
            return {}

        yearly_data = self.df.groupby('year')[metric].mean().reset_index()
        if len(yearly_data) < 2:
            return {}

        X = np.arange(len(yearly_data)).reshape(-1, 1)
        y = yearly_data[metric].values
        trend_model = LinearRegression().fit(X, y)

        return {
            "trend_direction": "artış" if trend_model.coef_[0] > 0 else "düşüş",
            "trend_strength": trend_model.score(X, y),
        }

    def _create_visualizations(self, analysis_type: str, metric: str) -> go.Figure:
        """Çeşitli veri analiz görsellerini oluştur."""
        if metric not in self.df.columns:
            return None

        fig = go.Figure()

        if analysis_type == "trend":
            yearly_data = self.df.groupby('year')[metric].mean().reset_index()
            fig = px.line(yearly_data, x="year", y=metric, title=f"{metric} Trend Analizi", template="plotly_dark")

        elif analysis_type == "comparison":
            latest_year = self.df["year"].max()
            latest_data = self.df[self.df["year"] == latest_year]
            fig = px.box(latest_data, x='regional_indicator', y=metric, title=f"{metric} Bölgesel Dağılım", template="plotly_dark")

        elif analysis_type == "correlation":
            corr_matrix = self.df[["life_ladder", "gdp_per_capita", metric]].corr()
            fig = px.scatter_matrix(self.df, dimensions=["life_ladder", "gdp_per_capita", metric], title="Korelasyon Matrisi")

        return fig

    def display_visuals(self, analysis_type: str, metric: str):
        with st.spinner('Görsel yükleniyor...'):
            fig = self._create_visualizations(analysis_type, metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    def _create_data_agent(self) -> LLMChain:
        """Veri analizi agent'ı oluştur."""
        prompt = PromptTemplate(
            template=DATA_ANALYSIS_TEMPLATE,
            input_variables=[
                    "total_countries",
                    "year_range",
                    "metrics_count",
                    "metrics",
                    "regions",
                    "global_mean",
                    "mean_gdp_per_capita",
                    "mean_life_expectancy",
                    "mean_unemployment_rate",
                    "mean_internet_users_percent",
                    "g20_count",
                    "oecd_count",
                    "brics_count",
                    "happiest",
                    "unhappiest",
        ]
        )
        return LLMChain(llm=load_llm_model(), prompt=prompt)


    def _create_causal_agent(self) -> LLMChain:
        """Nedensel analiz agent'ı oluştur."""
        prompt = PromptTemplate(
            template=FINAL_CAUSAL_ANALYSIS_TEMPLATE,
            input_variables=["question", "variables"]
        )
        # Daha spesifik veri analizi yönlendirmeleri ekleyelim
        prompt.template += "\n\nÖNEMLİ: Her iddia mutlaka sayısal bir veri ile desteklenmeli. Korelasyon katsayıları, ortalamalar ve yüzdelik değişimler kullanılmalı."
        return LLMChain(llm=load_llm_model(), prompt=prompt)

    def _create_qa_agent(self) -> LLMChain:
        """Genel soru-cevap agent'ı oluştur."""
        prompt = PromptTemplate(template=GENERAL_QA_TEMPLATE, input_variables=["question","variables"])
        return LLMChain(llm=load_llm_model(), prompt=prompt)

    def route_question(self, question: str) -> str:
        """Soruyu ilgili agent'a yönlendir."""
        question_lower = question.lower()
        if any(kw in question_lower for kw in ["neden", "niye", "sebebi", "etkisi", "faktör"]):
            return AgentType.CAUSAL
        elif any(kw in question_lower for kw in ["trend", "analiz", "karşılaştır", "grafik", "veri", "istatistik"]):
            return AgentType.DATA
        return AgentType.QA

    def get_answer(self, question: str) -> str:
        """Soruyu uygun agent'a yönlendir ve yanıt al."""
        # route_question sonucunu bir değişkene atıyoruz
        agent_type = self.route_question(question)

        # analysis_inputs sözlüğünün kopyasını alıp gerekli girişleri ekliyoruz
        inputs = self.analysis_inputs.copy()
        inputs["question"] = question

        # Eğer CAUSAL agent seçilmişse, "variables" anahtarını kesin olarak ekliyoruz.
        if agent_type == AgentType.CAUSAL:
            inputs["variables"] = ", ".join(self.df.columns)

        agent = self.agents.get(agent_type)
        return agent.invoke(inputs)["text"]

@st.cache_data(ttl=5000)  # 5 dakika sonra cache'i otomatik temizle
def load_dataset():
    try:
        # Veri yükleme kodunuz buraya
        df = pd.read_csv("cleaned_dataset.csv")  # Gerçek path'i kullanın
        return df
    except Exception as e:
     