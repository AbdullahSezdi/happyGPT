import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
import asyncio
import json
from scipy import stats
import re
# Load environment variables
load_dotenv()

# Configure Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")




@st.cache_data(ttl=3)  # 1 saat önbellek
def load_data():
    """Veri setini yükle ve önbellekle"""
    try:
        df = pd.read_csv('src/cleaned_dataset.csv')
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
        return None





@st.cache_data(ttl=3600)  # 1 saat önbellek
def preprocess_data(df):
    """Veri setini ön işle ve önbellekle"""
    try:
        # Ülke isimlerini standartlaştır
        country_mapping = {
                    'Turkey': 'Turkiye',
                    'Türkiye': 'Turkiye',
                    'Afganistan': 'Afghanistan',
                    'Arnavutluk': 'Albania',
                    'Cezayir': 'Algeria',
                    'Angola': 'Angola',
                    'Arjantin': 'Argentina',
                    'Ermenistan': 'Armenia',
                    'Avustralya': 'Australia',
                    'Avusturya': 'Austria',
                    'Azerbaycan': 'Azerbaijan',
                    'Bahreyn': 'Bahrain',
                    'Bangladeş': 'Bangladesh',
                    'Belarus': 'Belarus',
                    'Belçika': 'Belgium',
                    'Beliz': 'Belize',
                    'Benin': 'Benin',
                    'Bolivya': 'Bolivia',
                    'Bosna Hersek': 'Bosnia and Herzegovina',
                    'Botsvana': 'Botswana',
                    'Brezilya': 'Brazil',
                    'Brazilya': 'Brazil',
                    'Bulgaristan': 'Bulgaria',
                    'Burkina Faso': 'Burkina Faso',
                    'Burundi': 'Burundi',
                    'Kamboçya': 'Cambodia',
                    'Kanada': 'Canada',
                    'Orta Afrika Cumhuriyeti': 'Central African Republic',
                    'Çad': 'Chad',
                    'Şili': 'Chile',
                    'Çin': 'China',
                    'Kolombiya': 'Colombia',
                    'Komorlar': 'Comoros',
                    'Kosta Rika': 'Costa Rica',
                    'Hırvatistan': 'Croatia',
                    'Küba': 'Cuba',
                    'Kıbrıs': 'Cyprus',
                    'Çekya': 'Czechia',
                    'Danimarka': 'Denmark',
                    'Cibuti': 'Djibouti',
                    'Dominik Cumhuriyeti': 'Dominican Republic',
                    'Ekvador': 'Ecuador',
                    'El Salvador': 'El Salvador',
                    'Estonya': 'Estonia',
                    'Esvatini': 'Eswatini',
                    'Etiyopya': 'Ethiopia',
                    'Finlandiya': 'Finland',
                    'Fransa': 'France',
                    'Gabon': 'Gabon',
                    'Gürcistan': 'Georgia',
                    'Almanya': 'Germany',
                    'Gana': 'Ghana',
                    'Yunanistan': 'Greece',
                    'Guatemala': 'Guatemala',
                    'Gine': 'Guinea',
                    'Guyana': 'Guyana',
                    'Haiti': 'Haiti',
                    'Honduras': 'Honduras',
                    'Macaristan': 'Hungary',
                    'İzlanda': 'Iceland',
                    'Hindistan': 'India',
                    'Endonezya': 'Indonesia',
                    'Irak': 'Iraq',
                    'İrlanda': 'Ireland',
                    'İsrail': 'Israel',
                    'İtalya': 'Italy',
                    'Jamaika': 'Jamaica',
                    'Japonya': 'Japan',
                    'Ürdün': 'Jordan',
                    'Kazakistan': 'Kazakhstan',
                    'Kenya': 'Kenya',
                    'Kosova': 'Kosovo',
                    'Kuveyt': 'Kuwait',
                    'Letonya': 'Latvia',
                    'Lübnan': 'Lebanon',
                    'Lesotho': 'Lesotho',
                    'Liberya': 'Liberia',
                    'Libya': 'Libya',
                    'Litvanya': 'Lithuania',
                    'Lüksemburg': 'Luxembourg',
                    'Madagaskar': 'Madagascar',
                    'Malavi': 'Malawi',
                    'Malezya': 'Malaysia',
                    'Maldivler': 'Maldives',
                    'Mali': 'Mali',
                    'Malta': 'Malta',
                    'Moritanya': 'Mauritania',
                    'Meksika': 'Mexico',
                    'Moğolistan': 'Mongolia',
                    'Karadağ': 'Montenegro',
                    'Fas': 'Morocco',
                    'Mozambik': 'Mozambique',
                    'Myanmar': 'Myanmar',
                    'Namibya': 'Namibia',
                    'Hollanda': 'Netherlands',
                    'Yeni Zelanda': 'New Zealand',
                    'Nikaragua': 'Nicaragua',
                    'Nijer': 'Niger',
                    'Nijerya': 'Nigeria',
                    'Kuzey Makedonya': 'North Macedonia',
                    'Norveç': 'Norway',
                    'Umman': 'Oman',
                    'Pakistan': 'Pakistan',
                    'Panama': 'Panama',
                    'Paraguay': 'Paraguay',
                    'Peru': 'Peru',
                    'Filipinler': 'Philippines',
                    'Polonya': 'Poland',
                    'Portekiz': 'Portugal',
                    'Katar': 'Qatar',
                    'Romanya': 'Romania',
                    'Ruanda': 'Rwanda',
                    'Suudi Arabistan': 'Saudi Arabia',
                    'Senegal': 'Senegal',
                    'Sırbistan': 'Serbia',
                    'Sierra Leone': 'Sierra Leone',
                    'Singapur': 'Singapore',
                    'Slovenya': 'Slovenia',
                    'Somali': 'Somalia',
                    'Güney Afrika': 'South Africa',
                    'Güney Sudan': 'South Sudan',
                    'İspanya': 'Spain',
                    'Sri Lanka': 'Sri Lanka',
                    'Sudan': 'Sudan',
                    'Surinam': 'Suriname',
                    'İsveç': 'Sweden',
                    'İsviçre': 'Switzerland',
                    'Tacikistan': 'Tajikistan',
                    'Tanzanya': 'Tanzania',
                    'Tayland': 'Thailand',
                    'Togo': 'Togo',
                    'Trinidad ve Tobago': 'Trinidad and Tobago',
                    'Tunus': 'Tunisia',
                    'Türkiye': 'Turkiye',
                    'Turkey': 'Turkiye',
                    'Türkmenistan': 'Turkmenistan',
                    'Uganda': 'Uganda',
                    'Ukrayna': 'Ukraine',
                    'Birleşik Arap Emirlikleri': 'United Arab Emirates',
                    'BAE': 'United Arab Emirates',
                    'Birleşik Krallık': 'United Kingdom',
                    'İngiltere': 'United Kingdom',
                    'ABD': 'United States',
                    'Amerika Birleşik Devletleri': 'United States',
                    'Uruguay': 'Uruguay',
                    'Özbekistan': 'Uzbekistan',
                    'Zambiya': 'Zambia',
                    'Zimbabve': 'Zimbabwe'
        }
        df['country_name'] = df['country_name'].replace(country_mapping)
        
        # Corruption değerlerini 0-1 arasına normalize et (eğer değilse)
        if df['perceptions_of_corruption'].max() > 1:
            df['perceptions_of_corruption'] = df['perceptions_of_corruption'] / df['perceptions_of_corruption'].max()
        
        # Yıl sütununu integer yap
        df['year'] = df['year'].astype(int)
        
        return df
    except Exception as e:
        st.error(f"Veri ön işleme sırasında hata oluştu: {str(e)}")
        return None







def parse_dynamic_chart_command(command, valid_countries, metric_mapping):
    """
    Beklenen komut formatı: 
      "chart_type: x=..., y=..., countries=..., [other_key=value,...]"
    Örnek: "line: x=year, y=life_ladder, countries=ingiltere,iran"
    """
    command = command.strip()
    # İlk kısmı (chart_type) ayır: örn. "line:" veya "scatter:" vb.
    m = re.match(r'(\w+):\s*(.*)', command)
    if not m:
        return None
    chart_type = m.group(1).lower()  # örn: "line", "scatter", "bar", "box" vs.
    params_str = m.group(2)
    params = {"chart_type": chart_type}
    # Parametreleri virgül ile ayıralım
    for part in params_str.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip().lower()
            if key == "countries":
                # Ülkeleri ayırıp baş harflerini büyük yapalım; geçerli ülkelere göre doğrulama yapalım
                country_list = [c.strip().title() for c in value.split(",") if c.strip()]
                valid = [c for c in country_list if c in valid_countries]
                params[key] = valid
            else:
                # Eğer x veya y parametresi ise; metric_mapping ile eşleştirelim
                if key in ["x", "y"]:
                    for mk, mv in metric_mapping.items():
                        if mk in value:
                            params[key] = mv
                            break
                    else:
                        params[key] = value
                else:
                    params[key] = value
    return params








def create_dynamic_chart(params, df):
    """
    params sözlüğündeki değerler doğrultusunda dinamik grafik oluşturur.
    Beklenen parametreler:
      - chart_type: "scatter", "line", "bar", "box" vb.
      - x: x ekseni için sütun adı (örneğin "year")
      - y: y ekseni için sütun adı (örneğin "life_ladder")
      - countries: (isteğe bağlı) listesi; eğer belirtilmişse veri filtrelenecek.
      - Diğer parametreler de ileride eklenebilir.
    """
    chart_type = params.get("chart_type", "scatter")
    x = params.get("x")
    y = params.get("y")
    countries = params.get("countries", None)
    # Eğer ülkeler belirtilmişse, veri kümesini filtreleyelim.
    if countries:
        df = df[df['country_name'].isin(countries)]
    # Grafik türüne göre Plotly Express kullanarak grafik oluşturalım.
    if chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, color="country_name", template="plotly_dark",
                         title=f"Scatter Grafiği: {x} vs {y}")
    elif chart_type in ["line", "trend"]:
        fig = px.line(df, x=x, y=y, color="country_name", template="plotly_dark",
                      title=f"Line Grafiği: {x} vs {y}")
    elif chart_type == "bar":
        fig = px.bar(df, x=x, y=y, color="country_name", template="plotly_dark",
                     title=f"Bar Grafiği: {x} vs {y}")
    elif chart_type == "box":
        fig = px.box(df, x=x, y=y, color="country_name", template="plotly_dark",
                     title=f"Box Grafiği: {x} vs {y}")
    else:
        # Varsayılan olarak scatter grafiği
        fig = px.scatter(df, x=x, y=y, color="country_name", template="plotly_dark",
                         title=f"Scatter Grafiği: {x} vs {y}")
    return fig






def process_llm_response(response, df):
    """
    LLM yanıtını satır satır işler. Eğer satır "line:", "bar:", "scatter:" vb. ile başlıyorsa,
    bu komut arka planda talimat olarak ayrıştırılır (parse_dynamic_chart_command ile),
    komut metni temizlenir (kullanıcıya görünmez) ve dinamik grafik oluşturulur.
    Diğer satırlar normal metin olarak ekrana yazdırılır.
    """
    try:
        response = str(response).strip()
        if not response:
            st.warning("Yanıt boş veya geçersiz format.")
            return
        
        # Geçerli ülke listesi ve metrik eşleştirmesi:
        valid_countries = df['country_name'].unique().tolist()
        metric_mapping = {
            'mutluluk': 'life_ladder',
            'sosyal destek': 'social_support',
            'özgürlük': 'freedom_to_make_life_choices',
            'gdp': 'gdp_per_capita',
            'yaşam beklentisi': 'life_expectancy',
            'işsizlik': 'unemployment_rate',
            'internet': 'internet_users_percent'
        }
        chart_type_mapping = {
            'line': 'line',
            'trend': 'line',
            'bar': 'bar',
            'scatter': 'scatter',
            'box': 'box'
        }
        
        # Yanıtı satırlara bölelim
        lines = response.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Eğer satır grafik komutuyla başlıyorsa:
            lower_line = line.lower()
            if lower_line.startswith("line:") or lower_line.startswith("trend:") or \
               lower_line.startswith("bar:") or lower_line.startswith("scatter:") or \
               lower_line.startswith("box:"):
                # Komut satırını parse edelim:
                params = parse_dynamic_chart_command(line, valid_countries, metric_mapping)
                if params is None:
                    st.write("Komut anlaşılmadı:", line)
                    continue
                # Temiz metin kısmı; komutun kendisini ekranda göstermiyoruz
                st.write("(Grafik komutu işlendi)")
                # Grafik oluştur ve göster:
                fig = create_dynamic_chart(params, df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Grafik komutu içermeyen satırları normal metin olarak göster.
                st.write(line)
        return None
    except Exception as e:
        st.error(f"Yanıt işlenirken hata oluştu: {str(e)}")
        st.warning("Ham yanıt:")
        st.code(response)
        return None



@st.cache_data(ttl=3600)  # 1 saat önbellek
async def get_answer(question, df):
    """LLM yanıtı al ve işle"""
    try:
        # Soruyu küçük harfe çevir ve Türkçe karakterleri normalize et
        question_lower = question.lower()
        tr_to_en = str.maketrans("çğıöşüİ", "cgiosui")
        question_normalized = question_lower.translate(tr_to_en)
        
        # Multi-agent sistemini kullan
        from llm_agents import MultiAgentSystem, ConversationManager
        
        # Singleton pattern ile conversation manager'ı oluştur
        if 'conversation_manager' not in st.session_state:
            st.session_state.conversation_manager = ConversationManager()
        
        # Multi-agent sistemini oluştur
        multi_agent = MultiAgentSystem(df)
        
        # Geçmiş bağlamı kontrol et
        relevant_history = st.session_state.conversation_manager.get_relevant_context(question)
        if relevant_history:
            with st.expander("Benzer Sorular", expanded=False):
                for entry in relevant_history:
                    st.write(f"Soru: {entry['question']}")
                    st.write(f"Yanıt: {entry['answer'][:200]}...")
                    st.write("---")
        
        # Soruyu yanıtla
        try:
            # Agent'dan yanıt al
            answer = await multi_agent.get_answer(question)
            
            # Yanıtı geçmişe ekle
            if answer:  # Boş yanıt değilse ekle
                st.session_state.conversation_manager.add_to_history(
                    question=question,
                    answer=answer,
                    agent_type=multi_agent.route_question(question)
                )
            
            return answer
            
        except Exception as e:
            st.error(f"Yanıt alınırken hata oluştu: {str(e)}")
            return None
        
    except Exception as e:
        st.error(f"Analiz sırasında hata: {str(e)}")
        return None






def main():
    # Sayfa konfigürasyonu
    st.set_page_config(
        page_title="Global Mutluluk Analisi",
        page_icon="🌍",
        layout="centered"  # Tam genişlik için wide kullanıyoruz
    )

    # Özel CSS Stilleri
    st.markdown("""
    <style>
        /* Ana Tema Değişkenleri */
        :root {
            --background-primary: #0F0F0F;
            --background-secondary: #121212;
            --accent-color: #007AFF;
            --accent-gradient: linear-gradient(135deg, rgba(170, 0, 255, 0.9) 0%, rgba(0, 122, 255, 0.9) 50%, rgba(0, 255, 231, 0.9) 100%);
            --text-primary: #FFFFFF;
            --text-secondary: rgba(255, 255, 255, 0.8);
            --card-background: rgba(18, 18, 18, 0.8);
            --card-border: rgba(255, 255, 255, 0.1);
            --card-hover-border: rgba(0, 122, 255, 0.5);
        }

        /* Genel Stiller */
        .main {
            background: linear-gradient(135deg, #0A0A0A 0%, #1C1C1C 100%);
            color: var(--text-primary);
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
        }

        /* Hero Section */
        .hero-section {
            background: var(--background-primary);
            padding: 4rem 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            margin-bottom: 2rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .hero-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at center, rgba(0, 122, 255, 0.15) 0%, transparent 70%);
            pointer-events: none;
        }

        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, #00FFE7 0%, #007AFF 50%, #AA00FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 10px rgba(0, 198, 255, 0.3);
        }

        .hero-subtitle {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.9);
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
            font-weight: 500;
        }

        /* Dashboard Kartları */
        .dashboard-card {
            background: var(--card-background);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }

        .dashboard-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 122, 255, 0.1);
            border-color: var(--accent-color);
        }

        /* Parla Sayfası Stilleri */
        .qa-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* Spinner mesajını ortalama */
        .spinner-container {
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            font-size: 18px;
            font-weight: 500;
            height: 100vh;
        }

        /* Info card */
        .info-card {
            background: var(--card-background);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }

        .info-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent-color);
        }

        /* Chat message */
        .chat-message {
            background: var(--card-background);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            transition: all 0.3s ease;
        }

        .chat-message:hover {
            transform: translateY(-2px);
            border-color: var(--accent-color);
        }

        /* Input alanı */
        .stTextArea textarea {
            background: var(--card-background) !important;
            border: 1px solid var(--card-border) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            color: var(--text-primary) !important;
            height: 120px !important;
            transition: all 0.3s ease !important;
            font-size: 1rem !important;
            line-height: 1.5 !important;
            resize: none !important;
        }

        .stTextArea textarea:hover {
            border-color: var(--accent-color) !important;
            transform: translateY(-2px);
        }

        .stTextArea textarea:focus {
            border-color: var(--accent-color) !important;
            background: rgba(255, 255, 255, 0.07) !important;
            box-shadow: 0 4px 12px rgba(0, 122, 255, 0.1) !important;
            outline: none !important;
        }

        /* Butonlar */
        .stButton > button {
            background: var(--accent-gradient) !important;
            color: white !important;
            border: none !important;
            padding: 0.8rem 2rem !important;
            border-radius: 12px !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px !important;
            text-transform: uppercase !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 16px rgba(170, 0, 255, 0.2) !important;
        }

        /* Metrikler */
        .metric-container {
            background: var(--card-background);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        .metric-container:hover {
            border-color: var(--card-hover-border);
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 122, 255, 0.1);
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.75rem;
            line-height: 1.2;
        }

        .metric-label {
            color: var(--text-primary);
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            letter-spacing: 0.5px;
        }

        .metric-score {
            color: var(--text-secondary);
            font-size: 1rem;
            font-weight: 400;
        }

        /* Grafikler için container */
        .chart-container {
            background: var(--card-background);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2rem;
            margin: 1.5rem 0;
            transition: all 0.3s ease;
        }

        .chart-container:hover {
            border-color: var(--card-hover-border);
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 122, 255, 0.1);
        }

        /* Tab stilleri */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-radius: 12px !important;
            gap: 1rem;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: var(--card-background) !important;
            border-radius: 12px !important;
            border: 1px solid var(--card-border) !important;
            color: var(--text-primary) !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            border-color: var(--accent-color) !important;
            transform: translateY(-2px);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: var(--accent-gradient) !important;
            border: none !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--background-secondary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--accent-color);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent-gradient);
        }

        /* Responsive Tasarım */
        @media (max-width: 768px) {
            .qa-container {
                padding: 1rem;
            }

            .hero-title {
                font-size: 2.5rem;
            }

            .hero-subtitle {
                font-size: 1rem;
            }

            .metric-value {
                font-size: 2rem;
            }
            
            .metric-label {
                font-size: 1rem;
            }
            
            .section-title {
                font-size: 1.25rem;
            }
            
            .chart-container {
                padding: 1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Dünya Mutluluk Analizi</h1>
            <p class="hero-subtitle">Küresel mutluluk verilerini keşfedin ve ülkeler arasındaki ilişkileri analiz edin</p>
        </div>
    """, unsafe_allow_html=True)

    try:
        # Veri yükleme ve işleme
        df = load_data()
        if df is None:
            st.error("Veri yüklenemedi! Lütfen 'cleaned_dataset.csv' dosyasının varlığını kontrol edin.")
            return

        df = preprocess_data(df)
        if df is None:
            st.error("Veri işlenemedi!")
            return

        # Session state başlangıcı
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Ana-Sayfa'

        # Ana wrapper
        st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)
        
        # Navigasyon - Ortalanmış
        st.markdown('<div style="display: flex; justify-content: center; gap: 1rem; margin: 1rem 0;">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📊 Dashboard", key="home_btn", use_container_width=True):
                st.session_state.current_page = 'Ana-Sayfa'
                st.rerun()
        with col2:
            if st.button("✨ Parla", key="qa_btn", use_container_width=True):
                st.session_state.current_page = 'Soru-Cevap'
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Ana içerik
        if st.session_state.current_page == 'Ana-Sayfa':
            # Başlık
            st.markdown('<h1 class="dashboard-title">Dünya Mutluluk Analizi</h1>', unsafe_allow_html=True)
            
            # Filtre Bar - Ortalanmış
            with st.container():
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    years = sorted(df['year'].unique())
                    selected_year = st.selectbox('Yıl Seçin', ['Tümü'] + list(years), index=0)
                
                with col2:
                    regions = sorted(df['regional_indicator'].unique())
                    selected_region = st.selectbox('Bölge Seçin', ['Tümü'] + list(regions))
                
                st.markdown('</div>', unsafe_allow_html=True)

            # Tab Sistemi - Ortalanmış
            tab1, tab2, tab3 = st.tabs(["🌍 Genel Bakış", "📈 Trend Analizi", "🔍 Faktör Analizi"])
            
            with tab1:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">Dünya Mutluluk Haritası</h3>', unsafe_allow_html=True)
                
                # Harita verilerini hazırla
                if selected_year != 'Tümü':
                    map_data = df[df['year'] == selected_year].copy()
                    year_text = str(selected_year)
                else:
                    # Tüm yılların ortalamasını al
                    map_data = df.groupby('country_name')['life_ladder'].mean().reset_index()
                    year_text = "Tüm Yıllar"

                # Ülke isimlerini harita için uygun formata dönüştür
                country_name_mapping = {
                    'Turkiye': 'Turkey',
                    'United States': 'United States of America',
                    'Congo (Brazzaville)': 'Republic of Congo',
                    'Congo (Kinshasa)': 'Democratic Republic of the Congo',
                    'Palestinian Territories': 'Palestine',
                    'Taiwan Province of China': 'Taiwan',
                    'Hong Kong S.A.R. of China': 'Hong Kong',
                    'Czechia': 'Czech Republic',
                    'North Macedonia': 'Macedonia',
                    'Eswatini': 'Swaziland'
                }
                
                map_data['country_name'] = map_data['country_name'].replace(country_name_mapping)

                # Grafik renk paleti ve tema ayarları
                CHART_THEME = {
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'font': {'color': '#FFFFFF'},
                    'title': {
                        'font': {'size': 24, 'color': '#FFFFFF'},
                        'x': 0.5,
                        'y': 0.95,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    'xaxis': {
                        'gridcolor': 'rgba(255,255,255,0.1)',
                        'zerolinecolor': 'rgba(255,255,255,0.2)',
                        'titlefont': {'color': '#FFFFFF'},
                        'tickfont': {'color': '#FFFFFF'}
                    },
                    'yaxis': {
                        'gridcolor': 'rgba(255,255,255,0.1)',
                        'zerolinecolor': 'rgba(255,255,255,0.2)',
                        'titlefont': {'color': '#FFFFFF'},
                        'tickfont': {'color': '#FFFFFF'}
                    }
                }

                # Harita görselleştirmesi için renk paleti
                MAP_COLOR_SCALE = [
                    [0, 'rgba(255,0,0,0.8)'],     # En düşük - Kırmızı
                    [0.5, 'rgba(255,165,0,0.8)'], # Orta - Turuncu
                    [1, 'rgba(0,255,127,0.8)']    # En yüksek - Yeşil
                ]

                # Harita görselleştirmesi
                fig = go.Figure(data=go.Choropleth(
                    locations=map_data['country_name'],
                    locationmode='country names',
                    z=map_data['life_ladder'],
                    text=map_data['country_name'],
                    colorscale=MAP_COLOR_SCALE,
                    colorbar_title="Mutluluk<br>Skoru",
                    hovertemplate='<b>%{text}</b><br>Mutluluk Skoru: %{z:.2f}<extra></extra>'
                ))

                # Harita düzeni
                fig.update_layout(
                    **CHART_THEME,
                    geo=dict(
                        showframe=False,
                        showcoastlines=True,
                        projection_type='equirectangular',
                        coastlinecolor='rgba(255, 255, 255, 0.3)',
                        showland=True,
                        landcolor='rgba(255, 255, 255, 0.05)',
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    height=600,
                    margin=dict(l=0, r=0, t=50, b=0)
                )

                # Başlığı ayrıca güncelle
                fig.update_layout(
                    title_text=f"Dünya Mutluluk Haritası ({year_text})"
                )

                # Haritayı göster
                st.plotly_chart(fig, use_container_width=True)

                # Metrik kartları için container
                st.markdown('<div class="dashboard-metrics">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                        <div class="metric-container">
                            <div class="metric-value">{}</div>
                            <div class="metric-label">En Mutlu Ülke</div>
                            <div class="metric-score">{:.2f}</div>
                        </div>
                    """.format(
                        map_data.nlargest(1, 'life_ladder')['country_name'].iloc[0],
                        map_data.nlargest(1, 'life_ladder')['life_ladder'].iloc[0]
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class="metric-container">
                            <div class="metric-value">{:.2f}</div>
                            <div class="metric-label">Global Ortalama</div>
                            <div class="metric-score">±{:.2f} std</div>
                        </div>
                    """.format(
                        map_data['life_ladder'].mean(),
                        map_data['life_ladder'].std()
                    ), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                        <div class="metric-container">
                            <div class="metric-value">{}</div>
                            <div class="metric-label">En Mutsuz Ülke</div>
                            <div class="metric-score">{:.2f}</div>
                        </div>
                    """.format(
                        map_data.nsmallest(1, 'life_ladder')['country_name'].iloc[0],
                        map_data.nsmallest(1, 'life_ladder')['life_ladder'].iloc[0]
                    ), unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

                # Bölümler arası boşluk
                st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

                # Bölgesel Mutluluk Ortalamaları
                st.markdown("""
                    <div class="chart-container">
                        <h3 class="section-title">🌍 Bölgesel Mutluluk Ortalamaları</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Bölgesel ortalamaları hesapla
                if selected_year != 'Tümü':
                    regional_avg = df[df['year'] == selected_year].groupby('regional_indicator')['life_ladder'].mean().reset_index()
                    year_text = str(selected_year)
                else:
                    regional_avg = df.groupby('regional_indicator')['life_ladder'].mean().reset_index()
                    year_text = "Tüm Yıllar"
                
                # Ortalamalara göre sırala
                regional_avg = regional_avg.sort_values('life_ladder', ascending=False)
                
                # Bar chart oluştur
                fig_regional = go.Figure()

                # Renk skalası
                happiness_colors = [
                    [0, 'rgba(255,0,0,0.8)'],     # Kırmızı (en mutsuz)
                    [0.5, 'rgba(255,165,0,0.8)'], # Turuncu (orta)
                    [1, 'rgba(0,255,127,0.8)']    # Yeşil (en mutlu)
                ]
                
                fig_regional.add_trace(go.Bar(
                    y=regional_avg['regional_indicator'],
                    x=regional_avg['life_ladder'],
                    orientation='h',
                    marker=dict(
                        color=regional_avg['life_ladder'],
                        colorscale=happiness_colors
                    ),
                    text=regional_avg['life_ladder'].round(2),
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Mutluluk Skoru: %{x:.2f}<extra></extra>'
                ))
                
                # Bölgesel trend grafiği düzeni
                fig_regional.update_layout(
                    **CHART_THEME,
                    xaxis_title="Yıl",
                    yaxis_title="Mutluluk Skoru",
                    hovermode='x unified',
                    height=600,
                    showlegend=True,
                    title={
                        'text': "Bölgesel Mutluluk Trendleri",
                        'y': 0.95,  # Title'ı biraz yukarı taşı
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 24}
                    },
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.15,  # Legend'i title'ın üzerine taşı
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(255,255,255,0.2)",
                        font=dict(size=10)
                    ),
                    margin=dict(t=150)  # Üst marjini arttır
                )

                st.plotly_chart(fig_regional, use_container_width=True)
                
                # En mutlu/mutsuz bölge metriklerini göster
                col1, col2 = st.columns(2)
                
                with col1:
                    happiest_region = regional_avg.iloc[0]  # En yüksek skor (ascending=False olduğu için ilk eleman)
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">{happiest_region['regional_indicator']}</div>
                            <div class="metric-label">En Mutlu Bölge</div>
                            <div class="metric-score">Ortalama Skor: {happiest_region['life_ladder']:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    unhappiest_region = regional_avg.iloc[-1]  # En düşük skor (ascending=False olduğu için son eleman)
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-value">{unhappiest_region['regional_indicator']}</div>
                            <div class="metric-label">En Mutsuz Bölge</div>
                            <div class="metric-score">Ortalama Skor: {unhappiest_region['life_ladder']:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)

                # En mutlu 10 ülke grafiği
                st.markdown("""
                    <div class="chart-container">
                        <h3 class="section-title">🏆 En Mutlu 10 Ülke</h3>
                    </div>
                """, unsafe_allow_html=True)

                # En mutlu 10 ülkeyi seç
                if selected_year != 'Tümü':
                    top_10 = df[df['year'] == selected_year].nlargest(10, 'life_ladder')
                else:
                    top_10 = df.groupby('country_name')['life_ladder'].mean().nlargest(10).reset_index()

                # Mutluluk skoruna göre azalan sırada sırala (en mutlu en üstte olacak)
                top_10 = top_10.sort_values('life_ladder', ascending=False)

                # Top 10 grafiği
                fig_top = go.Figure()
                
                # En mutlu 10 ülke için yeşil tonları
                happy_color_scale = [
                    [0, 'rgba(0,100,0,0.8)'],     # Koyu yeşil
                    [0.5, 'rgba(0,180,0,0.8)'],   # Orta yeşil
                    [1, 'rgba(0,255,127,0.8)']    # Parlak yeşil
                ]
                
                fig_top.add_trace(go.Bar(
                    y=top_10['country_name'],
                    x=top_10['life_ladder'],
                    orientation='h',
                    marker=dict(
                        color=top_10['life_ladder'],
                        colorscale=happy_color_scale,
                        showscale=False
                    ),
                    text=top_10['life_ladder'].round(2),
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Mutluluk Skoru: %{x:.2f}<extra></extra>'
                ))

                # Top 10 grafik düzeni
                fig_top.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#FFFFFF'},
                    title={
                        'text': f"En Mutlu 10 Ülke ({year_text})",
                        'font': {'size': 24, 'color': '#FFFFFF'},
                        'x': 0.5,
                        'y': 0.95,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    xaxis={
                        'gridcolor': 'rgba(255,255,255,0.1)',
                        'zerolinecolor': 'rgba(255,255,255,0.2)',
                        'titlefont': {'color': '#FFFFFF'},
                        'tickfont': {'color': '#FFFFFF'},
                        'title': 'Mutluluk Skoru'
                    },
                    yaxis={
                        'gridcolor': 'rgba(255,255,255,0.1)',
                        'zerolinecolor': 'rgba(255,255,255,0.2)',
                        'titlefont': {'color': '#FFFFFF'},
                        'tickfont': {'color': '#FFFFFF'},
                        'autorange': 'reversed'  # En mutlu ülkeyi en üstte göstermek için
                    },
                    height=400,
                    margin=dict(l=0, r=0, t=50, b=0),
                    showlegend=False
                )

                st.plotly_chart(fig_top, use_container_width=True)

                # En mutsuz 10 ülke grafiği
                st.markdown("""
                    <div class="chart-container">
                        <h3 class="section-title">📉 En Mutsuz 10 Ülke</h3>
                    </div>
                """, unsafe_allow_html=True)

                # En mutsuz 10 ülkeyi seç
                if selected_year != 'Tümü':
                    bottom_10 = df[df['year'] == selected_year].nsmallest(10, 'life_ladder')
                else:
                    bottom_10 = df.groupby('country_name')['life_ladder'].mean().nsmallest(10).reset_index()

                # Mutluluk skoruna göre artan sırada sırala (en mutsuz en üstte olacak)
                bottom_10 = bottom_10.sort_values('life_ladder', ascending=True)

                # Bottom 10 grafiği
                fig_bottom = go.Figure()
                
                # En mutsuz 10 ülke için kırmızı tonları
                unhappy_color_scale = [
                    [0, 'rgba(255,0,0,0.9)'],     # Koyu kırmızı
                    [0.5, 'rgba(255,80,80,0.8)'], # Orta kırmızı
                    [1, 'rgba(255,160,160,0.8)']  # Açık kırmızı
                ]
                
                fig_bottom.add_trace(go.Bar(
                    y=bottom_10['country_name'],
                    x=bottom_10['life_ladder'],
                    orientation='h',
                    marker=dict(
                        color=bottom_10['life_ladder'],
                        colorscale=unhappy_color_scale,
                        showscale=False
                    ),
                    text=bottom_10['life_ladder'].round(2),
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Mutluluk Skoru: %{x:.2f}<extra></extra>'
                ))

                # Bottom 10 grafik düzeni
                fig_bottom.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#FFFFFF'},
                    title={
                        'text': f"En Mutsuz 10 Ülke ({year_text})",
                        'font': {'size': 24, 'color': '#FFFFFF'},
                        'x': 0.5,
                        'y': 0.95,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    xaxis={
                        'gridcolor': 'rgba(255,255,255,0.1)',
                        'zerolinecolor': 'rgba(255,255,255,0.2)',
                        'titlefont': {'color': '#FFFFFF'},
                        'tickfont': {'color': '#FFFFFF'},
                        'title': 'Mutluluk Skoru'
                    },
                    yaxis={
                        'gridcolor': 'rgba(255,255,255,0.1)',
                        'zerolinecolor': 'rgba(255,255,255,0.2)',
                        'titlefont': {'color': '#FFFFFF'},
                        'tickfont': {'color': '#FFFFFF'},
                        'autorange': 'reversed'  # En mutsuz ülkeyi en üstte göstermek için
                    },
                    height=400,
                    margin=dict(l=0, r=0, t=50, b=0),
                    showlegend=False
                )

                st.plotly_chart(fig_bottom, use_container_width=True)
            
            with tab2:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                
                
                
                # Yıllara göre global ortalama
                global_trend = df.groupby('year')['life_ladder'].agg(['mean', 'std']).reset_index()
                
                fig_global = go.Figure()
                
                # Ortalama çizgisi
                fig_global.add_trace(go.Scatter(
                    x=global_trend['year'],
                    y=global_trend['mean'],
                    mode='lines+markers',
                    name='Global Ortalama',
                    line=dict(color='#00FFE7', width=3),
                    marker=dict(size=8, color='#007AFF')
                ))
                
                # Standart sapma aralığı
                fig_global.add_trace(go.Scatter(
                    x=global_trend['year'],
                    y=global_trend['mean'] + global_trend['std'],
                    mode='lines',
                    name='Standart Sapma',
                    line=dict(color='rgba(170, 0, 255, 0.2)', width=0),
                    showlegend=False
                ))
                
                fig_global.add_trace(go.Scatter(
                    x=global_trend['year'],
                    y=global_trend['mean'] - global_trend['std'],
                    mode='lines',
                    name='Standart Sapma',
                    line=dict(color='rgba(170, 0, 255, 0.2)', width=0),
                    fill='tonexty'
                ))
                
                # Global trend grafiği düzeni
                fig_global.update_layout(
                    **CHART_THEME,
                    xaxis_title="Yıl",
                    yaxis_title="Mutluluk Skoru",
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(255,255,255,0.2)"
                    )
                )
                
                # Başlığı ayrıca güncelle
                fig_global.update_layout(
                    title_text="Global Mutluluk Trendi ve Değişkenlik"
                )

                st.plotly_chart(fig_global, use_container_width=True)
                
                
                
                # Global trend istatistikleri
                total_change = global_trend['mean'].iloc[-1] - global_trend['mean'].iloc[0]
                avg_change = total_change / (len(global_trend) - 1)
                volatility = global_trend['std'].mean()
                
                # Metrik kartları için özel stil
                metric_style = """
                    <div style="
                        background: linear-gradient(135deg, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0.1) 100%);
                        border: 1px solid rgba(255,255,255,0.1);
                        border-radius: 16px;
                        padding: 1.5rem;
                        height: 180px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        transition: all 0.3s ease;
                        backdrop-filter: blur(10px);
                        -webkit-backdrop-filter: blur(10px);
                    "
                    onmouseover="this.style.transform='translateY(-5px)'; this.style.borderColor='rgba(0,198,255,0.5)'"
                    onmouseout="this.style.transform='none'; this.style.borderColor='rgba(255,255,255,0.1)'"
                    >
                        <div style="
                            font-size: 1.1rem;
                            font-weight: 500;
                            color: rgba(255,255,255,0.9);
                            margin-bottom: 1rem;
                            text-align: center;
                        ">{title}</div>
                        <div style="
                            font-size: 2.5rem;
                            font-weight: 700;
                            background: linear-gradient(135deg, #00FFE7 0%, #007AFF 50%, #AA00FF 100%);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            margin-bottom: 0.75rem;
                            text-align: center;
                        ">{value}</div>
                        <div style="
                            color: rgba(255,255,255,0.6);
                            font-size: 0.9rem;
                            text-align: center;
                        ">{description}</div>
                    </div>
                """
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(
                        metric_style.format(
                            title="Toplam Değişim",
                            value="{:+.2f}".format(total_change),
                            description="2005'ten günümüze"
                        ),
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        metric_style.format(
                            title="Yıllık Ortalama Değişim",
                            value="{:+.2f}".format(avg_change),
                            description="Her yıl için"
                        ),
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.markdown(
                        metric_style.format(
                            title="Ortalama Değişkenlik",
                            value="{:.2f}".format(volatility),
                            description="Standart sapma"
                        ),
                        unsafe_allow_html=True
                    )
                
                # Bölgesel trend analizi
               
                
                # Bölgelere göre yıllık ortalamalar
                regional_trend = df.groupby(['year', 'regional_indicator'])['life_ladder'].mean().reset_index()
                
                fig_regional = go.Figure()
                
                # Her bölge için farklı renk
                region_colors = {
                    region: f'rgba({int(170 * (1-i/len(regional_trend["regional_indicator"].unique())))}, '
                           f'{int(0 + (255 * i/len(regional_trend["regional_indicator"].unique())))}, '
                           f'{int(255 * (i/len(regional_trend["regional_indicator"].unique())))}, 0.8)'
                    for i, region in enumerate(regional_trend['regional_indicator'].unique())
                }
                
                for region in regional_trend['regional_indicator'].unique():
                    region_data = regional_trend[regional_trend['regional_indicator'] == region]
                    fig_regional.add_trace(go.Scatter(
                        x=region_data['year'],
                        y=region_data['life_ladder'],
                        mode='lines+markers',
                        name=region,
                        line=dict(color=region_colors[region], width=2),
                        marker=dict(size=6)
                    ))
                
                # Bölgesel trend grafiği düzeni
                fig_regional.update_layout(
                    **CHART_THEME,
                    xaxis_title="Yıl",
                    yaxis_title="Mutluluk Skoru",
                    hovermode='x unified',
                    height=600,
                    showlegend=True,
                    title={
                        'text': "Bölgesel Mutluluk Trendleri",
                        'y': 0.95,  # Title'ı biraz yukarı taşı
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 24}
                    },
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.15,  # Legend'i title'ın üzerine taşı
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(255,255,255,0.2)",
                        font=dict(size=10)
                    ),
                    margin=dict(t=150)  # Üst marjini arttır
                )
                
                st.plotly_chart(fig_regional, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                
                # Korelasyon analizi
                st.markdown("""
                    <div class="chart-container">
                        <h3 class="section-title">🔄 Faktörler Arası Korelasyon</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Faktör listesi
                factors = ['life_ladder', 'gdp_per_capita', 'social_support', 
                          'freedom_to_make_life_choices', 'internet_users_percent',
                          'life_expectancy']
                
                # Faktör isimleri
                factor_names = {
                    'life_ladder': 'Mutluluk',
                    'gdp_per_capita': 'GDP',
                    'social_support': 'Sosyal Destek',
                    'freedom_to_make_life_choices': 'Özgürlük',
                    'internet_users_percent': 'İnternet Kullanımı',
                    'life_expectancy': 'Yaşam Beklentisi'
                }
                
                # Korelasyon matrisi
                corr_matrix = df[factors].corr()
                
                # Heatmap
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=[factor_names[f] for f in factors],
                    y=[factor_names[f] for f in factors],
                    colorscale=MAP_COLOR_SCALE,
                    zmid=0,
                    text=np.round(corr_matrix, 2),
                    texttemplate='%{text}',
                    textfont={"size": 12, "color": "white"},
                    hoverongaps=False
                ))
                
                # Korelasyon heatmap düzeni
                fig_corr.update_layout(
                    **CHART_THEME,
                    height=500,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                # Başlığı ayrıca güncelle
                fig_corr.update_layout(
                    title_text="Faktörler Arası Korelasyon"
                )

                st.plotly_chart(fig_corr, use_container_width=True)

                # Faktör etki analizi
                st.markdown("""
                    <div class="chart-container">
                        <h3 class="section-title">📊 Faktörlerin Mutluluk Üzerindeki Etkisi</h3>
                    </div>
                """, unsafe_allow_html=True)

                # Scatter plot için faktörler
                scatter_factors = ['gdp_per_capita', 'internet_users_percent', 'freedom_to_make_life_choices']
                
                for factor in scatter_factors:
                    # Scatter plot container
                    st.markdown(f"""
                        <div class="chart-container">
                            <h4 class="section-subtitle">{factor_names[factor]} ve Mutluluk İlişkisi</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Scatter plot
                    fig_scatter = go.Figure()
                    
                    # Ana scatter plot
                    fig_scatter.add_trace(go.Scatter(
                        x=df[factor],
                        y=df['life_ladder'],
                        mode='markers',
                        name='Ülkeler',
                        marker=dict(
                            color=df[factor],
                            colorscale=MAP_COLOR_SCALE,
                            size=8,
                            opacity=0.6,
                            showscale=True,
                            colorbar=dict(
                                title=factor_names[factor],
                                titleside="right"
                            )
                        ),
                        hovertemplate=
                        '<b>%{text}</b><br>' +
                        f'{factor_names[factor]}: %{{x:.2f}}<br>' +
                        'Mutluluk: %{y:.2f}<extra></extra>',
                        text=df['country_name']
                    ))
                    
                    # Trend çizgisi
                    slope, intercept, r_value, p_value, std_err = stats.linregress(df[factor], df['life_ladder'])
                    line_x = np.array([df[factor].min(), df[factor].max()])
                    line_y = slope * line_x + intercept
                    
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_y,
                            mode='lines',
                            name='Trend',
                            line=dict(color='#AA00FF', width=2, dash='dash'),
                            hovertemplate=f'R² = {r_value**2:.3f}<extra></extra>'
                        )
                    )
                    
                    # Scatter plot düzeni
                    fig_scatter.update_layout(
                        **CHART_THEME,
                        height=400,
                        xaxis_title=factor_names[factor],
                        yaxis_title="Mutluluk Skoru",
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            bgcolor="rgba(0,0,0,0)",
                            bordercolor="rgba(255,255,255,0.2)"
                        )
                    )
                    
                    # Başlığı ayrıca güncelle
                    fig_scatter.update_layout(
                        title_text=f"{factor_names[factor]} ve Mutluluk İlişkisi"
                    )

                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Korelasyon metriği
                    correlation = df['life_ladder'].corr(df[factor])
                    st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">Korelasyon Katsayısı</div>
                            <div class="metric-value">{correlation:.3f}</div>
                            <div class="metric-score">R² = {r_value**2:.3f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        elif st.session_state.current_page == 'Soru-Cevap':
            # Soru-cevap bölümü container'ı
            st.markdown('<div class="qa-container">', unsafe_allow_html=True)
            
            
            # Info card
            st.markdown("""
                <div class="info-card">
                    <h3 style='margin: 0; font-size: 1.2rem; color: #00c6ff;'>🔮 Veriye Dayalı Zeka, İçgörüyle Güçleniyor</h3>
                    <p style='margin-top: 10px; color: rgba(255,255,255,0.8);'>
                        OECD, World Bank ve küresel kaynaklardan beslenen verilerle, anlamlı analizler ve etkili içgörüler.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Kullanıcı girişi
            question = st.text_area("Soru", 
                                  placeholder="Merak ettiğiniz konuyu buraya yazın... ✨", 
                                  key="question_input",
                                  height=150,
                                  label_visibility="collapsed")

            # Gönder butonu
            if st.button("GÖNDER", key="submit_button", use_container_width=True):
                if question:
                    # Önce agent tipini belirle
                    from llm_agents import MultiAgentSystem
                    multi_agent = MultiAgentSystem(df)
                    agent_type = multi_agent.route_question(question)
                    
                    # Agent tipi açıklamaları
                    agent_descriptions = {
                        "data": "📊 Veri Analizi Uzmanı",
                        "causal": "🔍 Nedensellik Analisti",
                        "qa": "💡 Genel Bilgi Uzmanı"
                    }
                    
                    # Agent yönlendirme bilgisini göster
                    st.markdown(f"""
                        <div style='
                            background: rgba(0, 198, 255, 0.05);
                            border: 1px solid rgba(0, 198, 255, 0.1);
                            border-radius: 8px;
                            padding: 12px;
                            margin-bottom: 15px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            text-align: center;
                        '>
                            <div>
                                <div style='color: #00c6ff; font-weight: 500; margin-bottom: 5px;'>
                                    🎯 Soru Yönlendiriliyor
                                </div>
                                <div style='color: rgba(255,255,255,0.8); font-size: 0.9em;'>
                                    {agent_descriptions.get(agent_type, "Bilinmeyen Agent")} ile yanıtlanacak
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Yükleniyor animasyonu
                    with st.spinner("💫 Yanıt hazırlanıyor..."):
                        answer = multi_agent.get_answer(question)
                        if answer:
                            st.markdown(f"""
                            <div class="chat-message">
                                <div style='color: #00c6ff; font-weight: 500; margin-bottom: 8px;'>🤖 Analiz Sonuçları</div>
                                <div style='color: rgba(255,255,255,0.9);'>
                                    {answer}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("🤔 Üzgünüm, yanıt oluşturulamadı. Lütfen tekrar deneyin.")
                else:
                    st.warning("💡 Lütfen bir soru sorun...")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Container'ları kapat
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        st.error("Lütfen sayfayı yenileyin veya daha sonra tekrar deneyin.")

if __name__ == "__main__":
    main() 
