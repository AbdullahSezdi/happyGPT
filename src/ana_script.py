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
        df = pd.read_csv('cleaned_dataset.csv')
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
        layout="centered"  # wide yerine centered kullanıyoruz
    )

    # Özel CSS Stilleri
    st.markdown("""
    <style>
        /* Streamlit container override */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* Streamlit varsayılan margin override */
        .main > div {
            padding: 0;
            margin: 0;
        }

        /* Ana container stil */
        .qa-container {
            width: 600px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Üst navigasyon butonları */
        .nav-buttons {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            justify-content: center;
        }

        .nav-button {
            background: rgba(0, 198, 255, 0.1);
            border: 1px solid rgba(0, 198, 255, 0.2);
            padding: 0.5rem 2rem;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .nav-button:hover {
            background: rgba(0, 198, 255, 0.2);
            transform: translateY(-2px);
        }

        /* Input alanı */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.03) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            padding: 15px !important;
            color: white !important;
            height: 120px !important;
            transition: all 0.3s ease !important;
            font-size: 1rem !important;
            line-height: 1.5 !important;
            resize: none !important;
        }

        .stTextArea textarea:hover {
            border-color: rgba(0, 198, 255, 0.3) !important;
            background: rgba(255, 255, 255, 0.05) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        }

        .stTextArea textarea:focus {
            border-color: #00c6ff !important;
            background: rgba(255, 255, 255, 0.07) !important;
            box-shadow: 0 4px 12px rgba(0, 198, 255, 0.1) !important;
            outline: none !important;
        }

        /* Gönder butonu */
        .stButton button {
            background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
            border: none !important;
            padding: 0.8rem 2rem !important;
            color: white !important;
            width: 100% !important;
            border-radius: 8px !important;
            margin-top: 1rem !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px !important;
            text-transform: uppercase !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }

        .stButton button:hover {
            background: linear-gradient(90deg, #0072ff, #00c6ff) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        }

        .stButton button:active {
            transform: translateY(0) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }

        /* Info card */
        .info-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 1rem;
        }

        /* Chat message */
        .chat-message {
            background: rgba(0, 198, 255, 0.05);
            border: 1px solid rgba(0, 198, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-top: 1rem;
        }

        @media (max-width: 768px) {
            .qa-container {
                width: 100%;
                padding: 10px;
            }
        }
    </style>
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

                # Harita görselleştirmesi
                fig = go.Figure(data=go.Choropleth(
                    locations=map_data['country_name'],
                    locationmode='country names',
                    z=map_data['life_ladder'],
                    text=map_data['country_name'],
                    colorscale=[
                        [0, 'rgb(255,50,50)'],     # Kırmızı (en düşük)
                        [0.5, 'rgb(255,255,200)'],  # Açık sarı (orta)
                        [1, 'rgb(50,150,50)']      # Yeşil (en yüksek)
                    ],
                    colorbar_title="Mutluluk<br>Skoru",
                    hovertemplate='<b>%{text}</b><br>Mutluluk Skoru: %{z:.2f}<extra></extra>'
                ))

                # Harita düzeni
                fig.update_layout(
                    title=dict(
                        text=f"Dünya Mutluluk Haritası ({year_text})",
                        x=0.5,
                        y=0.95,
                        xanchor='center',
                        yanchor='top',
                        font=dict(size=20, color='white')
                    ),
                    geo=dict(
                        showframe=False,
                        showcoastlines=True,
                        projection_type='equirectangular',
                        coastlinecolor='rgba(255, 255, 255, 0.5)',
                        showland=True,
                        landcolor='rgba(200, 200, 200, 0.1)',
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=600,
                    margin=dict(l=0, r=0, t=50, b=0)
                )

                # Haritayı göster
                st.plotly_chart(fig, use_container_width=True)

                # İstatistikler
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "En Mutlu Ülke",
                        f"{map_data.nlargest(1, 'life_ladder')['country_name'].iloc[0]}",
                        f"{map_data.nlargest(1, 'life_ladder')['life_ladder'].iloc[0]:.2f}"
                    )
                with col2:
                    st.metric(
                        "Global Ortalama",
                        f"{map_data['life_ladder'].mean():.2f}",
                        f"±{map_data['life_ladder'].std():.2f} std"
                    )
                with col3:
                    st.metric(
                        "En Mutsuz Ülke",
                        f"{map_data.nsmallest(1, 'life_ladder')['country_name'].iloc[0]}",
                        f"{map_data.nsmallest(1, 'life_ladder')['life_ladder'].iloc[0]:.2f}"
                    )

                # Bölümler arası boşluk
                st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

                # Bölgesel Mutluluk Ortalamaları
                st.markdown("### 🌍 Bölgesel Mutluluk Ortalamaları")
                
                # Bölgesel ortalamaları hesapla
                if selected_year != 'Tümü':
                    regional_avg = df[df['year'] == selected_year].groupby('regional_indicator')['life_ladder'].mean().reset_index()
                    year_text = str(selected_year)
                else:
                    regional_avg = df.groupby('regional_indicator')['life_ladder'].mean().reset_index()
                    year_text = "Tüm Yıllar"
                
                # Ortalamalara göre sırala
                regional_avg = regional_avg.sort_values('life_ladder', ascending=True)
                
                # Bar chart oluştur
                fig_regional = go.Figure()
                
                # Renk skalası oluştur
                colors = [
                    f'rgb({int(255 - (i * (255-50)/(len(regional_avg)-1)))}, '
                    f'{int(50 + (i * (150-50)/(len(regional_avg)-1)))}, 50)'
                    for i in range(len(regional_avg))
                ]
                
                fig_regional.add_trace(go.Bar(
                    y=regional_avg['regional_indicator'],
                    x=regional_avg['life_ladder'],
                    orientation='h',
                    marker_color=colors,
                    text=regional_avg['life_ladder'].round(2),
                    textposition='auto'
                ))
                
                fig_regional.update_layout(
                    title=dict(
                        text=f"Bölgesel Mutluluk Ortalamaları ({year_text})",
                        x=0.5,
                        y=0.95,
                        xanchor='center',
                        yanchor='top',
                        font=dict(size=20, color='white')
                    ),
                    xaxis_title="Mutluluk Skoru",
                    yaxis_title=None,
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(l=0, r=0, t=50, b=0),
                    showlegend=False,
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)'
                    ),
                    yaxis=dict(
                        showgrid=False
                    )
                )
                
                st.plotly_chart(fig_regional, use_container_width=True)
                
                # Bölgesel içgörüler
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **En Mutlu Bölge**: {regional_avg.iloc[-1]['regional_indicator']}
                    - Ortalama Skor: {regional_avg.iloc[-1]['life_ladder']:.2f}
                    """)
                with col2:
                    st.warning(f"""
                    **En Mutsuz Bölge**: {regional_avg.iloc[0]['regional_indicator']}
                    - Ortalama Skor: {regional_avg.iloc[0]['life_ladder']:.2f}
                    """)

                # Bölümler arası boşluk
                st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

                # En Mutlu ve En Mutsuz 10 Ülke
                st.markdown("### 🌟 En Mutlu ve En Mutsuz 10 Ülke")
                
                # Verileri hazırla
                if selected_year != 'Tümü':
                    top_10 = df[df['year'] == selected_year].nlargest(10, 'life_ladder')[['country_name', 'life_ladder', 'regional_indicator']]
                    bottom_10 = df[df['year'] == selected_year].nsmallest(10, 'life_ladder')[['country_name', 'life_ladder', 'regional_indicator']]
                    comparison_title = f"En Mutlu ve En Mutsuz 10 Ülke ({selected_year})"
                else:
                    # Tüm yılların ortalamasını al
                    avg_happiness = df.groupby('country_name')['life_ladder'].mean().reset_index()
                    # Bölge bilgisini ekle (en son yılın bölge bilgisini kullan)
                    latest_year = df['year'].max()
                    region_info = df[df['year'] == latest_year][['country_name', 'regional_indicator']].drop_duplicates()
                    avg_happiness = avg_happiness.merge(region_info, on='country_name')
                    
                    top_10 = avg_happiness.nlargest(10, 'life_ladder')[['country_name', 'life_ladder', 'regional_indicator']]
                    bottom_10 = avg_happiness.nsmallest(10, 'life_ladder')[['country_name', 'life_ladder', 'regional_indicator']]
                    comparison_title = "En Mutlu ve En Mutsuz 10 Ülke (Tüm Yılların Ortalaması)"

                # Görselleştirme için iki sütun oluştur
                col1, col2 = st.columns(2)
                
                with col1:
                    # En mutlu 10 ülke grafiği
                    fig_top = go.Figure()
                    fig_top.add_trace(go.Bar(
                        y=top_10['country_name'],
                        x=top_10['life_ladder'],
                        orientation='h',
                        marker_color='rgb(50,150,50)',  # Yeşil
                        text=top_10['life_ladder'].round(2),
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>Mutluluk Skoru: %{x:.2f}<extra></extra>'
                    ))
                    
                    fig_top.update_layout(
                        title=dict(
                            text="En Mutlu 10 Ülke",
                            x=0.5,
                            y=0.95,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=16, color='white')
                        ),
                        xaxis_title="Mutluluk Skoru",
                        yaxis_title=None,
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(l=0, r=0, t=50, b=0),
                        showlegend=False,
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.2)'
                        ),
                        yaxis=dict(
                            showgrid=False,
                            autorange="reversed"  # En yüksek değeri en üstte göster
                        )
                    )
                    st.plotly_chart(fig_top, use_container_width=True)
                
                with col2:
                    # En mutsuz 10 ülke grafiği
                    fig_bottom = go.Figure()
                    fig_bottom.add_trace(go.Bar(
                        y=bottom_10['country_name'],
                        x=bottom_10['life_ladder'],
                        orientation='h',
                        marker_color='rgb(255,50,50)',  # Kırmızı
                        text=bottom_10['life_ladder'].round(2),
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>Mutluluk Skoru: %{x:.2f}<extra></extra>'
                    ))
                    
                    fig_bottom.update_layout(
                        title=dict(
                            text="En Mutsuz 10 Ülke",
                            x=0.5,
                            y=0.95,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=16, color='white')
                        ),
                        xaxis_title="Mutluluk Skoru",
                        yaxis_title=None,
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(l=0, r=0, t=50, b=0),
                        showlegend=False,
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.2)'
                        ),
                        yaxis=dict(
                            showgrid=False
                        )
                    )
                    st.plotly_chart(fig_bottom, use_container_width=True)
                
                # Bölgesel dağılım analizi
                top_regions = top_10['regional_indicator'].value_counts()
                bottom_regions = bottom_10['regional_indicator'].value_counts()
                
                # İçgörüler
                st.markdown("#### 🔍 Önemli İçgörüler")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **En Mutlu 10 Ülke Analizi**:
                    - Ortalama Mutluluk: {top_10['life_ladder'].mean():.2f}
                    - En Yaygın Bölge: {top_regions.index[0]} ({top_regions.iloc[0]} ülke)
                    - En Yüksek Skor: {top_10['life_ladder'].max():.2f} ({top_10.iloc[0]['country_name']})
                    """)
                
                with col2:
                    st.warning(f"""
                    **En Mutsuz 10 Ülke Analizi**:
                    - Ortalama Mutluluk: {bottom_10['life_ladder'].mean():.2f}
                    - En Yaygın Bölge: {bottom_regions.index[0]} ({bottom_regions.iloc[0]} ülke)
                    - En Düşük Skor: {bottom_10['life_ladder'].min():.2f} ({bottom_10.iloc[-1]['country_name']})
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">Mutluluk Trend Analizi</h3>', unsafe_allow_html=True)
                
                # Global trend analizi
                st.markdown("### 📈 Global Mutluluk Trendi")
                
                # Yıllara göre global ortalama
                global_trend = df.groupby('year')['life_ladder'].agg(['mean', 'std']).reset_index()
                
                fig_global = go.Figure()
                
                # Ortalama çizgisi
                fig_global.add_trace(go.Scatter(
                    x=global_trend['year'],
                    y=global_trend['mean'],
                    mode='lines+markers',
                    name='Global Ortalama',
                    line=dict(color='#8dd3c7', width=3),
                    marker=dict(size=8)
                ))
                
                # Standart sapma aralığı
                fig_global.add_trace(go.Scatter(
                    x=global_trend['year'],
                    y=global_trend['mean'] + global_trend['std'],
                    mode='lines',
                    name='Standart Sapma',
                    line=dict(color='rgba(141, 211, 199, 0.2)', width=0),
                    showlegend=False
                ))
                
                fig_global.add_trace(go.Scatter(
                    x=global_trend['year'],
                    y=global_trend['mean'] - global_trend['std'],
                    mode='lines',
                    name='Standart Sapma',
                    line=dict(color='rgba(141, 211, 199, 0.2)', width=0),
                    fill='tonexty'
                ))
                
                fig_global.update_layout(
                    title="Global Mutluluk Trendi ve Değişkenlik",
                    xaxis_title="Yıl",
                    yaxis_title="Mutluluk Skoru",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_global, use_container_width=True)
                
                # Bölgesel trend analizi
                st.markdown("### 🌍 Bölgesel Mutluluk Trendleri")
                
                # Bölgelere göre yıllık ortalamalar
                regional_trend = df.groupby(['year', 'regional_indicator'])['life_ladder'].mean().reset_index()
                
                fig_regional = go.Figure()
                
                for region in regional_trend['regional_indicator'].unique():
                    region_data = regional_trend[regional_trend['regional_indicator'] == region]
                    fig_regional.add_trace(go.Scatter(
                        x=region_data['year'],
                        y=region_data['life_ladder'],
                        mode='lines+markers',
                        name=region,
                        marker=dict(size=6)
                    ))
                
                fig_regional.update_layout(
                    title="Bölgesel Mutluluk Trendleri",
                    xaxis_title="Yıl",
                    yaxis_title="Mutluluk Skoru",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    height=600
                )
                
                st.plotly_chart(fig_regional, use_container_width=True)
                
                # Trend analizi içgörüleri
                st.markdown("### 🔍 Trend Analizi İçgörüleri")
                
                # Global trend istatistikleri
                total_change = global_trend['mean'].iloc[-1] - global_trend['mean'].iloc[0]
                avg_change = total_change / (len(global_trend) - 1)
                volatility = global_trend['std'].mean()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Toplam Değişim",
                        f"{total_change:+.2f}",
                        "2005'ten günümüze"
                    )
                
                with col2:
                    st.metric(
                        "Yıllık Ortalama Değişim",
                        f"{avg_change:+.2f}",
                        "Her yıl için"
                    )
                
                with col3:
                    st.metric(
                        "Ortalama Değişkenlik",
                        f"{volatility:.2f}",
                        "Standart sapma"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">📊 Faktör Analizi Sekmesi</h3>', unsafe_allow_html=True)
                
                st.markdown("### 1. Mutluluk ile Faktörler Arasındaki Korelasyon (Heatmap)")
                
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
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig_corr.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=500,
                    margin=dict(l=50, r=50, t=30, b=50)
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)

                # 2. Faktörlerin Etkisi (Scatter Plots)
                st.markdown("### 2. Faktörlerin Etkisi (Scatter Plots)")

                # Scatter plot için faktörler
                scatter_factors = ['gdp_per_capita', 'internet_users_percent', 'freedom_to_make_life_choices']
                
                for factor in scatter_factors:
                    # Önce scatter plot'u oluştur (trend çizgisi olmadan)
                    fig_scatter = px.scatter(
                        df,
                        x=factor,
                        y='life_ladder',
                        title=f"{factor_names[factor]} vs Mutluluk",
                        labels={
                            factor: factor_names[factor],
                            'life_ladder': 'Mutluluk Skoru'
                        },
                        template="plotly_dark"
                    )
                    
                    # Scatter noktalarının stilini ayarla
                    fig_scatter.data[0].marker.update(
                        color='#FFA500',  # Turuncu renk
                        size=6,
                        opacity=0.6,
                        line=dict(color='#ffffff', width=1)
                    )
                    
                    # Trend çizgisini ayrı bir trace olarak ekle
                    slope, intercept, r_value, p_value, std_err = stats.linregress(df[factor], df['life_ladder'])
                    line_x = np.array([df[factor].min(), df[factor].max()])
                    line_y = slope * line_x + intercept
                    
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_y,
                            mode='lines',
                            name='Trend',
                            line=dict(color='#ff0000', width=5)  # Koyu kırmızı ve kalın çizgi
                        )
                    )
                    
                    fig_scatter.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(l=0, r=0, t=50, b=0)
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # 3. Gelir Seviyesine Göre Mutluluk Dağılımı (Boxplot)
                st.markdown("### 3. Gelir Seviyesine Göre Mutluluk Dağılımı (Boxplot)")
                
                # GDP'ye göre ülkeleri kategorilere ayır
                df['income_level'] = pd.qcut(df['gdp_per_capita'], 
                                          q=3, 
                                          labels=['Düşük Gelir', 'Orta Gelir', 'Yüksek Gelir'])
                
                fig_box = px.box(
                    df,
                    x='income_level',
                    y='life_ladder',
                    title='Gelir Seviyesine Göre Mutluluk Dağılımı',
                    labels={
                        'income_level': 'Gelir Seviyesi',
                        'life_ladder': 'Mutluluk Skoru'
                    },
                    template="plotly_dark"
                )
                
                fig_box.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
                
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

                    with st.spinner("💫 Yanıt hazırlanıyor..."):
                        answer = multi_agent.get_answer(question)
                        if answer:
                            st.markdown(f"""
                            <div class="chat-message">
                                <div style='color: #00c6ff; font-weight: 500; margin-bottom: 8px;'>🤖 AI Asistan</div>
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
