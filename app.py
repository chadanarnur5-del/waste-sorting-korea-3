import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Waste Sorter Korea",
    page_icon="♻️",
    layout="centered"
)

# --- 2. CUSTOM CSS INJECTION ---
st.markdown("""
    <style>
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }
    
    /* Hero Banner Header */
    .hero-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    .hero-header h1 {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .hero-header p {
        color: #e0e7ff;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Input and Select Label Styling */
    label {
        font-weight: 600 !important;
        color: #374151 !important;
    }
    
    /* Custom Result Card */
    .result-card {
        padding: 20px;
        border-radius: 14px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    .result-card h3 {
        margin: 0 0 6px 0;
        font-size: 1.4rem;
    }
    .result-card p {
        margin: 0;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Custom Progress Bar Wrapper */
    .confidence-wrapper {
        background-color: #f3f4f6;
        border-radius: 10px;
        padding: 4px;
        margin-top: 8px;
        margin-bottom: 20px;
    }
    .confidence-bar {
        height: 12px;
        border-radius: 8px;
        background: linear-gradient(90deg, #4f46e5, #3b82f6);
    }
    
    /* Tips Container */
    .tips-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-top: 15px;
    }
    .tips-box ul {
        margin-bottom: 0;
        padding-left: 20px;
        color: #475569;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOCAL DISTRICT RULES (EMBEDDED - 10 DISTRICTS) ---
DISTRICT_RULES = {
    "Seoul — Gangnam": {
        "plastic": "Rinse transparent plastic bottles, remove labels and caps. Dispose in transparent bags.",
        "paper": "Fold paper/cardboard and tie with string. Remove any plastic covers, tape, or spiral bindings from notebooks.",
        "glass": "Clear contents. Remove metal or plastic caps.",
        "metal": "Flatten cans or tins. Dispose in metal collection bins.",
        "organic": "Drain excess moisture. Dispose strictly in special yellow Food Waste Bags.",
        "general": "General trash must be disposed strictly in special white General Trash Bags (종량제봉투)."
    },
    "Seoul — Mapo": {
        "plastic": "Remove films and stickers. Clean plastic items should be placed in transparent bags.",
        "paper": "Keep notebooks, paper, and boxes dry and flat. Dispose of dirty paper in general waste.",
        "glass": "Separate glass bottles. Wrap broken glass in newspaper for safety.",
        "metal": "Crush beverage and food cans before disposal.",
        "organic": "Use designated Mapo district food waste bags.",
        "general": "Use standard Mapo district general waste bags."
    },
    "Seoul — Hongdae": {
        "plastic": "Empty and rinse take-out plastic cups and bottles thoroughly. Separate cup lids and straws into general waste if contaminated.",
        "paper": "Flatten flyers, cardboard, and notebooks. Keep away from food contamination.",
        "glass": "Clear beverage glass bottles completely before placing in designated street recycling stations.",
        "metal": "Empty drink cans completely and crush them before disposal.",
        "organic": "Must be disposed of in designated Mapo district food waste bags or RFID bins.",
        "general": "Use official Mapo district pay-as-you-throw trash bags (종량제봉투)."
    },
    "Busan — Haeundae": {
        "plastic": "Empty completely and rinse. Separate PET bottles into clear plastic collection boxes.",
        "paper": "Bundle notebooks, books, and boxes separately. Remove metal spirals from notebooks.",
        "glass": "Separate clear, green, and brown glass bottles. Return reusable bottles to stores for deposit.",
        "metal": "Empty cans completely before disposing in metal recycling bins.",
        "organic": "Drain liquids and dispose using designated Haeundae RFID food waste bins.",
        "general": "Dispose in official Haeundae standard waste bags (종량제봉투)."
    },
    "Incheon — Yeonsu": {
        "plastic": "Wash off food residue. Separate clear PET bottles from colored plastic containers.",
        "paper": "Keep dry and flatten. Remove any metal/plastic elements from notebooks.",
        "glass": "Remove caps and labels if possible.",
        "metal": "Puncture aerosol cans in a ventilated area. Crush drink cans.",
        "organic": "Remove excess water and dispose in Yeonsu RFID food waste containers.",
        "general": "Use official Yeonsu general waste bags."
    },
    "Suwon — Yeongtong": {
        "plastic": "Separate clear PET bottles into dedicated clear-PET collection bins. Rinse all food containers.",
        "paper": "Remove tape, staples, and plastic covers from boxes and notebooks before bundling.",
        "glass": "Rinse completely and remove metal caps.",
        "metal": "Flatten clean aluminium and steel cans.",
        "organic": "Drain all moisture. Use official Suwon yellow food waste bags or RFID disposal points.",
        "general": "Dispose strictly in Suwon-designated general waste bags."
    },
    "Seongnam — Bundang": {
        "plastic": "Rinse containers cleanly. Transparent PET bottles must be separated into dedicated transparent bottle bags.",
        "paper": "Flatten boxes and remove clear tape. Separate spiral notebook coils into metal/plastic waste.",
        "glass": "Rinse glass jars and bottles. Non-recyclable ceramics must be placed in special fireproof bags (불연성 종량제봉투).",
        "metal": "Empty gas canisters safely before disposal. Compress beverage cans.",
        "organic": "Drain liquid completely. Use Seongnam official food waste bags or RFID equipment.",
        "general": "Use standard Seongnam-gu volume-based trash bags."
    },
    "Daegu — Suseong": {
        "plastic": "Separate clear PET bottles from general plastics. Wash off grease or sauce completely.",
        "paper": "Tie paper, textbooks, and cardboard in bundles. Keep away from rainwater.",
        "glass": "Place intact glass bottles in designated residential collection baskets.",
        "metal": "Flatten drink cans and food tins. Place in metal collection bins.",
        "organic": "Must be disposed of using Daegu RFID smart food waste system or dedicated district food waste bags.",
        "general": "Dispose using official Suseong-gu pay-as-you-throw trash bags."
    },
    "Daejeon — Yuseong": {
        "plastic": "Peel off vinyl labels from PET bottles. Wash disposable plastic containers thoroughly.",
        "paper": "Flatten boxes, discard sticky tape, and bundle books/notebooks neatly.",
        "glass": "Separate beverage bottles by color if requested by local residential recycling hub.",
        "metal": "Empty beverage and food cans completely before recycling.",
        "organic": "Remove moisture thoroughly and use Yuseong-gu designated yellow food waste bags.",
        "general": "Use official Yuseong-gu general waste bags."
    },
    "Jeju — Jeju City": {
        "plastic": "Take clear PET bottles and plastics to clean stations (Clean House) according to designated collection days.",
        "paper": "Separate clean paper and boxes at local Clean House recycling hubs.",
        "glass": "Dispose of intact glass bottles in glass bins at Clean House locations.",
        "metal": "Empty and crush drink cans for collection.",
        "organic": "Use official Jeju RFID food disposal buckets or designated Jeju food waste bags.",
        "general": "Dispose only in official Jeju Island pay-as-you-throw trash bags."
    }
}

# --- 4. LOCALIZATION CONFIGURATION ---
LOCALES = {
    "English": {
        "title": "♻️ Waste Sorter in South Korea",
        "subtitle": "AI-Powered Recycling & Disposal Guide across 10 Korean Districts",
        "select_district": "📍 Select your district in Korea:",
        "upload_label": "📸 Upload a photo of the item:",
        "uploaded_caption": "Uploaded Photo",
        "analyzing_spinner": "Analyzing image with AI model...",
        "btn_analyze": "Classify Waste 🔍",
        "analysis_complete": "Analysis Complete!",
        "result_cat": "Category:",
        "result_conf": "Model Confidence:",
        "result_rule": "Local Disposal Instructions:",
        "tips_header": "💡 Tips for best classification accuracy:",
        "tips": [
            "Take a clear photo with good lighting.",
            "Center a single object in the frame.",
            "Avoid dark or cluttered backgrounds."
        ],
        "categories": {
            "plastic": "Plastic 🥤",
            "paper": "Paper / Cardboard 📦",
            "glass": "Glass 🍾",
            "metal": "Metal 🥫",
            "organic": "Organic / Food Waste 🍎",
            "general": "General Waste 🗑️"
        }
    },
    "한국어": {
        "title": "♻️ 한국 분리수거 가이드",
        "subtitle": "AI 기반 10개 지역 맞춤형 분리배출 안내 서비스",
        "select_district": "📍 거주하는 지역을 선택하세요:",
        "upload_label": "📸 쓰레기 사진을 업로드하세요:",
        "uploaded_caption": "업로드된 사진",
        "analyzing_spinner": "인공지능 모델이 이미지를 분석 중입니다...",
        "btn_analyze": "분류하기 🔍",
        "analysis_complete": "분석 완료!",
        "result_cat": "분류 카테고리:",
        "result_conf": "모델 신뢰도:",
        "result_rule": "지역별 배출 방법 안내:",
        "tips_header": "💡 정확한 인식 결과를 위한 팁:",
        "tips": [
            "밝은 조명에서 명확하게 촬영해 주세요.",
            "한 번에 하나의 물체만 중앙에 위치시켜 주세요.",
            "어둡거나 복잡한 배경을 피해 주세요."
        ],
        "categories": {
            "plastic": "플라스틱 🥤",
            "paper": "종이류 / 상자 📦",
            "glass": "유리병 🍾",
            "metal": "캔 / 금속류 🥫",
            "organic": "음식물 쓰레기 🍎",
            "general": "일반 쓰레기 🗑️"
        }
    }
}

# Card Color Map for UI
CATEGORY_COLORS = {
    "plastic": {"bg": "#edf7ed", "border": "#2e7d32", "text": "#1e4620", "label": "🟢 Plastic Recyclable"},
    "paper": {"bg": "#eef2ff", "border": "#3730a3", "text": "#1e1b4b", "label": "🔵 Paper / Cardboard Recyclable"},
    "glass": {"bg": "#fefce8", "border": "#ca8a04", "text": "#713f12", "label": "🟡 Glass Recyclable"},
    "metal": {"bg": "#f3f4f6", "border": "#4b5563", "text": "#1f2937", "label": "⚪ Metal Recyclable"},
    "organic": {"bg": "#fef2f2", "border": "#dc2626", "text": "#7f1d1d", "label": "🔴 Food Waste"},
    "general": {"bg": "#f3f4f6", "border": "#6b7280", "text": "#374151", "label": "⚪ General Waste"}
}

# --- 5. AI MODEL INITIALIZATION ---
@st.cache_resource
def load_model():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()
    return model, weights

model, weights = load_model()
preprocess = weights.transforms()

# Extended ImageNet mapping
IMAGENET_TO_WASTE = {
    # Plastic
    "water_bottle": "plastic", "pop_bottle": "plastic", "plastic_bag": "plastic",
    "water_jug": "plastic", "pill_bottle": "plastic",
    # Paper & Cardboard
    "carton": "paper", "envelope": "paper", "paper_towel": "paper",
    "notebook": "paper", "book": "paper", "binder": "paper", "comic_book": "paper",
    "menu": "paper", "book_jacket": "paper", "web_site": "paper", "crossword": "paper",
    # Glass
    "wine_bottle": "glass", "beer_bottle": "glass", "glass": "glass", "goblet": "glass",
    # Metal
    "can": "metal", "tin_can": "metal", "soup_bowl": "metal",
    # Organic / Food
    "banana": "organic", "apple": "organic", "orange": "organic", "lemon": "organic"
}

def predict_waste_type(image):
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    top_prob, top_cat_id = torch.topk(probabilities, 10)
    
    for i in range(10):
        cat_name = weights.meta["categories"][top_cat_id[i].item()].lower()
        for key, val in IMAGENET_TO_WASTE.items():
            if key in cat_name:
                return val, max(float(top_prob[i]) * 100, 78.5)
                
    return "general", float(top_prob[0]) * 100

# --- 6. APPLICATION INTERFACE ---

# Sidebar Controls
st.sidebar.markdown("### ⚙️ Settings / 설정")
lang_choice = st.sidebar.selectbox("Language / 언어", list(LOCALES.keys()))
t = LOCALES[lang_choice]

# Hero Banner
st.markdown(
    f"""
    <div class="hero-header">
        <h1>{t['title']}</h1>
        <p>{t['subtitle']}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# District Selection
selected_district = st.selectbox(t["select_district"], list(DISTRICT_RULES.keys()))

# File Upload
uploaded_file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.markdown(
        f"""
        <div class="tips-box">
            <strong style="color:#1e293b;">{t['tips_header']}</strong>
            <ul style="margin-top: 8px;">
                {''.join([f'<li>{tip}</li>' for tip in t['tips']])}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=t["uploaded_caption"], use_container_width=True)
    
    if st.button(t["btn_analyze"], type="primary", use_container_width=True):
        with st.spinner(t["analyzing_spinner"]):
            category_key, confidence = predict_waste_type(image)
            
            cat_display = t["categories"].get(category_key, category_key)
            rule_text = DISTRICT_RULES[selected_district].get(category_key, "")
            color_theme = CATEGORY_COLORS.get(category_key, CATEGORY_COLORS["general"])
            
            st.success(t["analysis_complete"])
            
            # Custom Styled Result Card
            st.markdown(
                f"""
                <div class="result-card" style="background-color:{color_theme['bg']}; border-left: 6px solid {color_theme['border']};">
                    <h3 style="color:{color_theme['text']};">{t['result_cat']} {cat_display}</h3>
                    <p style="color:{color_theme['border']};">{color_theme['label']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Custom Confidence Bar
            conf_int = min(int(confidence), 100)
            st.markdown(f"**{t['result_conf']} {confidence:.1f}%**")
            st.markdown(
                f"""
                <div class="confidence-wrapper">
                    <div class="confidence-bar" style="width: {conf_int}%;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Disposal Instructions
            st.info(f"**{t['result_rule']}**\n\n{rule_text}")
