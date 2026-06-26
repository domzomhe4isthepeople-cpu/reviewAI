import streamlit as st
import pandas as pd
import time
import os                             
from dotenv import load_dotenv         
from apify_client import ApifyClient
from google import genai
from google.genai.errors import ClientError
import plotly.express as px  # <-- เพิ่ม Plotly สำหรับทำกราฟสวยๆ

# --- 1. SET UP CREDENTIALS ---
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

apify_client = ApifyClient(APIFY_TOKEN)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def get_reviews_from_maps(url, max_reviews=15, review_sort="newest"):
    """ฟังก์ชันดึงรีวิวเวอร์ชันอัปเกรด ดักจับทุกโครงสร้างข้อมูลเพื่อป้องกันตารางว่าง"""
    run_input = {
        "exportPlaceActorMetadata": True,    
        "includeReviews": True,            
        "language": "en",                  
        "maxReviews": max_reviews,                  
        "reviewsSort": review_sort,       
        "scrapeReviewerId": False,
        "startUrls": [{"url": url}]
    }
    
    run = apify_client.actor("compass/crawler-google-places").call(run_input=run_input)
    
    reviews_list = []
    place_name = "สถานที่ท่องเที่ยว"
    
    for item in apify_client.dataset(run.default_dataset_id).iterate_items():
        place_name = item.get("title") or item.get("name") or place_name
        reviews = item.get("reviewsData") or item.get("reviews") or item.get("detailedReviews") or item.get("latestReviews")
        
        if reviews and isinstance(reviews, list):
            for r in reviews:
                review_text = r.get("text") or r.get("textTranslated") or r.get("textOriginal")
                
                if not review_text or review_text.strip() == "":
                    review_text = "ไม่มีข้อความรีวิว (ให้คะแนนอย่างเดียว)"
                    
                reviews_list.append({
                    "Review Text": review_text,
                    "Stars": r.get("stars") or r.get("rating") or 5,
                    "Reviewed At": r.get("publishAt") or "ไม่ระบุเวลา"
                })
                
    return place_name, pd.DataFrame(reviews_list)

def analyze_with_ai(df_reviews, model_name='gemini-3.1-flash-lite', max_retries=3, initial_delay=10):
    all_reviews_text = ""
    for _, row in df_reviews.iterrows():
        all_reviews_text += f"- คะแนน: {row['Stars']} ดาว | รีวิว: {row['Review Text']}\n"
        
    prompt = f"""
    คุณเป็น AI ผู้เชี่ยวชาญด้านการวิเคราะห์ข้อมูลการท่องเที่ยวและการบริการ (Hospitality Data Analyst)
    จงสรุปรีวิวดิบด้านล่างนี้ออกมาเป็นภาษาไทยอย่างมืออาชีพ โดยใช้ Markdown ในการจัดหัวข้อ ตัวหนา และไอคอนอิโมจิให้สวยงามน่าอ่าน:
    
    💡 หัวข้อที่ต้องสรุป:
    1. ภาพรวมความพึงพอใจ (สรุปสัดส่วนแง่บวก/แง่ลบ หรือสรุปมู้ดรวมๆ)
    2. 🌟 จุดเด่น/ข้อดีที่คนชื่นชอบ (สรุปมาเป็นข้อๆ)
    3. ⚠️ จุดด้อย/สิ่งที่คนบ่นหรือเจอปัญหา (สรุปมาเป็นข้อๆ)
    4. 👥 กลุ่มลูกค้าหลัก (สถานที่นี้เหมาะกับใคร)
    5. 🚀 คำแนะนำเชิงกลยุทธ์สำหรับผู้ประกอบการในการปรับปรุงและทำมาร์เก็ตติ้ง
    
    --- ข้อมูลรีวิวดิบ ---
    {all_reviews_text}
    """
    
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(model=model_name, contents=prompt)
            return response.text
        except ClientError as e:
            if "429" in str(e) or "EXHAUSTED" in str(e):
                time.sleep(delay)
                delay *= 2
            else:
                raise e
    raise Exception("โควตา AI เต็มชั่วคราว กรุณาลองใหม่อีกครั้ง")

# --- 3. UI/UX CONFIGURATION ---
st.set_page_config(page_title="RoamInsight AI", page_icon="🗺️", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("🗺️ ReviewInsight AI — เครื่องมือวิเคราะห์สถานที่ท่องเที่ยวอัจฉริยะ")
st.caption("ดึง Insight จากรีวิวบน Google Maps แปลงข้อมูลดิบให้กลายเป็นกลยุทธ์ทางธุรกิจด้วยพลังของ Gemini LLM")
st.write("---")

# ส่วนรับข้อมูลเข้า (Input Section)
with st.sidebar:
    st.header("⚙️ ตั้งค่าการค้นหา")
    target_url = st.text_input("🔗 วางลิงก์ Google Maps ที่นี่:", placeholder="http://googleusercontent.com/maps.google.com/...")
    
    max_revs = st.number_input(
        "📊 จำนวนรีวิวที่ใช้วิเคราะห์:", 
        min_value=5, max_value=100, value=15, step=1
    )
    
    sort_options = {
        "✨ รีวิวล่าสุด (Newest)": "newest",
        "🔥 แนะนำ/เกี่ยวข้องที่สุด (Relevant)": "mostRelevant",
        "⭐ คะแนนสูงสุด (Highest Rating)": "highestRanking",
        "❌ คะแนนต่ำสุด (Lowest Rating)": "lowestRanking"
    }
    selected_sort_label = st.selectbox("🔝 จัดเรียงรีวิวตามแบบที่คุณต้องการ:", list(sort_options.keys()))
    review_sort_value = sort_options[selected_sort_label]
    
    st.write("---")
    st.markdown("""
    **💡 ทริคการใช้งาน:**
    1. ใช้ลิงก์ URL ด้านบนเบราว์เซอร์ จากGoogle Map
    2. จำกัดจำนวนรีวิวให้น้อยเพราะต้องประหยัดโควตา
    """)
    
    submit_btn = st.button("เริ่มวิเคราะห์ข้อมูล 🚀", type="primary")

# ประกาศตัวแปรใน session_state เพื่อจำค่าข้อมูลหลังการ Rerun
if "place_name" not in st.session_state:
    st.session_state.place_name = None
if "df_reviews" not in st.session_state:
    st.session_state.df_reviews = None
if "ai_report" not in st.session_state:
    st.session_state.ai_report = None

# เมื่อกดปุ่มค้นหา
if submit_btn:
    if not target_url:
        st.warning("⚠️ กรุณากรอกลิงก์ Google Maps ในแถบเมนูด้านซ้ายก่อนครับ")
    else:
        with st.spinner("🕵️‍♂️ กำลังเชื่อมต่อระบบและดึงข้อมูลรีวิวจาก Google Maps (ขั้นตอนนี้ใช้เวลาประมาณ 10-15 วินาที)..."):
            try:
                # รีเซ็ตรายงานเก่าก่อนดึงข้อมูลสถานที่ใหม่
                st.session_state.ai_report = None
                
                place_name, df = get_reviews_from_maps(target_url, max_reviews=max_revs, review_sort=review_sort_value)
                
                if df.empty:
                    st.error("❌ ดึงข้อมูลสำเร็จ แต่ไม่พบข้อความรีวิวในลิงก์นี้")
                else:
                    st.balloons() 
                    st.session_state.place_name = place_name
                    st.session_state.df_reviews = df
            except Exception as error:
                st.error(f"เกิดข้อผิดพลาดในการประมวลผลระบบ: {error}")

# แสดงผลลัพธ์ข้อมูลในระบบ
if st.session_state.df_reviews is not None:
    place_name = st.session_state.place_name
    df = st.session_state.df_reviews.copy() # ใช้ .copy() ป้องกัน Warning เผื่อมีการแปลง Data
    
    st.header(f"🏢 ผลลัพธ์การวิเคราะห์: {place_name}")
    
    # --- METRIC CARDS & SUMMARY ---
    avg_stars = df['Stars'].mean()
    total_reviews = len(df)
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric(label="⭐ คะแนนดาวเฉลี่ยจากกลุ่มตัวอย่าง", value=f"{avg_stars:.2f} / 5.0")
    m_col2.metric(label="💬 จำนวนรีวิวที่นำมาคำนวณ", value=f"{total_reviews} รีวิว")
    
    st.write("---")
    
    # --- 📊 NEW VISUALIZATION SECTION ---
    st.subheader("📊 การวิเคราะห์ข้อมูลเชิงภาพ (Data Visualizations)")
    
    # Feature Engineering แบบเร็วๆ: คำนวณความยาวข้อความรีวิว
    df['Review Length'] = df['Review Text'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
    
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        # 1. กราฟวงกลมแสดงสัดส่วนดาว (Pie Chart)
        star_counts = df['Stars'].value_counts().reset_index()
        star_counts.columns = ['Stars', 'Count']
        # เปลี่ยนเป็น string เพื่อให้ Plotly แสดงผลเป็นหมวดหมู่ (Discrete Color)
        star_counts['Stars'] = star_counts['Stars'].astype(str) + " ดาว" 
        
        fig_pie = px.pie(
            star_counts, 
            values='Count', 
            names='Stars', 
            title='สัดส่วนคะแนนรีวิวจากลูกค้า (%)',
            color_discrete_sequence=px.colors.sequential.YlOrRd[::-1], # ไล่สีโทนส้ม-แดงอุ่นๆ
            hole=0.3 # ทำเป็น Donut Chart ให้ดูโมเดิร์น
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with g_col2:
        # 2. กราฟวิเคราะห์ความยาวรีวิวตามระดับคะแนน (Box Plot / Stripplot)
        # ช่วยให้เห็นภาพว่า คนที่ให้ดาวน้อยหรือดาวมาก ตั้งใจพิมพ์รีวิวอธิบายยาวกว่ากัน
        df_chart = df.copy()
        df_chart['Stars'] = df_chart['Stars'].astype(str) + " ดาว"
        
        fig_box = px.box(
            df_chart, 
            x='Stars', 
            y='Review Length',
            title='ความยาวของข้อความรีวิว แยกตามระดับคะแนน (จำนวนตัวอักษร)',
            labels={'Stars': 'ระดับคะแนน', 'Review Length': 'ความยาวรีวิว (ตัวอักษร)'},
            color='Stars',
            category_orders={"Stars": ["5 ดาว", "4 ดาว", "3 ดาว", "2 ดาว", "1 ดาว"]},
            points="all" # แสดงจุดข้อมูลรีวิวแต่ละอันคู่กับกล่อง Boxplot
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.write("---")
    
    # แบ่งหน้าจอแสดงตารางและผลวิเคราะห์ AI
    col1, col2 = st.columns([4, 5])
    
    with col1:
        st.subheader("📋 ตารางรีวิวดิบจากลูกค้า")
        st.dataframe(df[['Review Text', 'Stars', 'Reviewed At']], use_container_width=True, height=500)
        
    with col2:
        st.subheader("🤖 บทสรุปและคำแนะนำเชิงลึกโดย AI")
        
        if st.session_state.ai_report is None:
            with st.spinner("🧠 AI กำลังอ่านและประมวลผลข้อมูล..."):
                st.session_state.ai_report = analyze_with_ai(df, model_name='gemini-3.1-flash-lite')
        
        st.info(st.session_state.ai_report)
        
        st.download_button(
            label="📥 ดาวน์โหลดรายงานสรุป (.txt)",
            data=st.session_state.ai_report,
            file_name=f"RoamInsight_{place_name.replace(' ', '_')}.txt",
            mime="text/plain"
        )
else:
    st.info("👈 กรอกข้อมูลในแถบด้านซ้ายและกดปุ่มเริ่มวิเคราะห์ข้อมูลเพื่อเริ่มต้นใช้งานระบบ")
