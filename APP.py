import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# تحميل المودل
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('chest_xray_model.h5')
    return model

model = load_model()

# معالجة الصورة
def preprocess_image(image):
    # تحويل لـ numpy array
    img = np.array(image)
    
    # تحويل لـ grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize
    img = cv2.resize(img, (128, 128))
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add dimensions
    img = np.expand_dims(img, axis=-1)  # (128, 128, 1)
    img = np.expand_dims(img, axis=0)   # (1, 128, 128, 1)
    
    return img

# واجهة التطبيق
st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🫁")

st.title(" تشخيص أشعة الصدر")
st.markdown("### رفع صورة أشعة للتحليل")

# رفع الصورة
uploaded_file = st.file_uploader(
    "اختر صورة أشعة صدر", 
    type=['png', 'jpg', 'jpeg'],
    help="ارفع صورة أشعة X للصدر"
)

if uploaded_file is not None:
    # عرض الصورة
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("الصورة المرفوعة")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    # زر التحليل
    if st.button(" تحليل الصورة", type="primary"):
        with st.spinner("جاري التحليل..."):
            # معالجة وتوقع
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img, verbose=0)[0][0]
            
            # عرض النتيجة
            with col2:
                st.subheader("النتيجة")
                
                if prediction > 0.5:
                    st.error(" غير طبيعي (Abnormal)")
                    confidence = prediction * 100
                else:
                    st.success(" طبيعي (Normal)")
                    confidence = (1 - prediction) * 100
                
                # st.metric("نسبة الثقة", f"{confidence:.1f}%")
                
                # Progress bar
                # st.progress(float(confidence/100))
                
                # تحذير
                st.warning("هذا التشخيص للإرشاد فقط. يُرجى استشارة طبيب متخصص.")

# معلومات جانبية
with st.sidebar:
    st.header("معلومات")
    st.info("""
    **كيفية الاستخدام:**
    1. ارفع صورة أشعة صدر
    2. اضغط على "تحليل"
    3. شاهد النتيجة
    
    **الفئات:**
    -  Normal: طبيعي
    -  Abnormal: غير طبيعي
    """)
    
    st.markdown("---")
    st.caption(" Powered by Deep Learning")


    # streamlit run APP.py