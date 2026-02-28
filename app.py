import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
import io

st.set_page_config(page_title="Face Attendance", layout="wide")
st.title("📸 Face Recognition Attendance System")

# --- Load embeddings ---
@st.cache_resource
def load_embeddings():
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)  # {"regno": embedding_vector}
    regnos = list(embeddings.keys())
    embed_vectors = np.array(list(embeddings.values()))
    return embeddings, regnos, embed_vectors

embeddings, regnos, embed_vectors = load_embeddings()

# --- Initialize FaceAnalysis model ---
@st.cache_resource
def load_face_model():
    app = FaceAnalysis(name="antelope", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, nms=0.4)
    return app

face_app = load_face_model()

# --- Attendance storage ---
if "attendance" not in st.session_state:
    st.session_state.attendance = {}

# --- Camera Feed ---
stframe = st.empty()
run = st.checkbox("Start Camera")
cap = cv2.VideoCapture(0)

def get_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def recognize(face_embedding):
    distances = [get_distance(face_embedding, ev) for ev in embed_vectors]
    min_idx = np.argmin(distances)
    if distances[min_idx] < 0.6:
        return regnos[min_idx]
    return "Unknown"

def mark_attendance(regno):
    now = datetime.now()
    st.session_state.attendance[regno] = {"Date": now.strftime("%Y-%m-%d"),
                                          "Time": now.strftime("%H:%M:%S")}

def create_excel():
    return pd.DataFrame.from_dict(st.session_state.attendance, orient="index")

def create_pdf(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Attendance Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    data = [list(df.columns)]
    for i in df.index:
        data.append([i] + list(df.loc[i]))
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.gray),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- Main loop ---
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not found")
        break
    frame = cv2.flip(frame, 1)
    faces = face_app.get(frame)
    for face in faces:
        box = face.bbox.astype(int)
        face_embedding = normalize(face.embedding.reshape(1, -1))[0]
        regno = recognize(face_embedding)
        if regno != "Unknown":
            mark_attendance(regno)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        cv2.putText(frame, regno, (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

cap.release()

# --- Attendance Display ---
st.subheader("📋 Attendance Table")
if st.session_state.attendance:
    df = create_excel()
    st.dataframe(df)

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=True)
    st.download_button("📥 Download Excel", data=excel_buffer,
                       file_name="attendance.xlsx",
                       mime="application/vnd.ms-excel")

    pdf_buffer = create_pdf(df)
    st.download_button("📄 Download PDF", data=pdf_buffer,
                       file_name="attendance.pdf",
                       mime="application/pdf")
else:
    st.info("No attendance marked yet.")