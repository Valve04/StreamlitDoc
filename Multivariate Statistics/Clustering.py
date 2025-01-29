import streamlit as st


st.title('Clustering')
st.markdown("""   
<ul>
    <p>Clustering คือกระบวนการจัดกลุ่มข้อมูล (data) โดยพิจารณาจากความเหมือนหรือความใกล้เคียงกันของข้อมูลในแต่ละกลุ่ม โดยที่ไม่มีข้อมูลเป้าหมาย (unsupervised learning) ใช้ในกรณีที่ต้องการค้นหารูปแบบหรือโครงสร้างในข้อมูลที่ไม่มีการจัดประเภทไว้ล่วงหน้า</p>
    <h5>Clustering ทำงานอย่างไร?</h5>
    <p>Clustering ทำหน้าที่แบ่งข้อมูลออกเป็นกลุ่มย่อยที่เรียกว่า clusters โดยพิจารณาจากระยะห่างหรือความคล้ายคลึงระหว่างจุดข้อมูล เช่น ข้อมูลที่มีความคล้ายคลึงกันมากที่สุดจะถูกจัดให้อยู่ในกลุ่มเดียวกัน และข้อมูลที่แตกต่างกันจะถูกจัดให้อยู่ในกลุ่มต่าง ๆ</p>
    <h5>ประเภทของ Clustering</h5>
        <ul>
            <h6>1.Partition-based Clustering</h6>
                <li>เป็นการแบ่งข้อมูลออกเป็นกลุ่มย่อย โดยทั่วไปต้องกำหนดจำนวนกลุ่มล่วงหน้า (เช่น k ใน k-means)</li>
                <li>ตัวอย่าง: k-means, k-medoids</li>
                <li>ข้อดี: เข้าใจง่ายและคำนวณได้เร็ว</li>
                <li>ข้อเสีย: ต้องกำหนดจำนวนกลุ่มล่วงหน้า</li>
            <h6>2.Hierarchical Clustering</h6>
                <li>แบ่งข้อมูลในรูปแบบลำดับชั้น โดยสร้างต้นไม้ (dendrogram) ที่แสดงถึงโครงสร้างของกลุ่ม</li>
                <li>มี 2 วิธี
                    <ul>
                        <li>Agglomerative (Bottom-Up): เริ่มจากข้อมูลแต่ละจุดเป็นกลุ่ม แล้วรวมกลุ่มขึ้นไปเรื่อย ๆ</li>
                        <li>Divisive (Top-Down): เริ่มจากกลุ่มใหญ่สุดแล้วแบ่งกลุ่มย่อยลงมา</li>
                    </ul>
                </li>
                <li>ข้อดี: ไม่ต้องกำหนดจำนวนกลุ่มล่วงหน้า</li>
                <li>ข้อเสีย: มีความซับซ้อนในการคำนวณเมื่อข้อมูลมีจำนวนมาก</li>
            <h6>3.Density-based Clustering</h6>
                <li>แบ่งข้อมูลโดยพิจารณาพื้นที่ที่มีความหนาแน่นสูง (density) ของจุดข้อมูล</li>
                <li>ตัวอย่าง: DBSCAN, OPTICS</li>
                <li>ข้อดี: สามารถจัดกลุ่มข้อมูลที่มีรูปร่างซับซ้อนและจุด outlier ได้ดี</li>
                <li>ข้อเสีย: มีปัญหาเมื่อค่าความหนาแน่นของกลุ่มแตกต่างกันมาก</li>
            <h6>4.Model-based Clustering</h6>
                <li>สร้างแบบจำลองความน่าจะเป็นสำหรับแต่ละกลุ่ม และกำหนดจุดข้อมูลให้อยู่ในกลุ่มที่มีความน่าจะเป็นสูงสุด</li>
                <li>ตัวอย่าง: Gaussian Mixture Models (GMM)</li>
                <li>ข้อดี: มีพื้นฐานทางคณิตศาสตร์ชัดเจน</li>
                <li>ข้อเสีย: สมมติฐานของแบบจำลองอาจไม่เหมาะสมกับข้อมูลจริง</li>
            <h6>5.Graph-based Clustering</h6>
                <li>ใช้โครงสร้างกราฟในการจัดกลุ่ม เช่น การใช้วิธีแบ่งกลุ่มจากความสัมพันธ์ในโครงข่าย (Graph Partitioning)</li>
                <li>ตัวอย่าง: Spectral Clustering</li>
                <li>ข้อดี: จัดการกับข้อมูลในลักษณะกราฟได้ดี</li>
                <li>ข้อเสีย: มีความซับซ้อนในการคำนวณสูง</li>
        </ul>  
    <h5>ตัวอย่างอัลกอริธึมยอดนิยม</h5>
        <ul>
            <h6>K-Means Clustering</h6>
                <li>ขั้นตอน:
                    <ul>
                        <li>เลือกจำนวนกลุ่ม (k) และสุ่มตำแหน่งเริ่มต้นของศูนย์กลางกลุ่ม (centroids)</li>
                        <li>กำหนดจุดข้อมูลให้กับกลุ่มที่ใกล้ที่สุด</li>
                        <li>คำนวณตำแหน่งใหม่ของศูนย์กลางกลุ่ม</li>
                        <li>ทำซ้ำจนกว่าศูนย์กลางกลุ่มจะไม่เปลี่ยน</li>
                    </ul>
                </li>
                <li>จุดเด่น: เรียบง่ายและคำนวณได้เร็ว</li>
                <li>จุดด้อย: ต้องกำหนดจำนวนกลุ่มล่วงหน้าและมีความไวต่อ outliers</li>
            <h6>DBSCAN (Density-Based Spatial Clustering of Applications with Noise)</h6>
                <li>ใช้ความหนาแน่นของจุดข้อมูลในการสร้างกลุ่ม</li>
                <li>มีพารามิเตอร์สำคัญ 2 ตัว:
                    <ul>
                        <li>ϵ (ระยะทางที่กำหนดว่าจุดอยู่ในพื้นที่เดียวกัน)</li>
                        <li>MinPts (จำนวนจุดขั้นต่ำในพื้นที่หนึ่งที่จะเป็นกลุ่ม)</li>
                    </ul>
                </li>
                <li>จุดเด่น: จัดการข้อมูลที่มีรูปทรงซับซ้อนได้ดี และกำจัด outliers ได้</li>
                <li>จุดด้อย: ไม่เหมาะกับข้อมูลที่ความหนาแน่นแตกต่างกันมาก</li>
            <h6>Hierarchical Clustering</h6>
                <li>สร้าง dendrogram เพื่อแสดงความสัมพันธ์ของข้อมูล</li>
                <li>เลือกระยะห่างที่เหมาะสมเพื่อตัด dendrogram และแบ่งข้อมูลออกเป็นกลุ่ม</li>
        </ul>
    <h5>การวัดคุณภาพของ Clustering</h5>
        <ul>
            <h6>1.Inertia (สำหรับ k-means)</h6>
                <li>วัดผลรวมของระยะทางจากจุดข้อมูลแต่ละจุดไปยังศูนย์กลางกลุ่ม</li>
                <li>ค่า inertia ต่ำบ่งบอกถึง clustering ที่ดี</li>
            <h6>2.Silhouette Score</h6>
                <li>วัดว่าแต่ละจุดอยู่ในกลุ่มที่เหมาะสมแค่ไหน โดยพิจารณาความใกล้เคียงกับจุดในกลุ่มเดียวกันและกลุ่มที่ใกล้ที่สุด</li>
                <li>ค่าอยู่ระหว่าง -1 ถึง 1 (ค่าที่ใกล้ 1 บ่งบอก clustering ที่ดี)</li>
            <h6>3.Davies-Bouldin Index</h6>
                <li>วัดความคล้ายคลึงระหว่างกลุ่มต่าง ๆ (ค่าน้อยแสดงว่า clustering ดี)</li>
            <h6>4.Rand Index (RI)</h6>
                <li>ใช้เปรียบเทียบการจัดกลุ่มกับการจัดกลุ่มที่เป็นจริง</li>
        </ul>
</ul>
""", unsafe_allow_html=True)
code = ("""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# สร้างข้อมูลตัวอย่าง
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# ใช้ KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# แสดงผลลัพธ์
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('KMeans Clustering')
plt.show()

# วัดคุณภาพด้วย Silhouette Score
silhouette_avg = silhouette_score(X, y_kmeans)
print(f"Silhouette Score: {silhouette_avg:.2f}")
""")
st.code(code,language='python')
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# สร้างข้อมูลตัวอย่าง
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# ใช้ KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# แสดงผลลัพธ์
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('KMeans Clustering')
st.pyplot(plt)

# วัดคุณภาพด้วย Silhouette Score
silhouette_avg = silhouette_score(X, y_kmeans)
st.write(f"Silhouette Score: {silhouette_avg:.2f}")

st.markdown("""   
<ul>
    <h5>สรุป</h5>
    <li>Clustering ช่วยแบ่งข้อมูลออกเป็นกลุ่มโดยไม่ต้องมี label</li>
    <li>อัลกอริธึมมีหลายประเภท เช่น k-means, DBSCAN, Hierarchical Clustering</li>
    <li>การเลือกวิธีการที่เหมาะสมขึ้นอยู่กับลักษณะของข้อมูลและวัตถุประสงค์</li>
    <li>สามารถใช้ PCA หรือการลดมิติเพื่อปรับปรุงผลลัพธ์ของ Clustering ได้</li>
</ul>
""", unsafe_allow_html=True)