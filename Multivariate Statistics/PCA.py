import streamlit as st


st.title('Principal Component Analysis')
st.markdown("""   
<ul>
    <p><strong>Principal Component Analysis (PCA)</strong> เป็นเทคนิคที่ใช้ในการลดมิติ (dimensionality reduction) ซึ่งมีจุดประสงค์ในการลดจำนวนของตัวแปรในชุดข้อมูลขณะที่ยังคงข้อมูลสำคัญไว้มากที่สุด PCA ช่วยให้เราสามารถย่อยข้อมูลที่มีมิติสูง (เช่น ข้อมูลที่มีหลายตัวแปร) ให้อยู่ในมิติที่ต่ำลงโดยไม่สูญเสียข้อมูลสำคัญมากจนเกินไป โดยอาศัยหลักการทางคณิตศาสตร์เพื่อหาทิศทางใหม่ที่สามารถอธิบายความแปรผันในข้อมูลได้ดีที่สุด</p>
    <h5>หลักการของ PCA</h5>
    <p>PCA เป็นเทคนิคที่ทำการแปลงข้อมูลจากชุดตัวแปรต้น (features) ไปยังชุดตัวแปรใหม่ที่เรียกว่า principal components (PCs) ซึ่งจะไม่สัมพันธ์กัน (uncorrelated) และมีการจัดลำดับตามความสำคัญของข้อมูลที่อธิบายได้มากที่สุด</p>
    <h6>ขั้นตอนการทำ PCA มีดังนี้:</h6>
        <p>1.การคำนวณค่าเฉลี่ย (Mean Centering)</p>
            <ul>
                <li>ในการทำ PCA ขั้นแรกจะต้องหาค่าเฉลี่ยของข้อมูลในแต่ละฟีเจอร์ (ตัวแปร) และทำการลบค่าเฉลี่ยออกจากแต่ละจุดข้อมูล เพื่อให้ข้อมูลมีค่าเฉลี่ยเป็น 0</li>
            </ul>
        <p>2.การคำนวณ Covariance Matrix</p>
            <ul>
                <li>Covariance matrix คือการวัดว่าความแปรผันของตัวแปรหนึ่งสัมพันธ์กับตัวแปรอื่นอย่างไร การคำนวณ covariance matrix จะช่วยให้เราทราบว่าแต่ละตัวแปรในชุดข้อมูลมีความสัมพันธ์กันอย่างไร</li>
            </ul>
        <p>3.การหาค่า Eigenvalues และ Eigenvector</p>
            <ul>
                <li>หลังจากคำนวณ covariance matrix แล้ว เราจะหาค่า Eigenvalues และ Eigenvectors ของมัน
                    <ul>
                        <li>Eigenvalue: บ่งบอกถึงความสำคัญของแต่ละ component ในการอธิบายความแปรผันในข้อมูล</li>
                        <li>Eigenvector: แสดงทิศทางของ component ที่สามารถอธิบายความแปรผันในข้อมูลได้มากที่สุด</li>
                    </ul>
                </li>
                <li>Eigenvectors ที่ได้จะเป็นทิศทางใหม่ที่เราใช้ในการแปลงข้อมูล และ Eigenvalues จะบอกว่า component ไหนอธิบายความแปรผันของข้อมูลได้มากที่สุด</li>
            </ul>
        <p>4.การหาค่า Eigenvalues และ Eigenvector</p>
            <ul>
                <li>ใการเลือก principal components (PCs) จะทำโดยการเลือก eigenvectors ที่มี eigenvalues ใหญ่ที่สุด ซึ่งบ่งบอกว่า component เหล่านั้นสามารถอธิบายข้อมูลได้มากที่สุด</li>
                <li>โดยปกติแล้วจะเลือกจำนวน principal components ที่สามารถอธิบายความแปรผันในข้อมูลได้มากที่สุด เช่น 95% หรือ 99%</li>
             </ul>
        <p>5.การแปลงข้อมูล (Transformation)</p>
            <ul>
                <li>เมื่อเลือก principal components ได้แล้ว ขั้นตอนสุดท้ายคือการแปลงข้อมูลเดิมไปอยู่ใน space ใหม่ที่ประกอบด้วย principal components ที่เลือกไว้</li>
                <li>ข้อมูลที่แปลงแล้วจะมีมิติที่ต่ำลง และเราได้ข้อมูลใหม่ที่ไม่สัมพันธ์กัน (uncorrelated)</li>
             </ul>
    <h5>ทำไมต้องใช้ PCA?</h5>
    <p>PCA ถูกใช้เมื่อมีข้อมูลที่มีมิติสูง (เช่น ข้อมูลที่มีหลายฟีเจอร์) ซึ่งทำให้การวิเคราะห์ข้อมูลยากและมีปัญหาด้านการคำนวณได้ เช่น Overfitting หรือการคำนวณที่ช้า การลดมิติด้วย PCA ช่วยลดความซับซ้อนของโมเดล และยังช่วยให้โมเดลสามารถจับความสัมพันธ์ในข้อมูลได้ดีขึ้น</p>
    <h5>ข้อดีของ PCA</h5>
        <ul>
            <p>1.ลดมิติข้อมูล: ช่วยลดจำนวนตัวแปรในชุดข้อมูลโดยที่ยังคงรักษาข้อมูลสำคัญไว้</p>
            <p>2.ลดความซับซ้อนของโมเดล: ลดการคำนวณที่เกี่ยวข้องกับข้อมูลมิติสูง</p>
            <p>3.ลดปัญหา Multicollinearity: PCA ช่วยกำจัดปัญหาการสัมพันธ์กันระหว่างตัวแปร (collinearity) เนื่องจาก components ที่ได้จะไม่สัมพันธ์กัน</p>
            <p>4.ปรับปรุงประสิทธิภาพ: ช่วยให้โมเดลทำงานได้เร็วขึ้นและมีประสิทธิภาพมากขึ้นเมื่อใช้ข้อมูลที่ลดมิติแล้ว</p>
        </ul>
    <h5>ข้อเสียของ PCA</h5>
        <ul>
            <p>1.ไม่สามารถอธิบายได้ง่าย: Components ที่ได้จาก PCA มักจะเป็นการผสมกันของฟีเจอร์หลายตัว จึงยากที่จะอธิบายความหมายของแต่ละ component</p>
            <p>2.สูญเสียข้อมูลบางอย่าง: หากเลือก principal components ที่อธิบายข้อมูลได้น้อยเกินไปอาจสูญเสียข้อมูลที่สำคัญ</p>
            <p>3.ไม่เหมาะกับข้อมูลที่ไม่เป็นเชิงเส้น: PCA เหมาะกับข้อมูลที่มีความสัมพันธ์เชิงเส้นเท่านั้น ถ้าข้อมูลมีความสัมพันธ์แบบไม่เป็นเชิงเส้น อาจต้องใช้เทคนิคอื่นเช่น Kernel PCA</p>
        </ul>
    <h5>ตัวอย่างการใช้ PCA ด้วย Python</h5>
</ul>
""", unsafe_allow_html=True)

code = ("""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# โหลดข้อมูล Iris
iris = load_iris()
X = iris.data  # ข้อมูลตัวแปรต้น
y = iris.target  # ข้อมูลเป้าหมาย

# ทำ PCA เพื่อลดมิติเป็น 2 มิติ
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# สร้างกราฟเพื่อแสดงข้อมูลที่ลดมิติแล้ว
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Iris Dataset')
plt.colorbar()
plt.show()

# อธิบายผลลัพธ์
print(f'Explained Variance Ratio: {pca.explained_variance_ratio_}')
print(f'Total Explained Variance: {sum(pca.explained_variance_ratio_)}')

""")
st.code(code,language='python')
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
# โหลดข้อมูล Iris
iris = load_iris()
X = iris.data  # ข้อมูลตัวแปรต้น
y = iris.target  # ข้อมูลเป้าหมาย
target_names = iris.target_names

# ทำ PCA เพื่อลดมิติเป็น 2 มิติ
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# สร้างกราฟเพื่อแสดงข้อมูลที่ลดมิติแล้ว
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Iris Dataset')
plt.colorbar()
plt.show()
st.pyplot(plt)

# อธิบายผลลัพธ์
st.write(f'Explained Variance Ratio: {pca.explained_variance_ratio_}')
st.write(f'Total Explained Variance: {sum(pca.explained_variance_ratio_)}')


#===================
# Convert to DataFrame for easier plotting
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y
df_pca['target_name'] = [target_names[i] for i in y]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PC1', y='PC2', 
    hue='target_name', 
    palette='viridis', 
    data=df_pca,
    s=100
)
plt.title("PCA of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(loc='best', title="Classes")
plt.grid(True)
st.pyplot(plt)

#===================
pca = PCA()
pca.fit(X)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Plot the explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--', label='Cumulative Variance')
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, label='Explained Variance', color='blue')
plt.title('Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance Explained')
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.legend(loc='best')
plt.grid(True)
st.pyplot(plt)


st.markdown("""   
<ul>
    <h5>อธิบายโค้ด</h5>
    <p>1.โหลดข้อมูล: ใช้ข้อมูลจาก Iris dataset ซึ่งมี 4 ฟีเจอร์ (sepal length, sepal width, petal length, petal width) และ 3 คลาส (species)</p>
    <p>2.ทำ PCA: ลดมิติข้อมูลจาก 4 มิติ (ฟีเจอร์เดิม) ไปยัง 2 มิติ</p>
    <p>3.แสดงผลลัพธ์: ใช้กราฟ scatter plot เพื่อแสดงข้อมูลหลังการลดมิติ พร้อมกับแสดงสีตามคลาสของข้อมูล</p>
    <p>ผลลัพธ์ที่ได้จาก explained_variance_ratio_ จะบอกว่าแต่ละ principal component สามารถอธิบายความแปรผันในข้อมูลได้กี่เปอร์เซ็นต์ เช่น หาก PC1 สามารถอธิบายได้ 80% และ PC2 อธิบายได้ 15% หมายความว่าเราใช้แค่ 2 components นี้ก็สามารถอธิบายความแปรผันของข้อมูลได้ 95%</p>           
    <h5>สรุป</h5>
    <p>PCA เป็นเครื่องมือที่ใช้สำหรับลดมิติข้อมูลในขณะที่ยังคงข้อมูลสำคัญไว้ การทำ PCA ช่วยให้ข้อมูลมีความสัมพันธ์ที่น้อยลงและสามารถแสดงผลได้ดีขึ้นในกราฟ ข้อดีของ PCA คือช่วยลดความซับซ้อนและเวลาในการคำนวณ แต่ก็มีข้อเสียคืออาจสูญเสียข้อมูลที่สำคัญบางอย่างหรือทำให้การตีความยากขึ้น</p>
</ul>
""", unsafe_allow_html=True)