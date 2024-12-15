

<img src="https://github.com/user-attachments/assets/79e359fb-809d-4aa6-87db-43e6223f052f" width = "400"/>

## Night Jars - The Dark Detector Using Intel AI Optimizataion Tools 

> #### 👉🏻 [Colab notebook🔗](https://colab.research.google.com/drive/1oVe-88LOAtx-HLJsvrhuNWzKaY4QZ2uE?usp=sharing)

> #### 👉🏻 [Jupyter notebook intel® tiber™ ai developer cloud 🔗](https://jupyter-batch-us-region-1.cloud.intel.com/hub/user-redirect/lab/tree/Nightjars_intel_bindascode.ipynb)

<br>

**Goal & Motivation** : Image enhancement specially in the Low-light conditions & Thermal Imaging. Live distance estimation, Tracking with DeepSort and Count of the objects under low-light environment

<br>

> #### Nightjars - Overall project Flow
<img width="700" alt="Screenshot_20241207_032418" src="https://github.com/user-attachments/assets/e575434c-01c8-49b9-9959-f28ea93e4921">

<br>

> #### NightJars final annotated image with distance estimation with improvisation of low illuminated scene

<img width="700" alt="Screenshot_20241207_031410" src="https://github.com/user-attachments/assets/fef8b741-f36c-4de6-8337-59ab4a956fb9">
<img width="700" src="https://github.com/user-attachments/assets/cf2a3738-2c98-49d5-9ea8-38d33bd7332a">

<br>

> #### Nightjars - The Dark Detector Using Improved YOLOv8cdark model and Intel Optimization Tools.

![yoloarch](https://github.com/user-attachments/assets/9f176758-9dd1-4938-8a95-8b6134ce1678)

### **Repository Structure**
```
├── streamlit_app.py             # Main File
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
├── yolov8xcdark.pt              # Nightjars Pytorch model
├── yolov8xcdark-seg.pt          # Nightjars Segmentation Pytorch model
├── yolov8xcdark.pt              # Nightjars Pytorch model
├── yolov8xcdark_openvino_model  # Optimized Nightjars Openvino model
└── ReferenceImages/             # Distance estimation util
```

### **Steps to Run the App**

#### **Step 1: Clone the Repository**

```python
!git clone https://github.com/Brindha-m/Nightjars_IntelAI.git
```


---

#### **Step 2: Navigate to the Extracted Directory**
Use the `cd` command to navigate into the cloned repository.

```python
%cd Nightjars_IntelAI
```
---

#### **Step 3: Install Dependencies**
Install the required Python packages specified in the `requirements.txt` file.

```python
!pip install -r requirements.txt
```

---

#### **Step 4: Run the Streamlit App**
Start the Streamlit app using the following command:

```python
!streamlit run streamlit_app.py
```

- use  `&>/dev/null &` part ensures the app runs in the background without blocking your Colab/Jupyter notebook.

---


## Give it a Try [Nightjars](https://nightjars-brindha.streamlit.app/)🌐 
