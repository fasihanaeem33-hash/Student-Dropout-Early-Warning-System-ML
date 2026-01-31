# 🎓 Student Dropout Early Warning System

A Streamlit-based web application that uses machine learning to predict student dropout risk and identify at-risk students early.

## Features

- **Model Upload**: Upload your trained scikit-learn model (`.joblib` or `.pkl`)
- **CSV Processing**: Upload student data and get instant risk predictions
- **Risk Classification**: Automatic categorization into High, Medium, and Low risk levels
- **Top Risk Students**: View the 20 highest-risk students sorted by risk score
- **Individual Inspection**: Select any student to see their detailed risk score and level
- **Robust Error Handling**: Automatic sklearn version compatibility and helpful error messages

## Project Structure

```
Student Dropout Prediction/
├── app.py                          # Main Streamlit application
├── student_dropout_model.joblib    # Trained ML model
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── venv/                          # Python virtual environment
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/student-dropout-prediction.git
cd "Student Dropout Prediction"
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application

```bash
streamlit run app.py --server.port 8506
```

Open your browser to: **http://localhost:8506**

### Using the App

1. **Upload a Model** (if using your own):
   - Click "Upload trained model" and select your `.joblib` or `.pkl` file

2. **Upload Student Data**:
   - Prepare a CSV file with student features (numeric columns only)
   - Click "Upload Student CSV" to submit data

3. **View Results**:
   - See top 20 highest-risk students
   - Select any student to view risk score and level

## Data Processing & ML Approach

### Data Cleaning
- Missing values → median / most frequent imputation
- Categorical encoding → OneHot encoding
- Feature scaling → StandardScaler

### Features Used
- Attendance rate
- Assignment submissions
- Quiz scores
- LMS activity
- Past failures

### Model & Metrics
- **Algorithm**: Logistic Regression / Random Forest
- **Metric**: ROC-AUC
- **Optimization**: High Recall for early detection

### Risk Thresholds

| Level | Range | Action |
|-------|-------|--------|
| 🔴 High Risk | ≥ 0.7 | Immediate intervention |
| 🟡 Medium Risk | 0.4–0.7 | Increase support |
| 🟢 Low Risk | < 0.4 | Monitor quarterly |

### Why Students Are At Risk
- **Low Attendance**: Missing classes indicates disengagement
- **Poor Early Assessments**: Low quiz/test scores suggest knowledge gaps
- **Low Engagement**: Minimal LMS activity and assignment submission
- **Previous Failures**: Past academic struggles predict future dropout

## Model Format

Your trained model should:
- Be a scikit-learn compatible estimator (RandomForest, LogisticRegression, etc.)
- Have a `predict_proba()` method
- Be serialized using `joblib.dump()`

### Example:

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, "student_dropout_model.joblib")
```

## CSV Input Format

Example:
```csv
student_id,gpa,attendance,test_score,study_hours,assignments_completed
1,3.5,95,85,20,18
2,2.8,70,65,10,8
3,3.9,98,92,25,24
```

## Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8507
```

### Model File Not Found
- Place `student_dropout_model.joblib` in the project root
- Or upload via the UI

### Feature Mismatch
Ensure your CSV has the same number of numeric features as the model expects.

### sklearn Version Issues
```bash
pip install --upgrade scikit-learn joblib
```

## Requirements

- Python 3.8+
- Streamlit
- pandas
- scikit-learn
- joblib
- numpy
- scipy

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

---

**Built with ❤️ using [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/)**