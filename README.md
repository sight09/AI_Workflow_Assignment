# AI_Workflow_Assignment

Author: Amanuel Alemu Zewdu

This repository contains:
- A formal academic report (PDF) describing the AI development workflow.
- Two demonstration Python scripts that implement simplified end-to-end workflows for two example problems:
  1. Student Dropout Prediction (`code_student_dropout_prediction.py`)
  2. Hospital Readmission Prediction (`code_hospital_readmission_prediction.py`)

## How to run the code
1. Create a Python environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: .\venv\Scripts\activate
   pip install pandas numpy scikit-learn
   ```

2. Run the student dropout example:
   ```bash
   python code_student_dropout_prediction.py
   ```

3. Run the hospital readmission example:
   ```bash
   python code_hospital_readmission_prediction.py
   ```

## Files
- `code_student_dropout_prediction.py`: End-to-end example with Random Forest.
- `code_hospital_readmission_prediction.py`: Logistic regression example for readmission.
- `AI_Workflow_Report.pdf`: The PDF report (created alongside these files).

## Notes
- The datasets used in these demonstration scripts are synthetic and intended for pedagogical purposes.
- All comments and docstrings are written in formal academic English to meet assignment requirements.
