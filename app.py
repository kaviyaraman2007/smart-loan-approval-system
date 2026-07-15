# ===========================================
# SMART LOAN APPROVAL SYSTEM (FLASK)
# ===========================================

import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)


# ===========================================
# DATA MANAGEMENT
# ===========================================

class DataManagement:

    def _safe_int(self, value, default=0):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _safe_float(self, value, default=0.0):
        try:
            # Strip commas or currency symbols if copy-pasted
            clean_val = str(value).replace(",", "").replace("$", "").strip()
            return float(clean_val)
        except (ValueError, TypeError):
            return default

    def receive_input(
        self,
        ApplicantName,
        Age,
        Mobile,
        Email,
        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Property_Area
    ):
        applicant = {
            "ApplicantName": ApplicantName,
            "Age": self._safe_int(Age),
            "Mobile": Mobile if Mobile else "",
            "Email": Email if Email else "",
            "Gender": Gender,
            "Married": Married,
            "Dependents": self._safe_int(Dependents),
            "Education": Education,
            "Self_Employed": Self_Employed,
            "ApplicantIncome": self._safe_float(ApplicantIncome),
            "CoapplicantIncome": self._safe_float(CoapplicantIncome),
            "LoanAmount": self._safe_float(LoanAmount),
            "Loan_Amount_Term": self._safe_float(Loan_Amount_Term),
            "Credit_History": self._safe_int(Credit_History, default=-1), 
            "Property_Area": Property_Area
        }
        return applicant

    def validate_data(self, applicant):
        if not applicant["ApplicantName"] or len(applicant["ApplicantName"].strip()) == 0:
            return False, "Applicant Name cannot be blank."

        if applicant["ApplicantIncome"] <= 0:
            return False, "Applicant Income must be greater than 0."

        if applicant["LoanAmount"] <= 0:
            return False, "Loan Amount must be greater than 0."

        if applicant["Age"] < 18:
            return False, "Applicant must be at least 18 years old."

        if len(applicant["Mobile"]) != 10 or not applicant["Mobile"].isdigit():
            return False, "Invalid Mobile Number. It must be exactly 10 digits."

        if "@" not in applicant["Email"] or "." not in applicant["Email"]:
            return False, "Invalid Email Address."

        if applicant["Loan_Amount_Term"] <= 0:
            return False, "Invalid Loan Term."

        if applicant["Credit_History"] not in [0, 1]:
            return False, "Please select a valid Credit History option."

        return True, "Validation Successful"

    def preprocess_data(self, applicant):
        gender = 1 if applicant["Gender"] == "Male" else 0
        married = 1 if applicant["Married"] == "Yes" else 0
        education = 1 if applicant["Education"] == "Graduate" else 0
        employed = 1 if applicant["Self_Employed"] == "Yes" else 0
        credit = applicant["Credit_History"]

        property_area = {
            "Urban": 2,
            "Semiurban": 1,
            "Rural": 0
        }.get(applicant["Property_Area"], 0)

        features = np.array([[
            gender,
            married,
            applicant["Dependents"],
            education,
            employed,
            applicant["ApplicantIncome"],
            applicant["CoapplicantIncome"],
            applicant["LoanAmount"],
            applicant["Loan_Amount_Term"],
            credit,
            property_area
        ]])
        return features


# ===========================================
# MACHINE LEARNING
# ===========================================

class MachineLearning:

    def load_model(self):
        return "Dummy Loan Model"

    def predict_loan(self, model, data):
        applicant_income = data[0][5]
        loan_amount = data[0][7]
        credit_history = data[0][9]

        if (
            credit_history == 1
            and applicant_income >= 5000
            and loan_amount <= 300000
        ):
            return [1]       # APPROVED
        else:
            return [0]       # REJECTED

    def confidence(self, model, data):
        applicant_income = data[0][5]
        credit_history = data[0][9]

        if credit_history == 1 and applicant_income >= 5000:
            return 94.75     
        else:
            return 68.50     


# ===========================================
# RESULT MANAGEMENT
# ===========================================

class ResultManagement:

    def display_result(self, applicant, prediction, confidence):
        if prediction[0] == 1:
            status = "APPROVED"
            message = "Congratulations! Your loan is approved."
        else:
            status = "REJECTED"
            message = "Sorry! Your loan application is rejected."

        return {
            "Applicant Name": applicant["ApplicantName"],
            "Mobile Number": applicant["Mobile"],
            "Email": applicant["Email"],
            "Loan Status": status,
            "Message": message,
            "Confidence": f"{round(confidence,2)}%"
        }

    def save_prediction(self, applicant, prediction, confidence):
        result = {
            "ApplicantName": applicant["ApplicantName"],
            "Mobile": applicant["Mobile"],
            "Email": applicant["Email"],
            "Prediction": "APPROVED" if prediction[0] == 1 else "REJECTED",
            "Confidence": round(confidence, 2)
        }

        df = pd.DataFrame([result])
        filename = "prediction_history.csv"

        file_exists = os.path.isfile(filename)
        df.to_csv(
            filename,
            mode="a",
            index=False,
            header=not file_exists
        )
        return "Prediction Saved Successfully!"


# ===========================================
# SMART LOAN SYSTEM
# ===========================================

class SmartLoanSystem:

    def __init__(self):
        self.data = DataManagement()
        self.ml = MachineLearning()
        self.result = ResultManagement()

    def run(
        self,
        ApplicantName,
        Age,
        Mobile,
        Email,
        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Property_Area
    ):
        applicant = self.data.receive_input(
            ApplicantName, Age, Mobile, Email, Gender, Married, Dependents,
            Education, Self_Employed, ApplicantIncome, CoapplicantIncome,
            LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
        )

        valid, message = self.data.validate_data(applicant)

        if valid:
            processed = self.data.preprocess_data(applicant)
            model = self.ml.load_model()
            prediction = self.ml.predict_loan(model, processed)
            confidence = self.ml.confidence(model, processed)

            result = self.result.display_result(applicant, prediction, confidence)
            self.result.save_prediction(applicant, prediction, confidence)
            return result
        else:
            return {"Error": message}


system = SmartLoanSystem()

# ===========================================
# FLASK ROUTES
# ===========================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index1")
def index1(): 
    return render_template("index1.html")

@app.route("/index2")
def index2(): 
    return render_template("index2.html")

@app.route("/index3")
def index3(): 
    return render_template("index3.html")

@app.route("/index4")
def index4(): 
    return render_template("index4.html")

@app.route("/index5")
def index5(): 
    return render_template("index5.html")


@app.route("/predict", methods=["POST"])
def predict():
    result = system.run(
        request.form.get("ApplicantName"),
        request.form.get("Age"),
        request.form.get("Mobile"),
        request.form.get("Email"),
        request.form.get("Gender"),
        request.form.get("Married"),
        request.form.get("Dependents"),
        request.form.get("Education"),
        request.form.get("Self_Employed"),
        request.form.get("ApplicantIncome"),
        request.form.get("CoapplicantIncome"),
        request.form.get("LoanAmount"),
        request.form.get("Loan_Amount_Term"),
        request.form.get("Credit_History"),
        request.form.get("Property_Area")
    )

    # 1. Validation failed -> Athe Form page-la clean-ah error message kaatum!
    if "Error" in result:
        return render_template("index.html", error=result["Error"])

    # 2. Validation Success -> Direct-a unga dynamic result.html page open aagum!
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
