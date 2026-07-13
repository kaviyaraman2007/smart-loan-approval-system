class DataManagement:

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
            "Age": int(Age),
            "Mobile": Mobile,
            "Email": Email,

            "Gender": Gender,
            "Married": Married,
            "Dependents": int(Dependents),
            "Education": Education,
            "Self_Employed": Self_Employed,
            "ApplicantIncome": float(ApplicantIncome),
            "CoapplicantIncome": float(CoapplicantIncome),
            "LoanAmount": float(LoanAmount),
            "Loan_Amount_Term": float(Loan_Amount_Term),
            "Credit_History": int(Credit_History),
            "Property_Area": Property_Area

        }

        return applicant


    def _get_int_input(self, value, field_name):
        try:
            return int(value)
        except ValueError:
            return f"Invalid input for {field_name}. Please enter an integer."


    def _get_float_input(self, value, field_name):
        try:
            return float(value)
        except ValueError:
            return f"Invalid input for {field_name}. Please enter a number."


    def validate_data(self, applicant):

        if applicant["ApplicantIncome"] <= 0:
            return False, "Invalid Applicant Income"

        if applicant["LoanAmount"] <= 0:
            return False, "Invalid Loan Amount"

        if applicant["Age"] < 18:
            return False, "Applicant must be at least 18 years old."

        if len(applicant["Mobile"]) != 10:
            return False, "Invalid Mobile Number"

        if "@" not in applicant["Email"]:
            return False, "Invalid Email Address"

        if applicant["Loan_Amount_Term"] <= 0:
            return False, "Invalid Loan Term"

        if applicant["Credit_History"] not in [0, 1]:
            return False, "Invalid Credit History"

        return True, "Validation Successful"


    def preprocess_data(self, applicant):

        # Encoding Example
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
        
class MachineLearning:

    def load_model(self):

        # No trained model, using rule-based dummy model
        model = "Dummy Loan Model"

        return model


    def predict_loan(self, model, data):

        applicant_income = data[0][5]
        loan_amount = data[0][7]
        credit_history = data[0][9]

        # Dummy ML decision rules

        if (
            credit_history == 1
            and applicant_income >= 30000
            and loan_amount <= 500000
        ):

            return [1]       # Approved

        else:

            return [0]       # Rejected


    def confidence(self, model, data):

        applicant_income = data[0][5]
        credit_history = data[0][9]

        # Dummy confidence calculation

        if credit_history == 1 and applicant_income >= 30000:

            return 94.75     # High confidence

        else:

            return 68.50     # Low confidence

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

            "Prediction":
                "APPROVED" if prediction[0] == 1 else "REJECTED",

            "Confidence":
                round(confidence,2)

        }

        df = pd.DataFrame([result])

        df.to_csv(
            "prediction_history.csv",
            mode="a",
            index=False,
            header=False
        )

        return "Prediction Saved Successfully!"  
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
        )

        valid, message = self.data.validate_data(applicant)

        if valid:

            processed = self.data.preprocess_data(applicant)

            model = self.ml.load_model()

            prediction = self.ml.predict_loan(model, processed)

            confidence = self.ml.confidence(model, processed)

            result = self.result.display_result(
                applicant,
                prediction,
                confidence
            )

            self.result.save_prediction(
                applicant,
                prediction,
                confidence
            )

            return result

        else:

            return {"Error": message}

system = SmartLoanSystem()
