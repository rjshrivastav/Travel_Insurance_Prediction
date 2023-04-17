# Travel_Insurance_Prediction

This Repository contains all code, reports and approach.
#### Task: build a solution that should able to predict the chances of buying travel insurance

## Dataset Description:
* **Age**- Age Of The Customer
* **Employment Type**- The Sector In Which Customer Is Employed
* **GraduateOrNot**- Whether The Customer Is College Graduate Or Not
* **AnnualIncome**- The Yearly Income Of The Customer In Indian Rupees[Rounded To Nearest 50 Thousand Rupees]
* **FamilyMembers**- Number Of Members In Customer's Family
* **ChronicDisease**- Whether The Customer Suffers From Any Major Disease Or Conditions Like Diabetes/High BP or Asthama,etc.
* **FrequentFlyer**- Derived Data Based On Customer's History Of Booking Air Tickets On Atleast 4 Different Instances In The Last 2 Years[2017-2019].
* **EverTravelledAbroad**- Has The Customer Ever Travelled To A Foreign Country[Not Necessarily Using The Company's Services]
* **TravelInsurance**- Did The Customer Buy Travel Insurance Package During Introductory Offering held in 2019.

## Installation

To get started with this project, follow these steps:

**Step 1:** Clone the repository to your local machine:
```git clone https://github.com/Rjshrivastav/Travel_Insurance_Prediction.git```


**Step 2:** Create a virtual environment and activate it. If you don't have virtualenv installed, you can install it using `pip install virtualenv`.

`virtualenv env # For creating virtual environment`

`source env/bin/activate # For Linux/Mac`

`.\env\Scripts\activate # For Windows`


**Step 3:** Install the required Python packages using the following command:
```pip install -r requirements.txt```


**Step 4:** To generate a trained model, run the following command:
```python src/components/data_ingestion.py```

This will create a tuned model that will be used for predictions.

**Step 5:** Navigate to the main directory of the project and run the following command to start the web application:
```python app.py```

This will start the server and the web application will be accessible in your localhost. Simply open your web browser and enter the URL `http://localhost:5000` to access the application.



That's it! You should now be able to use the Travel Insurance Prediction web application on your local machine.


