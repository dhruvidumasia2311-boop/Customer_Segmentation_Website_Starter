# Customer Segmentation Using Clustering — PDS CIPAT/PBL

## Files
- `app.py`: the Streamlit website.
- `sample_customers.csv`: sample customer dataset for testing.
- `requirements.txt`: Python libraries needed.

## Run on your laptop
1. Install Python 3.10 or newer from https://www.python.org/downloads/ (during Windows installation, tick "Add Python to PATH").
2. Extract this ZIP into a folder.
3. Open Command Prompt/Terminal in that folder.
4. Run: `python -m pip install -r requirements.txt`
5. Run: `python -m streamlit run app.py`
6. Your browser should open the website. If it does not, open the local URL shown in the terminal, usually `http://localhost:8501`.

## How to use
1. Keep the sample CSV or upload your own CSV.
2. Select at least two numeric columns.
3. Select K (number of clusters).
4. Click "Run Customer Segmentation".
5. Review the graph, cluster summary, and table; download the results if needed.

## Suggested columns for a customer dataset
- `AnnualIncome_k`
- `SpendingScore`
- `TenureMonths`
- `MonthlyCharges`

The sample file is fictional and is only for demonstration. K-Means cluster numbers are arbitrary; interpret clusters using their average feature values.
