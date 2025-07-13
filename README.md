# Titanic ML Model Interpretation Toolbox

A demonstration of machine learning interpretability techniques using the Titanic dataset, showcasing SHAP values, counterfactual analysis (DiCE), and natural language explanations.

## Installation and Use

1. **Prerequisites**
   - Python 3.8+

2. **Setup**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd mantel_project
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Create .env file (optional, for LLM explanations)
   echo "HF_API_KEY=your_huggingface_api_key_here" > .env
   ```

3. **Run the application**
   ```bash
   # Option 1: Using the provided script
   chmod +x run_demo.sh
   ./run_demo.sh
   
   # Option 2: Run directly with Streamlit
   streamlit run streamlit_shap_demo.py
   ```

4. **Application Testing**
   - Manually test the three tabs: SHAP Explainer, What-If Scenario, and Counterfactual Generator
   - Verify that explanations are generated correctly for different passenger profiles

## Project Structure

```
mantel_project/
├── modules/                  # Core functionality modules
├── streamlit_shap_demo.py    # Main Streamlit application
├── run_demo.sh               # Setup and run script
└── requirements.txt          # Python dependencies
```
