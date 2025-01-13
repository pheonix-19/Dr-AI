import joblib
import pandas as pd
import numpy as np

def load_model(model_path='model/trained_model.pkl'):
    """Load the trained model from disk."""
    return joblib.load(model_path)

def create_example_case():
    """Create an example case for prediction."""
    example_data = {
        'age': 45,
        'gender': 'M',
        'blood_pressure': 130,
        'cholesterol': 200,
        # Add other relevant features your model expects
    }
    return pd.DataFrame([example_data])

def prepare_data(df):
    """Prepare the data for prediction (apply same preprocessing as training)."""
    # Add any necessary preprocessing steps here
    # For example:
    df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    return df

def main():
    try:
        # Load the model
        model = load_model()
        
        # Create and prepare example case
        example_case = create_example_case()
        prepared_data = prepare_data(example_case)
        
        # Make prediction
        prediction = model.predict(prepared_data)
        prediction_prob = model.predict_proba(prepared_data)
        
        # Display results
        print("\nExample Case:")
        print(example_case.to_string())
        print("\nPrediction:", prediction[0])
        print("Probability:", prediction_prob[0])
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
