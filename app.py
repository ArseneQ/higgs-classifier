import streamlit as st
import torch
import pandas as pd
import numpy as np
import io
import os

# Define your model class (same as your existing model)
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.3):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.LeakyReLU(negative_slope=0.01))
            layers.append(torch.nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        layers.append(torch.nn.Linear(prev_size, output_size))
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Function to load model (simplified)
@st.cache_resource
def load_model(model_path):
    """Load the saved PyTorch model"""
    # Set up device
    device = torch.device("cpu")
    
    # Create model with the same architecture used during training
    model = MLP(
        input_size=106,  # Adjust to your actual input size
        hidden_layers=[512, 256, 128, 64],
        output_size=2  # Binary classification
    )
    
    # Load the weights directly
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to make predictions
def predict(model, data, threshold=0.5):
    """Make predictions with the model"""
    # Basic preprocessing - standardize the data
    normalized_data = (data - data.mean()) / data.std()
    
    # Convert to PyTorch tensor
    input_tensor = torch.tensor(normalized_data.values, dtype=torch.float32)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # Convert to numpy for easier handling
    probs_numpy = probabilities.numpy()
    
    # Apply threshold for binary classification
    signal_probs = probs_numpy[:, 1]
    predicted_classes = (signal_probs > threshold).astype(int)
    
    return predicted_classes, signal_probs

# Main Streamlit app
st.title("Higgs Boson Event Classifier")
st.write("Upload your collision data CSV file to classify events as potential Higgs boson signals or background.")

# Load model (adjust path to your model file)
model = load_model("best_model_FinalWithFocal.pth")

if model is None:
    st.error("Failed to load model. Please check if the model file exists.")
else:
    st.success("Model loaded successfully!")

    # File uploader
    uploaded_file = st.file_uploader("Upload collision data (CSV)", type=["csv"])
    
    # Threshold slider
    threshold = st.slider(
        "Classification Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        step=0.01,
        help="Adjust the threshold for signal classification"
    )
    
    if uploaded_file is not None:
        # Read the CSV file
        try:
            data = pd.read_csv(uploaded_file)
            st.write(f"Uploaded data: {data.shape[0]} events, {data.shape[1]} features")
            
            # Show a preview of the data
            st.subheader("Data Preview:")
            st.dataframe(data.head(5))
            
            # Make predictions
            if st.button("Classify Events"):
                with st.spinner("Processing data..."):
                    predicted_classes, signal_probabilities = predict(model, data, threshold)
                    
                    # Display results
                    signal_count = sum(predicted_classes)
                    total_count = len(predicted_classes)
                    signal_ratio = signal_count / total_count if total_count > 0 else 0
                    
                    st.subheader("Classification Results")
                    st.write(f"Total events: {total_count}")
                    st.write(f"Predicted signal events: {signal_count} ({signal_ratio:.2%})")
                    
                    # Create a results dataframe
                    results_df = pd.DataFrame({
                        "Event_ID": range(len(data)),
                        "Signal_Probability": signal_probabilities,
                        "Classification": ["Signal" if p else "Background" for p in predicted_classes]
                    })
                    
                    # Display results table
                    st.subheader("Sample Predictions (first 10 events):")
                    st.dataframe(results_df.head(10))
                    
                    # Add a download button for full results
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Full Results",
                        csv,
                        "higgs_classification_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # Visualization - distribution of probabilities
                    st.subheader("Distribution of Signal Probabilities")
                    hist_values = pd.DataFrame(signal_probabilities, columns=["Probability"])
                    st.bar_chart(hist_values.Probability.value_counts(bins=10, normalize=True))
                    
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin analysis.")

# Add information about the model and usage
st.sidebar.title("About")
st.sidebar.info(
    "This application uses a neural network to identify potential Higgs boson events "
    "from particle collision data. The model focuses on the H→ZZ→4ℓ decay channel, "
    "commonly known as the 'golden channel'."
)

st.sidebar.title("Instructions")
st.sidebar.markdown(
    """
    1. Upload a CSV file containing collision data
    2. Adjust the classification threshold if needed
    3. Click "Classify Events" to analyze the data
    4. View the results and download the full classification
    
    **Note:** Your CSV should contain the same features used during model training.
    """
)