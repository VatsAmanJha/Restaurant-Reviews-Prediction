# NLP-Based Sentiment Analysis for Restaurant Reviews

This project is an NLP-based web application for sentiment analysis of restaurant reviews. It utilizes a machine learning model served with FastAPI for the backend and an HTML file with Bootstrap and jQuery for the frontend.

## Project Overview

The goal of this project is to analyze the sentiment of restaurant reviews using Natural Language Processing (NLP) techniques. The backend server built with FastAPI provides an API endpoint for predicting the sentiment of user-provided reviews. The frontend client, implemented with HTML, Bootstrap, and jQuery, allows users to interactively enter their reviews and receive sentiment predictions.

## Project Structure

- `main.py`: FastAPI backend serving the sentiment analysis model.
- `client.html`: HTML file for the frontend user interface.
- `text_preprocessing.py`: Python module for text cleaning and preprocessing.
- `model.pkl`: Pre-trained machine learning model serialized using pickle.
- `restaurantReviews_model.ipynb`: Jupyter Notebook containing the model training and evaluation.

## Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/VatsAmanJha/Restaurant-Reviews-Prediction.git
    ```

2. **Set up a Virtual Environment**:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the FastAPI Server**:
    ```sh
    uvicorn main:app --reload
    ```

5. **Open the HTML Client**:
    Open the `client.html` file in your web browser.

## File Descriptions

### `main.py`

The FastAPI backend provides a single API endpoint `/predict` for sentiment prediction of restaurant reviews.

### `client.html`

The HTML file contains the user interface where users can input their reviews and view sentiment predictions.

### `text_preprocessing.py`

This Python module includes functions for text preprocessing, such as removing stop words, punctuation, and performing tokenization.

### `model.pkl`

The serialized machine learning model (RandomForestClassifier()) used for sentiment analysis.

### `restaurantReviews_model.ipynb`

A Jupyter Notebook containing the code for training the sentiment analysis model and evaluating its performance.

## Usage

1. **Start the FastAPI Server**:
    ```sh
    uvicorn main:app --reload
    ```

2. **Open the HTML Client**:
    Open `client.html` in your web browser.

3. **Enter a review and get the prediction**:
    - Enter a restaurant review in the text area.
    - Click the "Predict" button.
    - The sentiment prediction will be displayed as an emoji (😊 for positive, 😞 for negative).

## Example

1. Start the FastAPI server:
    ```sh
    uvicorn main:app --reload
    ```
2. Open `client.html` in your web browser.
3. Enter a review: "The food was excellent and the service was great!"
4. Click "Predict".
5. The prediction result (😊) will be displayed.

## Dependencies

- FastAPI
- pandas
- numpy
- pydantic
- scikit-learn
- uvicorn
- jQuery
- Bootstrap

## Contributing

Contributions via issues or pull requests are welcome!
