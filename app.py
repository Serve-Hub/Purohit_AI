from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from bson import ObjectId

app = Flask(__name__)

# MongoDB Connection
userName = "kuber"
password = "kuber123"
dbName = "purohit"
host = "kuber.3wbojvn.mongodb.net"
client = MongoClient(f"mongodb+srv://{userName}:{password}@{host}/{dbName}?retryWrites=true&w=majority&appName=Kuber")
db = client[dbName]

# Function to Get Accepted and Selected Pandits
def get_accepted_and_selected_pandits(booking_id):
    print(f"Fetching accepted and selected pandits for booking ID: {booking_id}")
    booking = db.bookings.find_one({"_id": ObjectId(booking_id)}, {"acceptedPandit": 1, "selectedPandit": 1})
    if booking:
        accepted_pandit = booking.get('acceptedPandit', [])
        selected_pandit = booking.get('selectedPandit', [])
        return accepted_pandit, selected_pandit
    else:
        print(f"No booking found for ID: {booking_id}")
        return [], []

# Function to Get Ratings for Accepted Pandits
def get_ratings(accepted_pandit, selected_pandit):
    print(f"Fetching ratings for accepted pandits: {accepted_pandit}")
    relevant_pandits = list(set(accepted_pandit) - set(selected_pandit))
    if not relevant_pandits:
        return pd.DataFrame()
    reviews_data = list(db.reviews.find({"pandit": {"$in": [ObjectId(p) for p in relevant_pandits]}}, {"user": 1, "pandit": 1, "rating": 1}))
    return pd.DataFrame(reviews_data)

# Function to Get Pandit Details
def get_pandit_details():
    print("Fetching pandit details.")
    pandits_data = list(db.users.find({"isPandit": True}, {"_id": 1, "firstName": 1, "lastName": 1}))
    return pd.DataFrame(pandits_data)

# Main Recommendation Logic with Weighted Score for Rating and Review Count
def recommend_pandits(booking_id):
    print(f"Starting recommendation for booking ID: {booking_id}")
    accepted_pandit, selected_pandit = get_accepted_and_selected_pandits(booking_id)
    accepted_pandit = [str(p) for p in accepted_pandit]
    selected_pandit = [str(p) for p in selected_pandit]
    
    if not accepted_pandit:
        print("No accepted pandits for this booking.")
        return pd.DataFrame()

    # Get the ratings for the accepted pandits
    ratings = get_ratings(accepted_pandit, selected_pandit)
    if ratings.empty:
        print("No ratings available for the accepted pandits.")
        return pd.DataFrame()
    
    # Fetch pandit details
    pandits = get_pandit_details()

    # Merge Ratings with Pandit Details
    merged_df = pd.merge(ratings, pandits, left_on='pandit', right_on='_id', how='left')
    ratings_with_name = merged_df[['user', 'pandit', 'firstName', 'lastName', 'rating']]

    # Pivot Table for User-Pandit Ratings
    pt = ratings_with_name.pivot_table(index="pandit", columns="user", values="rating")
    pt = pt.fillna(0)
    
    # Cosine Similarity Calculation
    similarity_score = cosine_similarity(pt.fillna(0))
    
    # Recommendation Logic
    def get_recommendations(accepted_pandit, selected_pandit):
        print(f"Generating recommendations for accepted pandits: {accepted_pandit}")
        recommended = []
        for pandit in accepted_pandit:
            try:
                index = np.where(pt.index == pandit)[0][0]
                similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]  # Get top 5 similar pandits
                for i in similar_items:
                    recommended.append(pt.index[i[0]])
            except IndexError:
                continue
        
        # Exclude Selected Pandits and Already Accepted Pandits from Recommendations
        final_recommended = list(set(recommended) - set(selected_pandit) - set(accepted_pandit))
        print(f"Final recommended pandits after exclusion: {final_recommended}")
        return final_recommended

    recommended_pandits = get_recommendations(accepted_pandit, selected_pandit)
    
    # If no recommendations, fallback to showing accepted pandits
    if not recommended_pandits:
        print("No recommendations found, using fallback.")
        recommended_pandits = accepted_pandit
    
    # Fetch Details of Recommended Pandits
    recommended_details = pandits[pandits['_id'].isin([ObjectId(p) for p in recommended_pandits])]
    
    # Calculate and show average ratings and review count for the recommended pandits
    average_ratings = {}
    review_counts = {}
    for pandit_id in recommended_pandits:
        # Filter ratings for this pandit
        pandit_ratings = ratings[ratings['pandit'] == ObjectId(pandit_id)]
        average_ratings[pandit_id] = pandit_ratings['rating'].mean()
        review_counts[pandit_id] = len(pandit_ratings)
    
    # Add average rating and review count to the recommended pandits
    recommended_details['averageRating'] = recommended_details['_id'].apply(lambda x: average_ratings.get(str(x), 'No ratings'))
    recommended_details['reviewCount'] = recommended_details['_id'].apply(lambda x: review_counts.get(str(x), 0))
    
    # Define weights for rating and review count
    w_rating = 2  # Weight for rating
    w_review_count = 1  # Weight for review count
    
    # Calculate Weighted Score
    recommended_details['weightedScore'] = (recommended_details['averageRating'] * w_rating + recommended_details['reviewCount'] * w_review_count)
    
    # Sort Pandits by weighted score
    recommended_details_sorted = recommended_details.sort_values(by='weightedScore', ascending=False)
    
    print(f"Returning {len(recommended_details_sorted)} recommended pandits.")
    return recommended_details_sorted

# Flask Route for Recommendation
@app.route('/')
def home():
    return "Welcome to the Purohit Recommendation System!"

@app.route('/recommend_pandits', methods=['POST'])
def recommend():
    print("Received a request to recommend pandits.")
    booking_id = request.form.get('booking_id')
    if not booking_id:
        print("No booking_id provided in the request.")
        return jsonify({"message": "Booking ID is required."})
    
    recommended_pandits = recommend_pandits(booking_id)
    
    if recommended_pandits.empty:
        print("No recommendations found.")
        return jsonify({"message": "No recommendations found."})
    
    # Convert ObjectId to string for JSON serialization
    result = recommended_pandits[['firstName', 'lastName', 'averageRating', 'reviewCount', 'weightedScore', '_id']].copy()
    result['_id'] = result['_id'].apply(lambda x: str(x))  # Convert ObjectId to string
    
    print(f"Returning the recommended pandits: {result}")
    return jsonify(result.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
