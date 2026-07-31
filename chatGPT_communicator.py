import os

import pandas as pd
from itertools import combinations
import openai
from google.cloud import storage
from google.oauth2 import service_account
import datetime
from dotenv import load_dotenv

# Create all unique pairs of sampled images
def create_pairs(sampled_images):
    print("Creating unique pairs of sampled images...")
    pairs = list(combinations(sampled_images, 2))
    print(f"Created {len(pairs)} pairs.")
    return pairs

class ChatGPTCommunicator:
    def __init__(self):

        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.SERVICE_ACCOUNT_FILE = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        self.signed_url_cache = {}

        self.input_csv_path = os.environ["INPUT_CSV_PATH"]
        self.output_csv_path = os.environ["OUTPUT_CSV_PATH"]
        self.bucket_name = os.environ["GCS_BUCKET_NAME"]

    def start(self):

        ser = self.load_image_names()
        self.perform_pairwise_ranking(list(ser))

    # Load the CSV file with image names and class information
    def load_image_names(self):
        print("Loading CSV file...")
        ser = pd.read_csv(self.input_csv_path)["file_name"]
        print(f"Loaded {len(ser)} image file names from CSV file.")
        return ser

    def get_gcs_image_url(self, file_name):
        current_time = datetime.datetime.now()

        # Check if URL is in cache and still valid
        if file_name in self.signed_url_cache:
            url, timestamp = self.signed_url_cache[file_name]
            # Check if the URL is about to expire in the next 15 minutes
            if (current_time - timestamp).seconds < 60 * 45:
                return url

        # Load the service account key file
        credentials = service_account.Credentials.from_service_account_file(self.SERVICE_ACCOUNT_FILE)

        # Initialize a Google Cloud Storage client with the credentials
        storage_client = storage.Client(credentials=credentials)

        # Get the bucket
        bucket = storage_client.bucket(self.bucket_name)

        # Get the blob for the given file name
        blob = bucket.blob(file_name)

        # Generate signed URL for the blob (image)
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=6),  # Increase expiration time to 6 hours
            method="GET",
            credentials=credentials
        )

        self.signed_url_cache[file_name] = (url, current_time)
        return url

    # Function to call OpenAI API and get the response for a question involving two images
    def get_openai_response(self, image1_url, image2_url, question):
        print(f"Calling OpenAI API with question: {question}")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": image1_url}},
                            {"type": "image_url", "image_url": {"url": image2_url}}
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=300,
            )
            print("OpenAI API call successful.")
            return response.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            if hasattr(e, 'response'):
                error_message = f"Error code: {e.response.status_code} - {e.response.text}"
            print(f"Error during OpenAI API call: {error_message}")
            return None

    # Rank each pair using OpenAI API and print the comparison results
    def rank_pair(self,image1, image2):
        print(f"Ranking pair: {image1} vs {image2}")
        image1_url = self.get_gcs_image_url(image1)
        image2_url = self.get_gcs_image_url(image2)

        for attempt in range(2):  # Allow up to 2 attempts to resolve ambiguity
            question = (
                f"Compare the following two images based on specific indicators of wealth such as: "
                f"1) Quality of infrastructure (e.g., building materials, visible road conditions), "
                f"2) Number of floors in the buildings, "
                f"3) Amount of visible greenery or well-maintained areas, "
                f"4) Presence of visible amenities (e.g., paved roads, power lines). "
                f"Please respond with either 'Image 1 is wealthier' or 'Image 2 is wealthier'."
            )
            response = self.get_openai_response(image1_url, image2_url, question)
            print(f"Model Response: {response}")  # Log the response

            if response is None:
                continue

            # Determine the winner from the response
            response_lower = response.lower()

            if "image 1" in response_lower and "image 2" not in response_lower:
                winner, loser = image1, image2
                print(f"Winner: {winner} | Loser: {loser}")
                return winner
            elif "image 2" in response_lower and "image 1" not in response_lower:
                winner, loser = image2, image1
                print(f"Winner: {winner} | Loser: {loser}")
                return winner
            else:
                print("Tie or ambiguous response detected.")

        return None

    # Perform pairwise ranking with improved direct comparison and transitivity propagation
    def perform_pairwise_ranking(self, sampled_images):
        print("Performing pairwise ranking...")
        all_pairs = set(create_pairs(sampled_images))

        previous_cmps = pd.read_csv(self.output_csv_path)
        prev_pairs = {
            frozenset(pair)
            for pair in previous_cmps[["image1", "image2"]].itertuples(index=False, name=None)
        }

        pairs = []
        for pair in all_pairs:
            image1 = pair[0]
            image2 = pair[1]

            # if neither the pair nor its reverse is in prev_pairs
            if frozenset((image1, image2)) not in prev_pairs:
                pairs.append(pair)

        # Make comparisons and write the results to output csv file
        for image_pair in pairs:
            image1, image2 = image_pair
            better_image = self.rank_pair(image1, image2)
            with open(self.output_csv_path, "a") as file:
                file.write(
                    f"{image1},{image2},{"tie" if better_image is None else better_image}\n")

        print("Pairwise ranking completed.")

def main():

    load_dotenv() # load the environment variables from the .env file
    communicator = ChatGPTCommunicator()
    communicator.start()

if __name__ == "__main__":
    main()