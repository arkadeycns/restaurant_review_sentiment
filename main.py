""" Sentiment Analysis for Restaurant Reviews - `llmware` Example

This program analyzes sentiment of restaurant reviews using `llmware`.
It demonstrates:

  1. Individual review sentiment analysis with confidence level check.
  2. Batch review sentiment analysis with journaling for multi-step processing.

"""

from llmware.agents import LLMfx

# Sample restaurant reviews
restaurant_reviews = [
    "This place is fantastic! Delicious food, friendly staff, and a great atmosphere. 5 stars!",
    "The food was disappointing, and the service was slow. Not impressed.",
    "Overall, a decent experience. The food was average, but the service was good.",
    "Had a wonderful experience! The food was amazing, and the service was excellent."
]


def analyze_review_sentiment(text, confidence_threshold=0.8):

  """ Analyzes sentiment of a single restaurant review with confidence check.

  Args:
      text: The text of the restaurant review.
      confidence_threshold: Minimum confidence score for sentiment classification (default 0.8).

  Returns:
      A string indicating the sentiment classification ("positive", "negative", or "neutral")
      if confidence score meets the threshold, otherwise "undetermined".
  """

  # Load sentiment analysis tool
  agent = LLMfx(verbose=True)
  agent.load_tool("sentiment")

  # Analyze sentiment
  sentiment = agent.sentiment(text)

  # Extract sentiment and confidence score
  sentiment_value = sentiment["llm_response"]["sentiment"]
  confidence_level = sentiment["confidence_score"]

  # Check confidence level before assigning sentiment
  if confidence_level >= confidence_threshold:
    return sentiment_value
  else:
    return "undetermined"


def analyze_batch_reviews(reviews, with_journaling=False):

  """ Analyzes sentiment of a batch of restaurant reviews with optional journaling.

  Args:
      reviews: A list of restaurant review strings.
      with_journaling: Enable detailed logging for each review analysis (default False).

  Returns:
      A list of sentiment classifications ("positive", "negative", "neutral", or "undetermined") 
      for each review.
  """

  # Load sentiment analysis tool
  agent = LLMfx()
  agent.load_tool("sentiment")

  # Load reviews for processing
  agent.load_work(reviews)

  # Analyze sentiment for each review with optional journaling
  review_sentiments = []
  while True:
    output = agent.sentiment()
    if not agent.increment_work_iteration():
      break
    sentiment_value = output["llm_response"]["sentiment"]
    if with_journaling:
      print(f"Review analysis: {sentiment_value}")
    review_sentiments.append(sentiment_value)

  # Clear work queue and agent state
  agent.clear_work()
  agent.clear_state()

  return review_sentiments


if __name__ == "__main__":

  # Analyze a single review with confidence check
  review = restaurant_reviews[0]
  sentiment = analyze_review_sentiment(review)
  print(f"Review sentiment: {sentiment}")

  # Analyze all reviews in the list with journaling enabled
  all_reviews_sentiment = analyze_batch_reviews(restaurant_reviews, with_journaling=True)
  print("\nReview sentiment analysis results:")
  for i, review_sentiment in enumerate(all_reviews_sentiment):
    print(f"Review {i+1}: {review_sentiment}")
