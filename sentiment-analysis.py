from transformers import pipeline

classifier = pipeline(
            "sentiment-analysis", 
            model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
            )


text = input("Enter text to analyze sentiment: ")

result = classifier(text)

print(result)