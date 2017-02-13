## Twitter Chat Bot
- A Generative Chat Bot based on Sequence to Sequence model using Twitter Conversations as training data

The bot has two components:
1) A training component that is used to train the a sequence to sequence model in tensor flow on twitter dataset
2) A web server in flask that is used to connect the trained model with alexa (the responses of the bot are returned as reponses to an alexa skill)
