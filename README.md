## Twitter Chat Bot
- A Generative Chat Bot based on Sequence to Sequence model using Twitter Conversations as training data <br/>

The bot has two components: <br/>
1) A training component that is used to train the a sequence to sequence model in tensor flow on twitter dataset <br/>
2) A web server in flask that is used to connect the trained model with alexa (the responses of the bot are returned as reponses to an alexa skill) <br/>

The training set size was taken as: 301812 The test set size was taked as: 75453 <br/>

Evaluation: The following are a few qualitative examples: <br/>
how are you?
>i am fine . how are you ?

what do you think about president trump?
>he ' s a b####

what makes you think you are funny?
>a lot of people all the time

what do you about hilary clinton?
>she ' s on the list

what list are you talking about?
>i ' m not sure what you mean lol

you are not making any sense
>i ' m sorry for this

The model was trained with cross-entropy as the optimization function and perplexity was monitored. <br/>

Other evaluation techniqiues that could be used are: <br/>
1) Syntactic parsing of the response (using parse tree) to test for grammatical mistakes. <br/> 
2) Word overlap using longest common subsequence (without stop words) to find a relation between what is asked and what is replied. <br/> 
3) Over the span of the whole conversation, we can get topics out (using LDA or some other topid modelling) for the user and the response generated by the model. We can then do a basic overlap between the users questions and the models responses on the topics, and we can measure if the model is staying on topic.
