# Use ML to determine if an application is stealing your data.
Analyze the network traffic, see if personal data is in there


## Types of personal data stolen
* Location data
* Internet address
* Type of device
* Browsing and search history
* Content of messages exchanged with others on the app
* Phone and social-network contacts
* GPS position
* Age
* Phone number


## Feasibility
* There is a clear path to:
    * Gathering the data: download known apps that steal data and recording their traffic. For the NLP side of things, the terms and conditions legally have to be publicly available.
    
    * Examining the data: It will not be encrypted if I perform a network capture and it will (probably) be relatively easy to classify the data for the model by hand
    
    * Making a model: This is a classification model. I will have to look more into which model would best suit my purposes, but this should not be incredibly difficult
    
        * For NLP, there are publicly available models that will make this relatively easy.

