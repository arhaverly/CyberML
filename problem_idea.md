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



What are the key things I want to extract?
How does a new contract compare to the big 5?




## Archive

### Questions from Professor Yang
Real-time or in the background?
* Data that has to be stolen real-time:
    * Location data
    * GPS location
* Data that can be stolen at any time (in background processes):
    * IP address
    * Type of device
    * Browsing and search history
    * Content of messages exchanged with others on the app
    * Phone and social-network contacts
    * Age
    * Phone number
* Data that only needs to be stolen once (or very rarely):
    * Type of device
    * Age
    * Phone number


Where specifically would ML be good? Is it a ML problem or is it a protocol issue?
* ML useful for:
    * Determining which data to pay attention to:
        * Ignore:
            * Video data
            * Messaging data
            * Other content data
        * Pay attention to:
            * Anomalies that contain private information

    * Anomalous internet traffic that could be running in the background

    * NLP for reading through the terms and conditions

### Feasibility -- Not feasible. It is compressed.
* There is a clear path to:
    * Gathering the data: download known apps that steal data and recording their traffic. For the NLP side of things, the terms and conditions legally have to be publicly available.
    
    * Examining the data: It will not be encrypted if I perform a network capture and it will (probably) be relatively easy to classify the data for the model by hand
    
    * Making a model: This is a classification model. I will have to look more into which model would best suit my purposes, but this should not be incredibly difficult
    
        * For NLP, there are publicly available models that will make this relatively easy.

    * Possibly check if URL is being used to steal data








