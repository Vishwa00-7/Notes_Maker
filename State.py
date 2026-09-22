from typing_extensions import TypedDict

#State

class State(TypedDict, total=False):
    #Inputs
    topic : str                 #Topic that need to be discussed
    methodology : str           #Methodology of study material
    prequeist_knowledge : str   #Prerequisite knowledge does the Reader have
    depth : str                 #depth of the study material
    level : str                 #level of the study material
    teaching_style : str        #teaching Style of study material

    #Researcher
    question : str              #Input Question
    roadmap : list              #List of Topics that Should be Learned
    length : int                #Length of the Roadmap

    #Progress
    progress : int              #Progress show how much is completed 
    last_completed : str        #Last node that is executed
    no_of_attempts_local : int  #no of time it failed (consecutively)
    no_of_attempts_global : int #no of time it failed 
    status : str                #Current status (e.g. running, stopped, completed)

    #failed
    failed_at : str

    #Prompt Generator
    meta_prompt : str           #Specialized module prompt
    prompts : str               #Prompts for the list of topics
    filename : str              #Last File name 

    #Solution
    answers : str               #Last answers provided by ai
    summary : list              #summary of the answer 
