import streamlit as st
from random import randrange
import pandas as pd
import json
import openai
import os
import chromadb
import time
import pprint
from haystack.core.component import Component
from haystack import Document
from haystack import component, default_to_dict, default_from_dict
from tqdm import tqdm 
import OpenAI

import knowledge_base
from sentence_transformers import SentenceTransformer
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

# important to export this before importing the local script (agent_interface)
openai_key = st.secrets["API_keys"]["openai"]
os.environ['OPENAI_API_KEY'] = openai_key

import prompts
import agent_interface

excel_profiles_path = "saved_profiles.csv"
HARDCODED = True
hardcoded = True
DEV = True

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Action Recommender Demo",
    layout="centered",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def init():
    # OpenAI client
    # client = openai.OpenAI(api_key = openai_key)
    client =OpenAI(api_key = openai_key)

    

    # DF with profiles that can be saved
    if os.path.isfile(excel_profiles_path):
        profiles = pd.read_csv("saved_profiles.csv") # pd.DataFrame(columns=["identification", "age", "gender", "diagnosis", "other_remarks"])
    else:
        profiles = pd.DataFrame(columns=["identification", "age", "gender", "diagnosis", "other_remarks", "history"])

    profiles = profiles.fillna("")

    
    return client, profiles

client, saved_profiles = init()    

#distance_docs_collection = doc_store.configuration_json['hnsw_configuration']['space']
#distance_caseStudies_collection = doc_store_cs.configuration_json['hnsw_configuration']['space']
            
@st.dialog("Choose a saved Profile")
def choose_profile():
    if saved_profiles.empty:
        st.warning("The List of saved profiles is Empty! \n Please create one or input the profile manually")
        return
    
    def write_name(index):
        return str(index) + ": " + saved_profiles.loc[index, 'identification']

    profile_index = st.selectbox("Profile:", range(len(saved_profiles)), format_func=write_name, index=None)

    if profile_index is not None:
        profile = saved_profiles.loc[profile_index]

        profile_string = f"{profile['age']} years, \n{profile['gender']}"   

        if profile['diagnosis'] != "":
            profile_string += f", \n{profile['diagnosis']}"
        if profile['other_remarks']:
            profile_string += f", \nOther Remarks: {profile['other_remarks']}"
        if profile['history']:
            profile_string += f", \nHistory of child's diagnosis: {profile['history']}"

        if st.button("Use this profile"):
            st.session_state['profile_string'] = profile_string
            st.session_state['profile_selected'] = True
            st.session_state['profile_announced'] = False
            st.rerun()

@st.dialog('Chat Backend Details')
def inspect_chat_dialogue():
    chat = st.session_state.chat_bot.get_chat_history()

    chat

    st.write(f"Chatting with agent currently id: {st.session_state.chat_bot.bot_type}")

def print_out_chunks(document_chunks=None, case_study_chunks=None):
    if document_chunks:
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("  --   Retrieved Document chunks:  -- /n")
        pprint.pprint(document_chunks)
        print("\n <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n\n")

    if case_study_chunks:
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("  --   Retrieved chunks - Case Studies:  -- /n")
        pprint.pprint(case_study_chunks)
        print("\n\n <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n\n")


# Main Web App
st.title("SEND AI Assistant")

st.markdown(
    """
    This app helps teachers working with students with special educational needs by:
    - **Evaluating an action** given a student profile and a situation.
    - **Recommending an action** given a student profile and a situation.
    - Providing a **Training Tool** for professionals.
    """
)

# Tabs for the three functionalities
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Input Profile", "Action Evaluator", "Action Recommender", "Training Module", "Conversational Assistant"])

# First Tab: Input Profiles
with tab1:

    new_profile = None

    known_diseases = ["None", "ADHD", "Autistic Spectrum Disorder", "Epilepsy", "Sensory Disorder", "Anxiety", "Dyslexia", "Dyspraxia","Dyscalculia", "Attachment Disorder"]

    with st.form("Input form to save a new profile", clear_on_submit=True):

        idname = st.text_input("Identification / Name")

        # st.markdown("#### :red[*] Age")
        age = st.text_input("Age")

        gender = st.radio("Gender", ("Male", "Female", "Prefer to self-describe", "Prefer not to say"), horizontal=True)

        diagnosis = st.multiselect("Diagnosis", known_diseases)

        history = st.text_input("Child's historical information")

        remark = st.text_input("Other comments")


        if st.form_submit_button():
            # fist check that there is Empty but neccessary field
            if (idname == "") or (age == "") or not gender:
                st.warning("The first 3 inputs are necessary")

            else:
                diagnosis_str = ", ".join(diagnosis)
                new_profile = {"identification": idname, 
                                "age": age, 
                                "gender": gender, 
                                "diagnosis": diagnosis_str, 
                                "history": history,
                                "other_remarks": remark
                                }

                saved_profiles.loc[len(saved_profiles)] = new_profile
                saved_profiles.to_csv(excel_profiles_path, index=False)

# Second Tab: Rate an Action
with tab2:
    st.subheader("Action Evaluator")

    if "text_student_profile" not in st.session_state:
        st.session_state.text_student_profile = ""
    if "text_situation" not in st.session_state:
        st.session_state.text_situation = ""
    if "text_action" not in st.session_state:
        st.session_state.text_action = ""

    if "profile_string" not in st.session_state:
            st.session_state.profile_string = ""

    button_left, button_right = st.columns([7, 3])

    with button_left:
        clear_tab2 = st.button('Clear content', key="tab2")
        if clear_tab2:
            st.session_state.text_student_profile = ""
            st.session_state.text_situation = ""
            st.session_state.text_action = ""
            st.session_state.profile_string = ""

    with button_right:
        if st.button("Load saved profile", key="LP_tab2", use_container_width=True):
            profile = choose_profile()
            if profile:
                st.session_state.profile_string = profile
                st.session_state.text_student_profile = profile


    if 'profile_string' in st.session_state and st.session_state.profile_string:
        st.session_state.text_student_profile = st.session_state.profile_string

    if hardcoded:
        text_student_profile = st.text_area("Student Profile:", 
                                            placeholder="Describe the student's profile...", 
                                            key = "text_student_profile", 
                                            height=250,
                                            value="9 years, \nMale, \nADHD, \nHistory of child's diagnosis: The child was noted to be very active and easily distracted from an early age, with difficulties settling into structured play during preschool. Teachers in the early years reported challenges with sustained attention during group activities. By age 7, concerns were raised about inconsistent academic performance, leading to further assessment and a formal ADHD diagnosis at age 8. Parents have described him as bright and curious, but prone to losing focus quickly and acting without thinking, especially in social or learning settings.")
        situation = st.text_area("Situation:", 
                                value="The child sometimes finds it hard to stay focused, which makes it difficult for him to answer questions even when he knows the answer. His performance can be inconsistent, with mistakes from inattention followed by correct answers on harder tasks. He can also act impulsively, for example leaving to get a drink or playing with something during a conversation, which may seem disruptive.",
                                placeholder="Describe the situation...", 
                                # key = 'a',
                                height = 150)
        action = st.text_area("Action:", 
                            value="I plan to let him continue working at his own pace without interruptions. I won’t push him when he gets distracted, but I also won’t provide any specific support or strategies to help him stay engaged.",
                            placeholder="Describe the action to be evaluated...", 
                            # key = "b", 
                            height = 100)
    else:
        text_student_profile = st.text_area("Student Profile:", 
                                        placeholder="Describe the student's profile...", 
                                        key = "text_student_profile", 
                                        height=250)
        situation = st.text_area("Situation:", 
                                placeholder="Describe the situation...", 
                                key = "text_situation", 
                                height = 150)
        action = st.text_area("Action:", 
                            placeholder="Describe the action to be evaluated...", 
                            key = "text_action", 
                            height = 100)    


    # use_knowledge_base = st.checkbox("Use Knowledge Base")

    rate_action_prompt = prompts.rate_action_prompt.format(sp=text_student_profile, s=situation, a=action)

    if st.button("Evaluate Action"):

        if HARDCODED:
            st.info("""Rate: 2 (Ineffective)\n\n
Comment: While allowing the child to work at his own pace is beneficial, the lack of specific support or strategies to help him stay engaged fails to address his ADHD-related challenges effectively. This approach can lead to inconsistent outcomes and may not facilitate improvement in his focus and participation.""")

        else:      
            if st.session_state.text_student_profile and situation and action:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": rate_action_prompt}
                    ]
                ) 

                st.write(response.choices[0].message.content)
            else:
                st.warning("Please fill in all fields before proceeding.")


    st.markdown("""
    Effectiveness Rating System:
    1. Very Ineffective
    2. Ineffective
    3. Neutral
    4. Effective 
    5. Very Effective  
    """)

# Third Tab: Suggest an Action
with tab3:

    st.subheader("Action Recommender")        

    if "text_student_profile_tab2" not in st.session_state:
        st.session_state.text_student_profile_tab2 = ""
    if "text_situation_tab2" not in st.session_state:
        st.session_state.text_situation_tab2 = ""
    if "profile_string" not in st.session_state:
        st.session_state.profile_string = ""

    button_left, button_right = st.columns([7, 3])

    with button_left:
        clear_tab3 = st.button('Clear content', key="tab3")
        if clear_tab3:
            st.session_state.text_student_profile_tab2 = ""
            st.session_state.text_situation_tab2 = ""
            st.session_state.profile_string = ""

    with button_right:
        if st.button("Load saved profile", use_container_width=True, key = 'LP_tab3'):
            profile = choose_profile()
            if profile:
                st.session_state.profile_string = profile
                st.session_state.text_student_profile_tab2 = profile 

    # st.session_state['text_student_profile_tab2'] = st.session_state.profile_string
    if 'profile_string' in st.session_state and st.session_state.profile_string:
        st.session_state.text_student_profile_tab2 = st.session_state.profile_string

    student_profile = st.text_area("Student Profile:",
                                   placeholder="Describe the student's profile...", 
                                   key = "text_student_profile_tab2", 
                                   height=250)
    if hardcoded:
        situation = st.text_area("Situation:",
                                value="The child sometimes finds it hard to stay focused, which makes it difficult for him to answer questions even when he knows the answer. His performance can be inconsistent, with mistakes from inattention followed by correct answers on harder tasks. He can also act impulsively, for example leaving to get a drink or playing with something during a conversation, which may seem disruptive.", 
                                placeholder="Describe the situation...", 
                                key = 'b',
                                height=150)
    else:
        situation = st.text_area("Situation:",
                             placeholder="Describe the situation...", 
                             key = "text_situation_tab2", 
                             height=150)

    if st.button("Suggest an Action"):

        if HARDCODED:
            st.info("""#### **Suggested Action**:

1. Implement a Structured Social Skills Program: Incorporate a social skills program that explicitly teaches the child how to begin and end conversations, attend to other parties in conversations, and manage non-verbal aspects of communication. This should include resources like "Circles of Friends" and "Socially Speaking", with sessions delivered 2x weekly and reinforced throughout the school day by staff knowledgeable about supporting students with Social Communication difficulties.

2. Adopt Task Management Techniques: Break down longer tasks into smaller, more manageable parts, setting time-bound goals for each section. This can help build the child’s confidence and give them a sense of achievement as they complete each segment. These techniques should be incorporated into daily support strategies by classroom staff to help maintain focus and engagement.

3. Enhance Listening and Memory Skills: Daily activities should be included that develop listening and memory skills, such as games like “Simon says” and structured story listening. Additionally, prompts and non-verbal cues should be utilized in class to signal when instructions are being given, assisting the child in staying focused and engaged.
---------------------------------------------------------------
##### **Relevant Information Found in Open-Source Knowledge Base**\n

1. **File Name**: Teacher SEND handbook 30th January 2024 (1).pdf \n
**Content**: This section discusses supporting learners with ADHD, emphasizing understanding executive functioning, reinforcing positive communication of strengths, and highlighting the importance of recognizing their rapid thinking and potential. Understanding ADHD means understanding Executive Functioning and all the information above will be helpful and relevant to adults working with children and young people who have this condition. Clear guidance on what constitutes ADHD can be found in the ADHD Diagnosis area of the NHS website. Relationship Learners with ADHD will surprise you with the rapidity of their thinking and the connections they can make, and frustrate you with the repetition and patience that is required to ensure they keep on track and stay organised. We need to work harder to make sure that we communicate the positives about the work, effort, enthusiasm and energy of these learners, and do not solely focus on what they have not been able to complete. Actively work to list the strengths of learners with ADHD and share these with them.\n
**Page Number**: 187\n
**URL**: doc_not_public_yet\n
                    
2. **File Name**: Teacher SEND handbook 30th January 2024 (1).pdf\n
**Content**: This excerpt explains the typical development and presentation of ADHD symptoms in learners, highlighting how hyperactivity and inattention patterns evolve with age and influence identification and assessment processes. The behaviours associated with ADHD are high activity levels, poor inhibitory control and short attention spans. Hyperactivity is more common in younger learners and evidence shows that it tends to decrease as learners get older, whereas the inattentive type (less movement, but obvious concentration absences) is seen less in pre-schoolers and tends to emerge over the school age period. During the school age period, learners are usually identified and referred for ADHD assessments because of classroom disruptiveness and academic underachievement. The symptoms of inattention increase, and the levels of hyperactivity start to decrease. The fidgety behaviours calm but the brain is cognitively hyperactive, increasingly in need of satiation, jumping from one thing to another.\n
**Page Number**: 187\n
**URL**: doc_not_public_yet\n
                    
---------------------------------------------------------------
                    
##### **Relevant Information Found in Expert Knowledge Base**
                    
1. **File Name**: Anon case 2.docx\n
**Content**: Special Educational Needs: He struggles to maintain his attention and so may be unable to respond to questions that are well within his ability. During formal testing there were many occasions when his performance was erratic, making errors through apparent inattention followed by a correct response to a seemingly more difficult task. He displays impulsive patterns of behaviour which break interaction and may seem rude, for example, standing up and going to get a drink or play with something that has taken his attention in the middle of a conversation. Suggested Outcome(s): By the end of Key Stage 2 (July 2026) Outcome 1: He will attend to questions from a conversational partner and engage in at least 2 related exchanges on 80% of occasions when communication with him is initiated by an adult or peer. Outcome 2: He will develop his ability to end interactions in socially acceptable ways, such as explaining his reason for leaving or making concluding remarks on 80% of occasions.\n
**Table in file**: 0\n
**Source**: Case study by Educational Psychologist\n

2. **File Name**: Anon case 2.docx\n
**Content**: He displays difficulties with attention control and concentration across contexts. He acts impulsively to respond to something that has taken his attention, to fulfil an internal drive such as a sensation of thirst or to provide himself with stimulation, for example, discontinuing a successful strategy for completing a task, just to see if he would still be able to succeed without using this approach. He displays low academic self-esteem, referring repeatedly to himself as ‘dumb’, commenting that ‘my brain is very small’ and using the rating ‘never’ for the description, ‘I am a good thinker’.\n
**Table in file**: 1\n
**Source**: Case study by Educational Psychologist"""
)

        else:

            if student_profile and situation:

                query = student_profile + " " + situation

                retrieved_chunks = knowledge_base.query_knowledge_base(query)
                chunks_prompt = knowledge_base.format_chunks(retrieved_chunks)

                retrieved_chunks_cs = knowledge_base.hybrid_search_cs(query)
                chunks_prompt_cs = knowledge_base.format_chunks_cs(retrieved_chunks_cs)

                if DEV:
                    print_out_chunks(chunks_prompt, chunks_prompt_cs)

                suggest_action_kb_prompt = prompts.suggest_action_prompt.format(sp=student_profile, s=situation, a=action, chunks=chunks_prompt, case_studies=chunks_prompt_cs)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": suggest_action_kb_prompt}
                    ]
                )
                suggested_action = response.choices[0].message.content

                st.success(f"Suggested Action: {suggested_action}")

            else:
                st.warning("Please fill in the student profile and situation before proceeding.")

        

with tab4:
        st.subheader("Provide a solution to the following scenario:")

        def load_scenario():
            if HARDCODED:
                scenario_path = os.path.join("data", "new_synthetic_scenarios_v2", "scenario_0047.json")
            else:
                scenario_path = knowledge_base.get_rand_scenario_high_grade()
            with open(scenario_path, 'r') as file:
                st.session_state.data = json.load(file)

            # st.session_state.text_reaction_test = ""  
            st.session_state.text_reaction_true = ""

            st.session_state.text_situation_gen = f""" 
**Age of student:** {st.session_state.data['student_profile']['age']}\n
**Gender:** {st.session_state.data['student_profile']['gender']}\n
**Conditions:** {', '.join(st.session_state.data['student_profile']["diagnosis"])}\n
**Situation:** {st.session_state.data['situation']}"""


        if "data" not in st.session_state:
            load_scenario()            
        
        if st.button("Load Training Scenario"):
            load_scenario()

        st.markdown("###### Scenario Description:")
        st.markdown(f"""
                <div style="border: 1px solid #ccc;
                     padding: 10px;
                     border-radius: 5px;
                     background-color: #f8f9fa;
                     color: black;
                     font-size: 16px;">
                    {st.session_state.text_situation_gen}
                </div>
            """, unsafe_allow_html=True)

        if hardcoded:
            reaction = "I will remind the group to speak clearly and at a pace Sarah can follow, and I’ll check in with her to make sure she feels included in the discussion. This way, she can participate more comfortably alongside her peers."
        else:
            reaction = None

        st.markdown("###### What would you do?")
        text_area_react_test = st.text_area(label = "text_area_react", 
                                            value = reaction, 
                                            height=100,  
                                            key="text_reaction_test",
                                            label_visibility='collapsed')  
    
        
        if st.button("Submit"):
            # if text_area_react_test:
            st.session_state.text_reaction_true = st.session_state.data['action']
            # else:
                # st.warning("Please fill in all fields before proceeding.")


        st.markdown("###### Action Suggested by Specialist:")
        st.markdown(f"""
                <div style="border: 1px solid #ccc;
                     padding: 10px;
                     border-radius: 5px;
                     background-color: #f8f9fa;
                     color: black;
                     font-size: 16px;">
                    {st.session_state.text_reaction_true}
                </div>
            """, unsafe_allow_html=True)
        
        
                                
with tab5:
    # This function get a string and displays it with some delay per word
    def stream_response(response_text, speed=0.02):
        def generator():
            for token in response_text.split(" "):  # stream word by word
                yield token + " "
                time.sleep(speed)
        st.write_stream(generator())
        
    st.subheader("Conversational Assistant")
    
    final_summary = []

    if "messages" not in st.session_state:
        chat_bot = agent_interface.ChatBot()
        start_message_history = chat_bot.start_chat()
        st.session_state['messages'] = start_message_history
        st.session_state['profile_string'] = ""
        st.session_state['profile_selected'] = False
        st.session_state['profile_announced'] = False
        st.session_state['chat_bot'] = chat_bot

    button_left, button_right = st.columns([7, 3])

    with button_left:

        if DEV:
            if st.button("🔄 Start New Chat"):
                del st.session_state['messages']
                del st.session_state['profile_string']
                del st.session_state["profile_selected"]
                del st.session_state['profile_announced']
                del st.session_state['chat_bot']

                st.rerun()   

            if st.button("Start Chat with Navigator"): 
                # initialise a different chatbot
                chat_bot = agent_interface.ChatBot("navigation_helper")
                start_message_history = chat_bot.start_chat()
                st.session_state['messages'] = start_message_history
                st.session_state['profile_string'] = ""
                st.session_state['profile_selected'] = False
                st.session_state['profile_announced'] = False
                st.session_state['chat_bot'] = chat_bot


        else: 
            if st.button("🔄 Start New Chat"):
                del st.session_state['messages']
                del st.session_state['profile_string']
                del st.session_state["profile_selected"]
                del st.session_state['profile_announced']
                del st.session_state['chat_bot']

                st.rerun()   

        if DEV:
            if st.button("Inspect Message History"):
                inspect_chat_dialogue()

    with button_right:
        if st.button("Load saved profile", use_container_width=True, key='assistant_load'): 
            profile = choose_profile()

    if st.session_state.get("profile_selected") and not st.session_state.get("profile_announced"):
        profile_text = st.session_state['profile_string']

        profile_msg = st.session_state.chat_bot.load_send_profile(profile_text)[0]

        st.session_state.messages.append(profile_msg)
        st.session_state["profile_announced"] = True


    #: BG - 18 years old; EN - 25 years old More actions
    #  and note that you work with children
    # - Are there communication and interaction problems? Are there Cognitive impairments? Are there social, emotional or mental health problems?
                # - Are there special sensory and/or physical needs? Are there specific medical or neurological needs?

    # - Ask at least 2 clarifying questions for each symptom to get enough detail.

        # Do not ask about age, gender, or diagnosis again.
        # Give a summary of what you have learned from the child's profile information.
        # Continue with the rest of the questions in the guide.

    # TODO: Give a suggestion directly after if the user gives a summary 

    
    chat_container = st.container()
            
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # User input box
    user_input = st.chat_input("Write a message...")

    if user_input:               
        
        # TODO fix the moderation
        # flagged, categories, category_scores = moderate_text(user_input)
        
        #if flagged:
        #        st.warning("⚠️ Вашето съобщение беше маркирано като неподходящо. Достъпът ви до чата е ограничен.")
        #        # st.write(category_scores)
        #        st.stop()  # Stop further execution
        #print(flagged)

        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
    
        # Append user message
        user_message = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_message)

        bot_responses = st.session_state.chat_bot.send_message(user_message)
        
        # If the bost responds with summary and suggestion depict the suggestion differently (don't show the response) 
        # This only occurs if the response are 2 messages

        with chat_container:
            response = bot_responses[-1]
            st.session_state.messages.append(response)

            with st.chat_message("assistant"):
                        stream_response(response['content'], speed=0.05)



if DEV:
    st.sidebar.title("Sidebar Menu")

    cpy_paste_texts = [
        "the child is my student and started slamming his book onto his head during reading class. I didn't notice any sudden events that could have triggered this. It was the first time he started acting like that. I have no to react properly in such a situation. it happened in the usual calssroom environment. It was reading session just before lunch and he was supposed to read silently as were the other 20 pupils",
        "i would love to send my child to a specialist with expertise in dealing with young children on the autistic spectrum. i live in sofia are there any good one here?",
    ]

    for text in cpy_paste_texts:
        st.sidebar.code(text, language=None, wrap_lines=True)

                #### **Suggested Action**
                
            #     1. Step 1...
            #     2. Step 2...
            #     3. Step 3...

            #     -------------------------------------------------------------
            #     2. After the solution, provide two separate lists in the following formats:

            #     - #### Relevant Information Found in **Open-Source Knowledge Base**
            #         - File Name
            #             - Content
            #             - Page Number
            #             - URL
                    
            #         (ignore the score)

            #     (If no suitable chunks are used: "No suitable chunks have been found from this knowledge source.")
            #     -------------------------------------------------------------
            #     - #### Relevant Information Found in **Expert Knowledge Base**
            #         - File Name
            #             - Content
            #             - Table in file
            #             - Source
                    
            #         (ignore the score)

            #     (If no suitable chunks are used: "No suitable chunks have been found from this knowledge source.")

            #     You must only base your answer on the provided chunks. 
            #     Do not use outside knowledge.
            #     """

            #     suggestion = chatbot_response([{"role": "system", "content": summary_ready_prompt}])
            #     with chat_container:
            #         with st.chat_message("assistant"):
            #             st.info(suggestion)
            # else:
            #     with chat_container:
            #         with st.chat_message("assistant"):
            #             stream_response(bot_response, speed = 0.05)
        
