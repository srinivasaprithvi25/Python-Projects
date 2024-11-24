import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load raw data
def load_raw_data():
    data = {
        'text': [
            # Mood and Emotions
            "Pretty balanced – work, some exercise, and time to unwind in the evening.",  # happy
            "It’s kind of all over the place, honestly. Hard to keep a routine.",         # sad
            "I really enjoy reading or spending time with friends after work.",           # happy
            "Lately, I’m not sure if anything stands out, just going through the motions.",  # sad
            "Energized and ready to tackle things.",                                      # happy
            "Drained or indifferent – it’s hard to get going.",                           # sad

            # Coping and Stress
            "I take a deep breath, assess, and try to adapt.",                             # content
            "I tend to feel overwhelmed and it’s hard to move past it.",                   # anxious
            "Listening to music or going for a short walk helps me unwind.",               # content
            "Mostly just scrolling online for hours or binge-watching shows.",             # sad
            "Not often; I have a pretty solid evening routine.",                           # happy
            "Yes, my mind feels restless even when I’m tired.",                            # anxious

            # Sleep Quality and Routine
            "Most days, I get a good amount of sleep.",                                    # happy
            "Hardly ever – I feel tired no matter how much I sleep.",                      # sad
            "Fairly steady with maybe a slight dip in the afternoon.",                     # content
            "Up and down a lot, but mostly low energy lately.",                            # sad
            "Around 10 PM to 6:30 AM – I stick to a routine.",                             # happy
            "It’s different each day – sometimes late, sometimes early.",                  # anxious

            # Social Connections and Support
            "Almost daily – it’s nice to stay in touch.",                                  # happy
            "Maybe once or twice; I’ve been a bit distant.",                               # sad
            "My close friends and family – they’re always there.",                         # happy
            "I’m not sure. I don’t really have anyone specific.",                          # sad
            "Quite often; I feel connected and supported.",                                # happy
            "Not very often – I feel a bit misunderstood lately.",                         # sad

            # Self-Perception and Self-Esteem
            "Proud – I feel like I’m making progress.",                                    # happy
            "They don’t feel like much; it’s hard to feel proud.",                         # sad
            "Positive, driven, and curious.",                                              # happy
            "Stressed, unsure, and often tired.",                                          # sad
            "I enjoy my own company and reflection time.",                                 # content
            "I find it challenging; it brings up too much worry.",                         # anxious

            # Motivation and Outlook
            "Learning new skills at work and maybe planning a trip.",                      # happy
            "I’m not sure. Haven’t felt much excitement lately.",                          # sad
            "Motivated to find solutions, even if it’s difficult.",                        # content
            "I often feel stuck and doubt my ability to handle them.",                     # anxious
            "Yes, I’ve been wanting to start hiking or painting.",                         # happy
            "Nothing really comes to mind; I just don’t feel up for it.",                  # sad

            # Physical Health and Lifestyle
            "A few times a week, I like to stay active when I can.",                       # content
            "Rarely – I’ve been feeling sluggish.",                                        # sad
            "I focus on healthy eating, exercise, and setting aside time for myself.",     # content
            "I don’t do much; it’s hard to focus on myself these days.",                   # sad
        ],
        'label': [
            'happy', 'sad', 'happy', 'sad', 'happy', 'sad',        # Mood and Emotions
            'content', 'anxious', 'content', 'sad', 'happy', 'anxious',  # Coping and Stress
            'happy', 'sad', 'content', 'sad', 'happy', 'anxious',        # Sleep Quality and Routine
            'happy', 'sad', 'happy', 'sad', 'happy', 'sad',        # Social Connections and Support
            'happy', 'sad', 'happy', 'sad', 'content', 'anxious',  # Self-Perception and Self-Esteem
            'happy', 'sad', 'content', 'anxious', 'happy', 'sad',  # Motivation and Outlook
            'content', 'sad', 'content', 'sad',                    # Physical Health and Lifestyle
        ]
    }
    return pd.DataFrame(data)

# Function to preprocess text data
def preprocess_text(df):
    # Example preprocessing: lowercase text
    df['text'] = df['text'].str.lower()
    return df

# Function to split and save dataset
def split_and_save_data(df):
    # Split dataset into training and testing sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create directories if they don't exist
    processed_data_dir = 'data/processed'
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Save the datasets
    train.to_csv(os.path.join(processed_data_dir, 'mental_health_train.csv'), index=False)
    test.to_csv(os.path.join(processed_data_dir, 'mental_health_test.csv'), index=False)
    print("Datasets saved successfully.")

def main():
    # Load raw data
    raw_data = load_raw_data()
    
    # Preprocess data
    processed_data = preprocess_text(raw_data)
    
    # Split and save data
    split_and_save_data(processed_data)

if __name__ == "__main__":
    main()
